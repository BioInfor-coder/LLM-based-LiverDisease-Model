# -*- coding: utf-8 -*-

# 标准库
import argparse
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 第三方库
import mysql.connector
import pandas as pd


# ============================================================
# 全局常量（原脚本中的魔法数字统一在此声明）
# ============================================================
NEGATIVE_RESULT_FLAG = -1          # 检验结果缺失/未查的占位值
MISSING_RATIO_THRESHOLD = 0.6      # 阶段四：非缺失检验项占比阈值
MIN_ADULT_AGE = 18                 # 阶段四：成年判定年龄下限
MERGED_JSON_SUFFIX = "_merged.json"  # 阶段二起统一使用的合并文件后缀


# ============================================================
# 阶段一：从 MySQL 数据库提取单患者结构化 JSON
# ============================================================
def _json_datetime_default(obj: Any) -> str:
    """
    json.dump 的自定义序列化函数，用于处理 datetime 类型字段

    参数:
        obj: 待序列化对象

    返回值:
        格式化后的时间字符串

    异常:
        TypeError: 当对象既不是 datetime 也无法序列化时
    """
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def query_patient_record(cursor: Any, patient_id: str, lab_table_config: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    根据患者编号查询各业务表并汇总为单患者结构化字典

    参数:
        cursor: mysql-connector 的 dictionary cursor
        patient_id: 患者编号
        lab_table_config: 患者编号 -> 检验结果表名列表 的映射（来自 config.json）

    返回值:
        包含病理、超声、放射、胃镜、诊断、检验、主诉等字段的字典
    """
    record: Dict[str, Any] = {
        "患者编号": patient_id,
        "病理": [],
        "超声": [],
        "放射": [],
        "胃镜": [],
        "诊断": [],
        "检验": [],
        "主诉": [],
    }

    # 1. 病理 / 超声 / 放射 / 胃镜：结构一致，逐表查询
    simple_table_map = {
        "病理": ("bingli_result", "患者ID", "标本接收时间"),
        "超声": ("chaosheng_result", "患者编号", "检查时间"),
        "放射": ("fangshe_result", "患者编号", "检查时间"),
        "胃镜": ("weijing_result", "患者编号", "检查时间"),
    }
    for field, (table, id_col, order_col) in simple_table_map.items():
        query = f"SELECT * FROM {table} WHERE {id_col} = %s ORDER BY {order_col};"
        cursor.execute(query, (patient_id,))
        record[field] = cursor.fetchall()

    # 2. 旧诊断信息（多张历史表拼接后统一改名为标准字段）
    legacy_diagnosis_tables = ["shuju2010-2015", "shuju2016-2020", "shuju2016-2020", "shuju2023-2025"]
    legacy_rows: List[Dict[str, Any]] = []
    for table in legacy_diagnosis_tables:
        query = f"SELECT * FROM `{table}` WHERE 患者编号 = %s ORDER BY 就诊日期;"
        cursor.execute(query, (patient_id,))
        legacy_rows.extend(cursor.fetchall())

    for row in legacy_rows:
        record["诊断"].append({
            "就诊日期": row["就诊日期"],
            "患者编号": row["患者编号"],
            "患者身份证号": row["身份证号"],
            "姓名": row["姓名"],
            "诊疗科室": row["就诊科室"],
            "医保类型": "",
            "性别": row["性别"],
            "医生姓名": row["就诊医生"],
            "就诊流水号": row["就诊流水号"],
            "诊断名称": row["诊断内容"],
            "医嘱名称": row["医嘱内容"],
        })

    # 3. 新诊断系统信息
    query = "SELECT * FROM `shujuxinxitong` WHERE 患者编号 = %s ORDER BY 就诊日期;"
    cursor.execute(query, (patient_id,))
    record["诊断"].extend(cursor.fetchall())

    # 4. 主诉信息
    query = "SELECT * FROM `zhusu_result` WHERE 患者编号 = %s ORDER BY 创建时间;"
    cursor.execute(query, (patient_id,))
    record["主诉"] = cursor.fetchall()

    # 5. 检验信息：需按 config.json 中登记的表名逐一查询
    formatted_key = str(patient_id).zfill(10)
    lab_tables = lab_table_config.get(formatted_key)
    if lab_tables:
        for table in lab_tables:
            query = f"SELECT * FROM {table} WHERE 患者编号 = %s ORDER BY 检验日期;"
            cursor.execute(query, (patient_id,))
            record["检验"].extend(cursor.fetchall())

    return record


def extract_patient_jsons(
    db_config: Dict[str, Any],
    lab_table_config: Dict[str, List[str]],
    patient_ids: List[str],
    save_dir: Path,
) -> None:
    """
    批量从数据库提取患者数据并逐一保存为单患者 JSON 文件

    参数:
        db_config: mysql-connector 连接参数（host/port/user/password/database）
        lab_table_config: 患者编号 -> 检验表名列表 的映射
        patient_ids: 待提取的患者编号列表
        save_dir: JSON 输出目录，单患者文件命名为 `{patient_id}.json`

    返回值:
        无（副作用为写文件）

    异常:
        mysql.connector.Error: 数据库连接或查询失败时
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True, buffered=False)  # buffered=False 流式读取，降低内存占用

    try:
        for patient_id in patient_ids:
            out_path = save_dir / f"{patient_id}.json"
            if out_path.exists():
                continue  # 断点续跑：已提取过的患者直接跳过

            record = query_patient_record(cursor, patient_id, lab_table_config)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([record], f, ensure_ascii=False, indent=4, default=_json_datetime_default)
            print(f"[阶段一] {datetime.datetime.now()} 已提取患者 {patient_id}")
    finally:
        cursor.close()
        conn.close()

    print(f"✅ [阶段一] 数据库提取完成，输出目录：{save_dir}")


# ============================================================
# 共享工具：从「检验」记录列表中提取年龄 / 性别
# ============================================================
def extract_age_gender_from_labs(lab_records: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    从检验记录列表中提取首条包含年龄/性别的记录

    参数:
        lab_records: 「检验」字段对应的记录列表

    返回值:
        (age, gender) 二元组，若均未找到则为 (None, None)
    """
    age, gender = None, None
    for rec in lab_records:
        age = rec.get("年龄") or rec.get(" 年龄 ") or rec.get("\u5e74\u9f84")
        gender = rec.get("性别") or rec.get(" 性别 ") or rec.get("\u6027\u5225")
        if age is not None:
            break
    return age, gender


# ============================================================
# 阶段二：筛除「检验」字段为空的样本
# ============================================================
def keep_non_empty_lab(src_file: Path, dst_file: Path) -> bool:
    """
    仅当「检验」列表非空时，将样本写出到目标路径

    参数:
        src_file: 输入 JSON 文件路径
        dst_file: 输出 JSON 文件路径

    返回值:
        True 表示保留并写出，False 表示跳过
    """
    data = json.loads(src_file.read_text(encoding="utf-8"))
    if data.get("检验"):
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        dst_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False


def filter_empty_lab_batch(merged_dir: Path, filtered_dir: Path, file_pattern: str) -> None:
    """
    批量筛选：仅保留检验非空的样本文件

    参数:
        merged_dir: 输入目录（阶段一输出或人工合并后的目录）
        filtered_dir: 输出目录
        file_pattern: glob 匹配模式，如 "*.json" 或 "*_merged.json"

    返回值:
        无
    """
    filtered_dir.mkdir(parents=True, exist_ok=True)
    for fp in merged_dir.rglob(file_pattern):
        out_fp = filtered_dir / fp.relative_to(merged_dir)
        if keep_non_empty_lab(fp, out_fp):
            print(f"[阶段二] ✅ 保留：{fp.name}")
        else:
            print(f"[阶段二] 🚫 跳过（检验为空）：{fp.name}")


# ============================================================
# 阶段三：删除「检验」结果全部为 -1（占位值）的样本
# ============================================================
def is_all_negative(test_items: List[Dict[str, Any]]) -> bool:
    """
    判断检验记录是否全部为占位负值

    参数:
        test_items: 「检验」字段对应的记录列表，每项应为 {"项目名称":..., "结果":...}

    返回值:
        True 表示全部为占位值（应删除），False 表示至少一项有效
    """
    for item in test_items:
        if not isinstance(item, dict) or item.get("结果") != NEGATIVE_RESULT_FLAG:
            return False
    return True


def remove_all_negative_samples(json_folder: Path) -> None:
    """
    删除文件夹内所有「检验」结果全部为 -1 的 JSON 文件（原地删除）

    参数:
        json_folder: JSON 文件所在文件夹路径

    返回值:
        无
    """
    if not json_folder.is_dir():
        print(f"[阶段三] ❌ 路径不存在：{json_folder}")
        return

    print(f"[阶段三] 🔍 正在检查文件夹：{json_folder}")
    for fp in json_folder.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[阶段三] ❌ 无法读取 {fp.name}：{e}")
            continue

        test_items = data.get("检验")
        if not isinstance(test_items, list):
            print(f"[阶段三] 跳过（检验字段缺失或不是列表）：{fp.name}")
            continue

        if is_all_negative(test_items):
            fp.unlink()
            print(f"[阶段三] 🗑️ 删除：{fp.name}（所有结果均为 -1）")
        else:
            print(f"[阶段三] ✅ 保留：{fp.name}")

    print("[阶段三] ✅ 清理完成！")


# ============================================================
# 阶段四：按检验缺失率与年龄进一步筛选，并输出统计 CSV
# ============================================================
def compute_non_negative_ratio(test_items: List[Dict[str, Any]]) -> Tuple[int, int, int, float]:
    """
    统计检验记录中非占位值（有效结果）的数量与占比

    参数:
        test_items: 「检验」字段对应的记录列表

    返回值:
        (检验项目总数, 结果为-1数量, 结果非-1数量, 结果非-1占比) 四元组
    """
    total = len(test_items)
    neg_count = sum(1 for item in test_items if isinstance(item, dict) and item.get("结果") == NEGATIVE_RESULT_FLAG)
    non_neg_count = total - neg_count
    non_neg_ratio = (non_neg_count / total) if total > 0 else 0.0
    return total, neg_count, non_neg_count, non_neg_ratio


def filter_by_missing_ratio_and_age(
    json_folder: Path,
    output_csv: Path,
    filtered_folder: Path,
    threshold: float = MISSING_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """
    统计各样本检验缺失情况，保留“非缺失占比 > threshold 且成年”的样本

    参数:
        json_folder: 输入 JSON 文件夹（阶段三输出）
        output_csv: 统计结果输出 CSV 路径
        filtered_folder: 符合条件的样本转存目录
        threshold: 非 -1 占比阈值，默认 MISSING_RATIO_THRESHOLD

    返回值:
        每个样本检验统计信息的 DataFrame
    """
    if not json_folder.is_dir():
        print(f"[阶段四] ❌ 路径不存在：{json_folder}")
        return pd.DataFrame()

    filtered_folder.mkdir(parents=True, exist_ok=True)
    print(f"[阶段四] 🔍 正在统计文件夹：{json_folder}")

    result_rows: List[Dict[str, Any]] = []

    for fp in json_folder.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[阶段四] ❌ 无法读取 {fp.name}：{e}")
            continue

        test_items = data.get("检验")
        if not isinstance(test_items, list):
            print(f"[阶段四] 跳过（检验字段缺失或不是列表）：{fp.name}")
            continue

        total, neg_count, non_neg_count, non_neg_ratio = compute_non_negative_ratio(test_items)
        result_rows.append({
            "文件名": fp.name,
            "检验项目总数": total,
            "结果为 -1 数量": neg_count,
            "结果非 -1 数量": non_neg_count,
            "结果非 -1 占比": non_neg_ratio,
        })

        # 检验记录为空则本样本无法判断年龄，跳过（原脚本此处误用 return False
        # 会导致整个批处理提前退出，此处修正为 continue，仅跳过当前文件）
        if not test_items:
            continue

        age, _ = extract_age_gender_from_labs(test_items)
        if age is None:
            print(f"[阶段四] 跳过（无法提取年龄）：{fp.name}")
            continue

        try:
            age_value = int(age)
        except (ValueError, TypeError):
            print(f"[阶段四] 跳过（年龄字段无法解析）：{fp.name}")
            continue

        # ✅ 非 -1 占比 > threshold 且年龄 >= 成年下限，转存文件
        if non_neg_ratio > threshold and age_value >= MIN_ADULT_AGE:
            shutil.copy(fp, filtered_folder / fp.name)
            print(f"[阶段四] ✅ 已转存：{fp.name}（非 -1 占比={non_neg_ratio:.2%}）")

    df = pd.DataFrame(result_rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[阶段四] ✅ 统计完成！CSV 保存到：{output_csv}")
    print(f"[阶段四] ✅ 过滤后文件保存到：{filtered_folder}")
    return df


# ============================================================
# 阶段五：JSON 转自然语言文本 + 检验项宽表汇总
# ============================================================
def _clean_text(txt: Any) -> str:
    """
    清洗文本：去除换行/制表符/星号等噪声字符并压缩空白

    参数:
        txt: 任意可转为字符串的原始文本

    返回值:
        清洗后的字符串，输入为空时返回空字符串
    """
    if not txt:
        return ""
    return " ".join(
        str(txt)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("★", "")
        .split()
    )


def json_to_nl(data: Dict[str, Any]) -> str:
    """
    将单患者结构化 JSON 转换为大模型输入用的自然语言描述

    参数:
        data: 单患者结构化字典（含病理/超声/胃镜/放射/检验字段）

    返回值:
        拼接后的自然语言字符串；若无有效检查记录则返回提示文本
    """
    pieces: List[str] = []

    # 1. 病理 / 2. 超声 / 3. 胃镜 / 4. 放射：结构一致，逐项拼接
    section_map = [("病理", "病理提示"), ("超声", "超声提示"), ("胃镜", "胃镜提示"), ("放射", "放射提示")]
    for field, label in section_map:
        section = data.get(field)
        if section and section.get("检查结论" if field != "病理" else "病理诊断结论"):
            key = "检查结论" if field != "病理" else "病理诊断结论"
            pieces.append(f"{label}：{_clean_text(section[key])}")

    # 5. 实验室检查（含年龄、性别）
    labs = data.get("检验")
    if labs:
        age, gender = extract_age_gender_from_labs(labs)
        pieces.append(f"年龄：{age}")
        pieces.append(f"性别：{gender}")

        lab_desc = [f"{_clean_text(item.get('项目名称', ''))} {_clean_text(item.get('结果', ''))}" for item in labs]
        if lab_desc:
            pieces.append("实验室检查：" + "；".join(lab_desc))

    return "；".join(pieces) or "无有效检查记录"


def merge_all_to_nl(root_dirs: List[Path], class_labels: List[int], save_dir: Path, file_pattern: str) -> None:
    """
    批量将各疾病目录下的样本 JSON 转为自然语言文本，并生成对应标签文件

    参数:
        root_dirs: 各疾病样本所在目录列表（与 class_labels 一一对应）
        class_labels: 各目录对应的类别标签
        save_dir: 输出目录，生成 total.txt（文本）与 labels.txt（标签）
        file_pattern: glob 匹配模式，如 "*.json" 或 "*_merged.json"

    返回值:
        无
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    total_f = save_dir / "total.txt"
    label_f = save_dir / "labels.txt"

    with open(total_f, "w", encoding="utf-8") as f_txt, open(label_f, "w", encoding="utf-8") as f_lbl:
        for label, folder in zip(class_labels, root_dirs):
            for fp in folder.rglob(file_pattern):
                nl = json_to_nl(json.loads(fp.read_text(encoding="utf-8")))
                f_txt.write(nl + "\n")
                f_lbl.write(f"{label}\n")

    print(f"[阶段五] ✅ 自然语言合并完成！共生成：\n  {total_f}\n  {label_f}")


def create_lab_wide_table(root_dirs: List[Path], class_labels: List[int], save_path: Path, file_pattern: str) -> pd.DataFrame:
    """
    汇总所有检验项与各类检查结论，生成宽表并保存为 Excel

    参数:
        root_dirs: 各疾病样本所在目录列表（与 class_labels 一一对应）
        class_labels: 各目录对应的类别标签
        save_path: 输出 Excel 文件路径
        file_pattern: glob 匹配模式，如 "*.json" 或 "*_merged.json"

    返回值:
        汇总后的宽表 DataFrame
    """
    rows: List[Dict[str, Any]] = []
    all_columns: set = set()

    for label, folder in zip(class_labels, root_dirs):
        source_name = folder.name

        for fp in folder.rglob(file_pattern):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            row: Dict[str, Any] = {
                "样本来源": source_name,
                "文件名": fp.name,
                "完整路径": str(fp),
                "疾病标签": label,
            }

            # 各类检查结论
            conclusion_map = [("病理", "病理诊断结论", "病理诊断结论"), ("超声", "检查结论", "超声结论"),
                               ("胃镜", "检查结论", "胃镜结论"), ("放射", "检查结论", "放射结论")]
            for field, src_key, dst_key in conclusion_map:
                section = data.get(field)
                row[dst_key] = section.get(src_key, "") if section else ""
                all_columns.add(dst_key)

            # 检验项目（宽表展开），仅取首条记录的年龄、性别
            labs = data.get("检验", [])
            if isinstance(labs, list) and labs:
                for idx, item in enumerate(labs):
                    name = item.get("项目名称")
                    if name:
                        row[name] = item.get("结果")
                        all_columns.add(name)
                    if idx == 0:
                        row["年龄"] = item.get("年龄")
                        row["性别"] = item.get("性别")

            rows.append(row)

    df = pd.DataFrame(rows)
    front_cols = ["样本来源", "文件名", "完整路径", "疾病标签", "年龄", "性别"]
    other_cols = sorted(c for c in all_columns if c not in front_cols)
    df = df.reindex(columns=front_cols + other_cols)

    df.to_excel(save_path, index=False)
    print(f"[阶段五] ✅ 已保存宽表：{save_path}")
    print(f"[阶段五] 共 {df.shape[0]} 个样本，{len(other_cols)} 个检查项目列")
    return df


# ============================================================
# 主流程：串联五个阶段
# ============================================================
def run_pipeline(
    db_config: Dict[str, Any],
    lab_table_config_path: Path,
    disease_patient_csv: Dict[str, Path],
    work_root: Path,
    class_label_map: Dict[str, int],
    missing_ratio_threshold: float = MISSING_RATIO_THRESHOLD,
    run_db_extraction: bool = True,
) -> None:
    """
    按疾病批量执行「数据库提取 -> 三级筛选 -> 自然语言/宽表转换」完整流程

    参数:
        db_config: MySQL 连接参数
        lab_table_config_path: 患者编号->检验表名映射的 config.json 路径
        disease_patient_csv: 疾病名 -> 患者编号 CSV 路径 的映射（阶段一输入）
        work_root: 工作根目录，各阶段中间产物均在此目录下按疾病建子目录
        class_label_map: 疾病名 -> 类别标签 的映射（如 {"AIH": 0, "PBC": 1, ...}）
        missing_ratio_threshold: 阶段四的非缺失占比阈值
        run_db_extraction: 是否执行阶段一数据库提取；若样本已在本地，
            可设为 False 直接从阶段二开始

    返回值:
        无

    示例:
        >>> run_pipeline(
        ...     db_config={"user": "root", "password": "***", "host": "192.168.13.16",
        ...                "port": 3330, "database": "youan", "raise_on_warnings": True},
        ...     lab_table_config_path=Path("config.json"),
        ...     disease_patient_csv={"AIH": Path("hit_自身免疫性肝炎_earliest.csv")},
        ...     work_root=Path("/home/LLM/YA_LLM_data_v2"),
        ...     class_label_map={"AIH": 0, "PBC": 1, "DILI": 2, "CHB": 3},
        ... )
    """
    with open(lab_table_config_path, "r", encoding="utf-8") as f:
        lab_table_config = json.load(f)

    disease_names = list(class_label_map.keys())
    filtered_dirs: List[Path] = []   # 收集阶段四输出目录，供阶段五使用
    class_labels: List[int] = []

    for disease in disease_names:
        raw_dir = work_root / f"{disease}_raw"
        nonempty_dir = work_root / f"{disease}_filtered"
        threshold_stats_csv = work_root / f"{disease}_stats.csv"
        cleaned_dir = work_root / f"{disease}_cleaned"

        # ---------- 阶段一：数据库提取 ----------
        if run_db_extraction:
            patient_ids = pd.read_csv(disease_patient_csv[disease], usecols=["患者编号"])["患者编号"].tolist()
            extract_patient_jsons(db_config, lab_table_config, patient_ids, raw_dir)

        # ---------- 阶段二：筛除检验为空的样本 ----------
        filter_empty_lab_batch(raw_dir, nonempty_dir, file_pattern="*.json")

        # ---------- 阶段三：删除检验全为 -1 的样本（原地过滤） ----------
        remove_all_negative_samples(nonempty_dir)

        # ---------- 阶段四：按缺失率与年龄筛选 ----------
        filter_by_missing_ratio_and_age(
            nonempty_dir, threshold_stats_csv, cleaned_dir, threshold=missing_ratio_threshold
        )

        filtered_dirs.append(cleaned_dir)
        class_labels.append(class_label_map[disease])

    # ---------- 阶段五：自然语言转换 + 宽表汇总 ----------
    nl_save_dir = work_root / f"total_nl_labels_0.2_{missing_ratio_threshold}"
    merge_all_to_nl(filtered_dirs, class_labels, nl_save_dir, file_pattern="*.json")

    wide_table_path = work_root / f"lab_results_df_0.2_{missing_ratio_threshold}.xlsx"
    create_lab_wide_table(filtered_dirs, class_labels, wide_table_path, file_pattern="*.json")

    print("\n✅ 全部流程执行完成！")


def _parse_key_value_list(pairs: List[str], value_caster: Any = str) -> Dict[str, Any]:
    """
    将形如 ["AIH=0", "PBC=1"] 的命令行参数列表解析为字典

    参数:
        pairs: "键=值" 格式的字符串列表
        value_caster: 值的类型转换函数，默认原样保留为字符串

    返回值:
        解析后的 {键: 转换后的值} 字典

    异常:
        ValueError: 当某一项不包含 "=" 分隔符时
    """
    result: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"参数格式错误，应为 键=值，实际收到：{pair}")
        key, value = pair.split("=", 1)
        result[key] = value_caster(value)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器

    参数:
        无

    返回值:
        配置好的 ArgumentParser 对象
    """
    parser = argparse.ArgumentParser(
        description="肝病多病种患者数据完整处理流程（数据库提取 -> 三级筛选 -> 自然语言/宽表转换）"
    )

    parser.add_argument("--work-root", required=True, type=Path,
                         help="工作根目录，各阶段中间产物均在此目录下按疾病建子目录")
    parser.add_argument("--lab-table-config", required=True, type=Path,
                         help="患者编号->检验表名映射的 config.json 路径")

    parser.add_argument("--class-labels", required=True, nargs="+", metavar="疾病名=标签",
                         help="疾病名与类别标签的映射，如 AIH=0 PBC=1 DILI=2 CHB=3")
    parser.add_argument("--disease-csv", nargs="*", default=[], metavar="疾病名=CSV路径",
                         help="疾病名与患者编号 CSV 路径的映射（仅数据库提取阶段需要），"
                              "如 AIH=hit_AIH_earliest.csv")

    parser.add_argument("--db-host", default=None, help="MySQL 主机地址")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL 端口，默认 3306")
    parser.add_argument("--db-user", default=None, help="MySQL 用户名")
    parser.add_argument("--db-password", default=None,
                         help="MySQL 密码；出于安全考虑，也可通过环境变量 PIPELINE_DB_PASSWORD 传入")
    parser.add_argument("--db-database", default=None, help="MySQL 数据库名")

    parser.add_argument("--missing-ratio-threshold", type=float, default=MISSING_RATIO_THRESHOLD,
                         help=f"阶段四非缺失检验项占比阈值，默认 {MISSING_RATIO_THRESHOLD}")
    parser.add_argument("--skip-db-extraction", action="store_true",
                         help="跳过阶段一数据库提取，直接使用 {work_root}/{疾病名}_raw 下已有样本")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    class_label_map = _parse_key_value_list(args.class_labels, value_caster=int)
    disease_patient_csv = {k: Path(v) for k, v in _parse_key_value_list(args.disease_csv).items()}

    run_db_extraction = not args.skip_db_extraction
    if run_db_extraction:
        missing_diseases = set(class_label_map) - set(disease_patient_csv)
        if missing_diseases:
            raise SystemExit(
                f"需要执行数据库提取，但以下疾病未通过 --disease-csv 提供患者编号 CSV：{missing_diseases}"
            )
        db_password = args.db_password or os.environ.get("PIPELINE_DB_PASSWORD")
        if not all([args.db_host, args.db_user, db_password, args.db_database]):
            raise SystemExit(
                "需要执行数据库提取，请通过 --db-host/--db-user/--db-password/--db-database "
                "（或环境变量 PIPELINE_DB_PASSWORD）提供完整连接信息，或使用 --skip-db-extraction 跳过"
            )
        db_config = {
            "user": args.db_user,
            "password": db_password,
            "host": args.db_host,
            "port": args.db_port,
            "database": args.db_database,
            "raise_on_warnings": True,
        }
    else:
        db_config = {}

    run_pipeline(
        db_config=db_config,
        lab_table_config_path=args.lab_table_config,
        disease_patient_csv=disease_patient_csv,
        work_root=args.work_root,
        class_label_map=class_label_map,
        missing_ratio_threshold=args.missing_ratio_threshold,
        run_db_extraction=run_db_extraction,
    )