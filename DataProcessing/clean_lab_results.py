# -*- coding: utf-8 -*-


# 标准库
import argparse
import re
from typing import Any, List, Optional, Tuple, Union

# 第三方库
import pandas as pd
from sklearn.impute import KNNImputer


# 阴阳性关键词
POSITIVE_KEYWORDS: List[str] = ['阳性', 'pos', '+']
NEGATIVE_KEYWORDS: List[str] = ['阴性', 'neg', '-']

# 性别列名（写死，命中该列名的列会执行男/女 -> 1/0 转换）
GENDER_COLUMN_NAMES: List[str] = ["性别"]
MALE_KEYWORDS: List[str] = ['男', 'male', 'm']
FEMALE_KEYWORDS: List[str] = ['女', 'female', 'f']

# 缺失值/占位值默认标记（与上游 data_preprocessing.py 中 NEGATIVE_RESULT_FLAG 保持一致）
DEFAULT_MISSING_VALUE = -1.0
# KNN插值默认近邻数
DEFAULT_N_NEIGHBORS = 5
# 判定为二分类（如阴阳性0/1）而非连续变量的取值集合
BINARY_CATEGORY_VALUES = {0.0, 1.0}

# 需要在读入后直接整列删除的列名，写死。
# 病理/超声/放射/胃镜的自由文本诊断结论，业务上不需要保留在最终清洗结果里，
# 直接从DataFrame中剔除，而非仅跳过清洗。
DROP_COLUMNS: List[str] = ["病理诊断结论", "超声结论", "放射结论", "胃镜结论"]

# 强制视为分类列（保留、但不参与KNN插值/离群值清除/混杂文本清洗/无信息列剔除）
# 的元数据列名，写死。这些列本身就含字母数字混排（如"0000000001.json"），
# 若不排除会被误判为"混杂异常文本"而清空，丢失溯源信息，因此需保留但跳过清洗。
EXCLUDE_COLUMNS: List[str] = ["样本来源", "文件名", "完整路径", "疾病标签"]

# 离群值检测默认参数：超出 [Q1 - 倍数*IQR, Q3 + 倍数*IQR] 视为离群值
DEFAULT_OUTLIER_IQR_MULTIPLIER = 3.0
# 连续列缺失率剔除默认阈值：缺失率超过该比例的连续列将被剔除
DEFAULT_MAX_MISSING_RATIO_CONTINUOUS = 0.2
# 无信息列剔除默认阈值：清洗后非缺失比例低于该值的列将被剔除（设为0可关闭该功能）
DEFAULT_MIN_VALID_RATIO = 0.05


def classify_yinyang(value: Any) -> Union[int, Any]:
    """
    将阴阳性文本分类为 0 或 1

    参数:
        value: 单元格原始值

    返回值:
        1表示阳性，0表示阴性，无法判断则原样返回
    """
    if isinstance(value, str):
        val_lower = value.lower()
        if any(kw in val_lower for kw in POSITIVE_KEYWORDS):
            return 1
        elif any(kw in val_lower for kw in NEGATIVE_KEYWORDS):
            return 0
    return value


def classify_gender(value: Any) -> Union[int, Any]:
    """
    将性别文本分类为 1（男）或 0（女）

    仅应作用于 GENDER_COLUMN_NAMES 中指定的性别列，采用精确匹配
    （而非子串匹配），避免像 'm'/'f' 这类单字母关键词在其他列上
    误命中

    参数:
        value: 单元格原始值（通常已经过 classify_yinyang 等前序清洗步骤）

    返回值:
        1表示男性，0表示女性，无法判断则原样返回
    """
    if isinstance(value, str):
        val_norm = value.strip().lower()
        if val_norm in MALE_KEYWORDS:
            return 1
        if val_norm in FEMALE_KEYWORDS:
            return 0
    return value


def normalize_scientific_notation(value: Any) -> Union[float, Any]:
    """
    统一清洗科学计数法格式

    支持格式包括：
        - 1.30E十3 / 1.30E+3 / 1.30E3 -> 1300.0
        - 3.32×10^2 / 3.32x10^2 -> 332.0

    注意：形如 "<100" / ">130" 的比较符号前缀统一交由
    convert_comparison_symbol 处理，本函数不再处理该格式，避免
    "<X" 被提前转换为纯数字而丢失比较符号语义，造成与 ">X" 处理
    不一致（">X" 因不匹配本函数任何模式而保留为字符串，"<X" 却被
    转换为纯浮点数，导致同一列内出现类型不一致的异常）

    参数:
        value: 单元格原始值

    返回值:
        转换后的浮点数；无法匹配任何模式时原样返回
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    # 处理 1.30E十3 / 1.30E+3 / 1.30E3
    match1 = re.match(r'^([\d\.]+)\s*[Ee]十?(\d+)$', value)
    if match1:
        base, exp = match1.groups()
        return float(base) * (10 ** int(exp))

    # 处理 3.32×10^2 或 3.32x10^2
    match2 = re.match(r'^([\d\.]+)\s*[x×]10\^(\d+)$', value)
    if match2:
        base, exp = match2.groups()
        return float(base) * (10 ** int(exp))

    return value


def convert_comparison_symbol(value: Any) -> Union[str, Any]:
    """
    将 >130.0 或 <2.00 等带比较符号的字符串归一化为规范字符串

    参数:
        value: 单元格原始值

    返回值:
        归一化后的比较符号字符串（如">130"）；不匹配则原样返回
    """
    if isinstance(value, str):
        match = re.match(r'^(>|<)\s*([\d\.]+)', value.strip())
        if match:
            symbol, num = match.groups()
            trimmed = num.rstrip('0').rstrip('.') if '.' in num else num
            return f"{symbol}{trimmed}"
    return value


def normalize_blank_string(value: Any, missing_value: float) -> Union[float, Any]:
    """
    将空白/纯空格字符串统一归一化为缺失值标记

    参数:
        value: 单元格原始值
        missing_value: 缺失值标记

    返回值:
        missing_value（当原值为空字符串或纯空白字符串时）；否则原样返回
    """
    if isinstance(value, str) and not value.strip():
        return missing_value
    return value


def normalize_mixed_text_result(value: Any, missing_value: float) -> Union[float, Any]:
    """
    识别既非阴阳性、又非科学计数法/比较符号格式，却混杂无关文字与数字的
    异常检验结果（如"Normal 3.4"），统一转换为缺失值标记

    该函数应在 classify_yinyang / normalize_scientific_notation /
    convert_comparison_symbol 之后调用，且仅应作用于「检验结果」类列
    （不应作用于病理/超声/放射/胃镜结论等自由文本列，避免误清洗正常
    诊断描述文本）

    参数:
        value: 经过前序清洗步骤处理后的单元格值
        missing_value: 缺失值标记

    返回值:
        missing_value（若判定为无法可靠解析的异常混杂文本）；否则原样返回
    """
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return missing_value

    # 已经是规范的比较符号格式（如">130" "<2"），保留不动
    if re.match(r'^[<>]\s*[\d\.]+$', stripped):
        return value

    # 纯数字（含负号/小数点），非异常，交由后续类型识别处理
    if re.match(r'^-?[\d\.]+$', stripped):
        return value

    # 同时包含字母/汉字与数字，且不属于以上已知规范格式 -> 判定为混杂异常文本
    if re.search(r'[A-Za-z\u4e00-\u9fa5]', stripped) and re.search(r'\d', stripped):
        print(f"[⚠] 发现无法解析的混杂异常文本，已标记为缺失：{value!r}")
        return missing_value

    return value


def clean_dataframe(
    df: pd.DataFrame,
    missing_value: float = DEFAULT_MISSING_VALUE,
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    依次执行原生NaN归一化（真正空白单元格 -> 缺失值标记）、性别归一化
    （男/女 -> 1/0，仅作用于GENDER_COLUMN_NAMES指定列）、阴阳性归一化、
    科学计数法归一化、比较符号归一化、空白字符串归一化，并对非自由文本列
    额外执行混杂异常文本清洗，清洗整个DataFrame

    参数:
        df: 原始DataFrame
        missing_value: 缺失值标记，用于空白值/原生NaN与混杂异常文本的归一化
        exclude_columns: 自由文本列名列表（如病理/超声/放射/胃镜结论等），
            这些列不参与混杂异常文本清洗，避免误清除正常诊断描述

    返回值:
        df_cleaned: 清洗后的DataFrame（不修改原对象）
    """
    exclude_columns = exclude_columns or []
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        # 真正的空白Excel单元格读入后通常直接是float('nan')而非空字符串，
        # 需先统一转换为缺失值标记，避免后续 isinstance(v, str) 判断漏判
        series = df_cleaned[col].where(df_cleaned[col].notna(), missing_value)
        if col in GENDER_COLUMN_NAMES:
            series = series.apply(classify_gender)
        series = (
            series
            .apply(classify_yinyang)
            .apply(normalize_scientific_notation)
            .apply(convert_comparison_symbol)
            .apply(lambda v: normalize_blank_string(v, missing_value))
        )
        if col not in exclude_columns:
            series = series.apply(lambda v: normalize_mixed_text_result(v, missing_value))
        df_cleaned[col] = series

    return df_cleaned


def _is_missing(value: Any, missing_value: float) -> bool:
    """
    判断单个值是否等于缺失值标记，或为pandas/numpy原生缺失值（NaN/None）

    真正的空白Excel单元格经pandas读入后通常直接是float('nan')而非空字符串，
    此前版本仅比较 float(value) == missing_value，无法识别这类原生NaN，
    导致含空白单元格的列被误判为连续列并交由KNN插值出无意义的小数结果

    参数:
        value: 待判断的值
        missing_value: 缺失值标记（如-1）

    返回值:
        True表示该值等于缺失值标记，或本身就是NaN/None
    """
    if pd.isna(value):
        return True
    try:
        return float(value) == missing_value
    except (TypeError, ValueError):
        return False


def identify_column_types(
    df: pd.DataFrame,
    missing_value: float,
    exclude_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """
    区分DataFrame中的分类列与连续数值列

    判定规则：
        1. exclude_columns中显式指定的列，直接归为分类列（不参与插值）
        2. 非缺失值全部可转为浮点数、且取值不局限于{0,1}的列，判定为连续列
        3. 其余（含非数值字符串，或仅取值{0,1}的二分类列）判定为分类列

    参数:
        df: 清洗后的DataFrame
        missing_value: 缺失值标记
        exclude_columns: 强制视为分类列（不参与插值）的列名列表

    返回值:
        categorical_cols: 分类列名列表
        continuous_cols: 连续数值列名列表
    """
    categorical_cols: List[str] = []
    continuous_cols: List[str] = []

    for col in df.columns:
        if col in exclude_columns:
            categorical_cols.append(col)
            continue

        non_missing_values = [v for v in df[col] if not _is_missing(v, missing_value)]

        numeric_values: List[float] = []
        is_numeric = True
        for v in non_missing_values:
            try:
                numeric_values.append(float(v))
            except (TypeError, ValueError):
                is_numeric = False
                break

        if not is_numeric or not numeric_values:
            categorical_cols.append(col)
            continue

        if set(numeric_values).issubset(BINARY_CATEGORY_VALUES):
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    return categorical_cols, continuous_cols


def clear_outliers_iqr(
    df: pd.DataFrame,
    continuous_cols: List[str],
    missing_value: float,
    iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """
    基于IQR（四分位距）方法检测连续列中的离群值，并将其转换为缺失值标记，
    以便后续交由KNN重新插值填补，而非保留极端异常数值参与插值计算

    判定规则：
        对每个连续列，取非缺失数值计算 Q1、Q3、IQR = Q3 - Q1；
        取值范围在 [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR] 之外的
        视为离群值

    参数:
        df: 清洗后的DataFrame
        continuous_cols: 待检测的连续数值列名列表
        missing_value: 缺失值标记
        iqr_multiplier: IQR倍数阈值，默认 DEFAULT_OUTLIER_IQR_MULTIPLIER（3.0）；
            设为 <= 0 时跳过离群值检测，直接返回原DataFrame

    返回值:
        df_result: 离群值已替换为缺失值标记的DataFrame（副本，不修改原对象）
    """
    df_result = df.copy()
    if iqr_multiplier <= 0 or not continuous_cols:
        return df_result

    for col in continuous_cols:
        numeric_col = pd.to_numeric(df_result[col], errors="coerce")
        non_missing_mask = ~numeric_col.apply(lambda v: _is_missing(v, missing_value)) & numeric_col.notna()
        non_missing_values = numeric_col[non_missing_mask]

        if len(non_missing_values) < 4:
            # 样本量过少时四分位数不具统计意义，跳过该列离群值检测
            continue

        q1, q3 = non_missing_values.quantile(0.25), non_missing_values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outlier_mask = non_missing_mask & ((numeric_col < lower_bound) | (numeric_col > upper_bound))
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            df_result.loc[outlier_mask, col] = missing_value
            print(f"[⚠] 列「{col}」检测到 {outlier_count} 个离群值"
                  f"（合理范围 [{lower_bound:.2f}, {upper_bound:.2f}]），已标记为缺失")

    return df_result


def drop_uninformative_columns(
    df: pd.DataFrame,
    missing_value: float,
    categorical_cols: List[str],
    continuous_cols: List[str],
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    剔除清洗后仍几乎全部缺失、或非缺失取值单一（零方差、无信息量）的列

    参数:
        df: 清洗后的DataFrame（含离群值清除结果）
        missing_value: 缺失值标记
        categorical_cols: identify_column_types 得到的分类列名列表
        continuous_cols: identify_column_types 得到的连续列名列表
        min_valid_ratio: 非缺失比例阈值，低于该值的列将被剔除；
            设为 <= 0 时跳过该功能，直接返回原DataFrame与列名列表

    返回值:
        (df_result, categorical_cols_kept, continuous_cols_kept) 三元组：
        剔除无信息列后的DataFrame，以及同步更新后的分类列/连续列名列表
    """
    if min_valid_ratio <= 0:
        return df.copy(), list(categorical_cols), list(continuous_cols)

    n_rows = len(df)
    columns_to_drop: List[str] = []

    for col in list(categorical_cols) + list(continuous_cols):
        values = df[col]
        non_missing = [v for v in values if not _is_missing(v, missing_value)]
        valid_ratio = len(non_missing) / n_rows if n_rows > 0 else 0.0

        # 情况一：非缺失比例过低，几乎全部缺失
        if valid_ratio < min_valid_ratio:
            columns_to_drop.append(col)
            print(f"[⚠] 列「{col}」非缺失比例仅 {valid_ratio:.2%}，判定为无信息列并剔除")
            continue

        # 情况二：非缺失取值全部相同（零方差），同样不提供区分信息
        unique_values = set(non_missing)
        if len(non_missing) > 0 and len(unique_values) == 1:
            columns_to_drop.append(col)
            print(f"[⚠] 列「{col}」非缺失取值全部相同（{unique_values.pop()!r}），判定为无信息列并剔除")

    if not columns_to_drop:
        return df.copy(), list(categorical_cols), list(continuous_cols)

    df_result = df.drop(columns=columns_to_drop)
    categorical_cols_kept = [c for c in categorical_cols if c not in columns_to_drop]
    continuous_cols_kept = [c for c in continuous_cols if c not in columns_to_drop]
    return df_result, categorical_cols_kept, continuous_cols_kept


def drop_high_missing_continuous_columns(
    df: pd.DataFrame,
    continuous_cols: List[str],
    missing_value: float,
    max_missing_ratio: float = DEFAULT_MAX_MISSING_RATIO_CONTINUOUS,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    剔除缺失率超过阈值的连续数值列（不影响分类列）

    仅针对连续型变量：缺失率过高的连续列即便交给KNN插值，插值结果也主要由
    「大量填充值」构成而非真实观测，容易引入偏差，因此在插值前直接剔除

    参数:
        df: 清洗后的DataFrame（建议在离群值清除之后、KNN插值之前调用，
            以便离群值转换的缺失也计入缺失率）
        continuous_cols: identify_column_types 得到的连续列名列表
        missing_value: 缺失值标记
        max_missing_ratio: 缺失率阈值，超过该比例的连续列将被剔除，
            默认 DEFAULT_MAX_MISSING_RATIO_CONTINUOUS（0.2，即20%）；
            设为 >= 1 时跳过该功能，直接返回原DataFrame与列名列表

    返回值:
        (df_result, continuous_cols_kept) 二元组：
        剔除高缺失率连续列后的DataFrame，以及同步更新后的连续列名列表
    """
    if max_missing_ratio >= 1 or not continuous_cols:
        return df.copy(), list(continuous_cols)

    n_rows = len(df)
    columns_to_drop: List[str] = []

    for col in continuous_cols:
        missing_count = sum(1 for v in df[col] if _is_missing(v, missing_value))
        missing_ratio = missing_count / n_rows if n_rows > 0 else 0.0
        if missing_ratio > max_missing_ratio:
            columns_to_drop.append(col)
            print(f"[⚠] 连续列「{col}」缺失率 {missing_ratio:.2%} 超过阈值 "
                  f"{max_missing_ratio:.2%}，判定为高缺失率列并剔除")

    if not columns_to_drop:
        return df.copy(), list(continuous_cols)

    df_result = df.drop(columns=columns_to_drop)
    continuous_cols_kept = [c for c in continuous_cols if c not in columns_to_drop]
    return df_result, continuous_cols_kept


def knn_impute_continuous_columns(
    df: pd.DataFrame,
    continuous_cols: List[str],
    missing_value: float,
    n_neighbors: int,
) -> pd.DataFrame:
    """
    对连续数值列中标记为缺失值的元素做KNN插值，分类列保持不变

    参数:
        df: 清洗后的DataFrame
        continuous_cols: 待插值的连续数值列名列表
        missing_value: 缺失值标记
        n_neighbors: KNN插值近邻数

    返回值:
        df_result: 连续列已完成插值的DataFrame（分类列原样保留，
            其缺失值仍为missing_value）

    异常:
        无（整列均缺失时跳过该列插值并打印警告，保留缺失值标记）
    """
    df_result = df.copy()
    if not continuous_cols:
        return df_result

    # 转为数值类型，并将缺失值标记替换为NaN以供KNNImputer识别
    sub_df = df_result[continuous_cols].apply(pd.to_numeric, errors="coerce")
    sub_df = sub_df.mask(sub_df == missing_value)

    # 整列全部缺失时无法插值，单独剔除并保留原缺失标记
    all_missing_cols = [c for c in continuous_cols if sub_df[c].isna().all()]
    impute_cols = [c for c in continuous_cols if c not in all_missing_cols]
    if all_missing_cols:
        print(f"[⚠] 以下列全部为缺失值，跳过KNN插值，保留缺失标记：{all_missing_cols}")

    if impute_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_array = imputer.fit_transform(sub_df[impute_cols])
        imputed_df = pd.DataFrame(imputed_array, columns=impute_cols, index=df_result.index)
        for col in impute_cols:
            df_result[col] = imputed_df[col]

    return df_result


def clean_excel_file(
    input_path: str,
    output_path: str,
    missing_value: float = DEFAULT_MISSING_VALUE,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    outlier_iqr_multiplier: float = DEFAULT_OUTLIER_IQR_MULTIPLIER,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    max_missing_ratio_continuous: float = DEFAULT_MAX_MISSING_RATIO_CONTINUOUS,
) -> None:
    """
    读取Excel，先整列删除 DROP_COLUMNS 指定的自由文本诊断结论列，再完成清洗
    （含混杂异常文本清除）、离群值清除、无信息列剔除、连续列高缺失率剔除、
    连续列KNN插值后保存为新文件

    需要整列删除的列名写死为文件顶部的 DROP_COLUMNS 常量（病理诊断结论/
    超声结论/放射结论/胃镜结论）；另有 EXCLUDE_COLUMNS 常量（样本来源/
    文件名/完整路径/疾病标签）为保留但跳过清洗的元数据列。如需调整直接
    修改这两个常量即可，无需命令行参数

    参数:
        input_path: 输入Excel文件路径
        output_path: 输出Excel文件路径
        missing_value: 缺失值标记，默认-1
        n_neighbors: KNN插值近邻数，默认5
        outlier_iqr_multiplier: 连续列离群值检测的IQR倍数阈值，默认3.0，
            设为 <= 0 可关闭离群值清除
        min_valid_ratio: 无信息列剔除的非缺失比例阈值，默认0.05，
            设为 <= 0 可关闭无信息列剔除
        max_missing_ratio_continuous: 连续列缺失率剔除阈值，默认0.2（20%），
            缺失率超过该比例的连续列将被剔除；设为 >= 1 可关闭该功能

    返回值:
        无

    异常:
        FileNotFoundError: 当输入文件不存在时
    """
    df = pd.read_excel(input_path)

    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"[i] 已整列删除：{cols_to_drop}")

    df_cleaned = clean_dataframe(df, missing_value=missing_value, exclude_columns=EXCLUDE_COLUMNS)

    categorical_cols, continuous_cols = identify_column_types(
        df_cleaned, missing_value=missing_value, exclude_columns=EXCLUDE_COLUMNS
    )
    print(f"[i] 分类列（不插值）共 {len(categorical_cols)} 列")
    print(f"[i] 连续数值列（KNN插值）共 {len(continuous_cols)} 列：{continuous_cols}")

    df_outlier_cleared = clear_outliers_iqr(
        df_cleaned, continuous_cols, missing_value=missing_value, iqr_multiplier=outlier_iqr_multiplier
    )

    df_filtered, categorical_cols, continuous_cols = drop_uninformative_columns(
        df_outlier_cleared, missing_value=missing_value,
        categorical_cols=categorical_cols, continuous_cols=continuous_cols,
        min_valid_ratio=min_valid_ratio,
    )
    print(f"[i] 剔除无信息列后：分类列 {len(categorical_cols)} 列，连续列 {len(continuous_cols)} 列")

    df_filtered, continuous_cols = drop_high_missing_continuous_columns(
        df_filtered, continuous_cols=continuous_cols, missing_value=missing_value,
        max_missing_ratio=max_missing_ratio_continuous,
    )
    print(f"[i] 剔除高缺失率连续列后：连续列共 {len(continuous_cols)} 列")

    df_result = knn_impute_continuous_columns(
        df_filtered, continuous_cols, missing_value=missing_value, n_neighbors=n_neighbors
    )

    df_result.to_excel(output_path, index=False)
    print(f"✅ 清洗完成，保存为：{output_path}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数

    参数:
        无

    返回值:
        args: 包含input_path、output_path、missing_value、n_neighbors、
            outlier_iqr_multiplier、min_valid_ratio、
            max_missing_ratio_continuous字段的命名空间对象

    说明:
        强制视为分类列（不参与插值）的自由文本列名写死为文件顶部的
        EXCLUDE_COLUMNS 常量，不再通过命令行参数传入
    """
    parser = argparse.ArgumentParser(description="检验结果宽表清洗与KNN缺失值插值脚本")
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="输入Excel文件路径",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="输出Excel文件路径",
    )
    parser.add_argument(
        "--missing_value", type=float, default=DEFAULT_MISSING_VALUE,
        help=f"缺失值/占位值标记，默认{DEFAULT_MISSING_VALUE}",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=DEFAULT_N_NEIGHBORS,
        help=f"KNN插值近邻数，默认{DEFAULT_N_NEIGHBORS}",
    )
    parser.add_argument(
        "--outlier_iqr_multiplier", type=float, default=DEFAULT_OUTLIER_IQR_MULTIPLIER,
        help=f"连续列离群值检测的IQR倍数阈值，默认{DEFAULT_OUTLIER_IQR_MULTIPLIER}，"
             f"设为 <= 0 可关闭离群值清除",
    )
    parser.add_argument(
        "--min_valid_ratio", type=float, default=DEFAULT_MIN_VALID_RATIO,
        help=f"无信息列剔除的非缺失比例阈值，默认{DEFAULT_MIN_VALID_RATIO}，"
             f"设为 <= 0 可关闭无信息列剔除",
    )
    parser.add_argument(
        "--max_missing_ratio_continuous", type=float, default=DEFAULT_MAX_MISSING_RATIO_CONTINUOUS,
        help=f"连续列缺失率剔除阈值，默认{DEFAULT_MAX_MISSING_RATIO_CONTINUOUS}（20%），"
             f"缺失率超过该比例的连续列将被剔除；设为 >= 1 可关闭该功能",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本主入口：解析参数并执行Excel清洗、离群值清除、无信息列剔除、
    连续列高缺失率剔除与插值流程

    参数:
        无

    返回值:
        无
    """
    args = parse_args()
    clean_excel_file(
        input_path=args.input_path,
        output_path=args.output_path,
        missing_value=args.missing_value,
        n_neighbors=args.n_neighbors,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        min_valid_ratio=args.min_valid_ratio,
        max_missing_ratio_continuous=args.max_missing_ratio_continuous,
    )


if __name__ == "__main__":
    main()