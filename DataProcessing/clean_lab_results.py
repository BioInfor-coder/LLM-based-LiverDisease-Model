# -*- coding: utf-8 -*-


# 标准库
import argparse
import re
from typing import Any, List, Tuple, Union

# 第三方库
import pandas as pd
from sklearn.impute import KNNImputer


# 阴阳性关键词
POSITIVE_KEYWORDS: List[str] = ['阳性', 'pos', '+']
NEGATIVE_KEYWORDS: List[str] = ['阴性', 'neg', '-']

# 缺失值/占位值默认标记（与上游 data_preprocessing.py 中 NEGATIVE_RESULT_FLAG 保持一致）
DEFAULT_MISSING_VALUE = -1.0
# KNN插值默认近邻数
DEFAULT_N_NEIGHBORS = 5
# 判定为二分类（如阴阳性0/1）而非连续变量的取值集合
BINARY_CATEGORY_VALUES = {0.0, 1.0}


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


def normalize_scientific_notation(value: Any) -> Union[float, Any]:
    """
    统一清洗科学计数法及比较符号前缀格式

    支持格式包括：
        - 1.30E十3 / 1.30E+3 / 1.30E3 -> 1300.0
        - 3.32×10^2 / 3.32x10^2 -> 332.0
        - <100 -> 100.0

    参数:
        value: 单元格原始值

    返回值:
        转换后的浮点数；无法匹配任何模式时原样返回
    """
    if not isinstance(value, str):
        return value

    value = value.strip()

    # 处理 <100 这类值
    if re.match(r'^<\s*[\d\.]+$', value):
        num = re.findall(r'[\d\.]+', value)[0]
        return float(num)

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    依次执行阴阳性归一化、科学计数法归一化、比较符号归一化，清洗整个DataFrame

    参数:
        df: 原始DataFrame

    返回值:
        df_cleaned: 清洗后的DataFrame（不修改原对象）
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        df_cleaned[col] = (
            df_cleaned[col]
            .apply(classify_yinyang)
            .apply(normalize_scientific_notation)
            .apply(convert_comparison_symbol)
        )
    return df_cleaned


def _is_missing(value: Any, missing_value: float) -> bool:
    """
    判断单个值是否等于缺失值标记

    参数:
        value: 待判断的值
        missing_value: 缺失值标记（如-1）

    返回值:
        True表示该值等于缺失值标记
    """
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
    exclude_columns: List[str] = None,
) -> None:
    """
    读取Excel，完成清洗、连续列KNN插值后保存为新文件

    参数:
        input_path: 输入Excel文件路径
        output_path: 输出Excel文件路径
        missing_value: 缺失值标记，默认-1
        n_neighbors: KNN插值近邻数，默认5
        exclude_columns: 强制视为分类列（不参与插值）的列名列表，默认None

    返回值:
        无

    异常:
        FileNotFoundError: 当输入文件不存在时
    """
    exclude_columns = exclude_columns or []

    df = pd.read_excel(input_path)
    df_cleaned = clean_dataframe(df)

    categorical_cols, continuous_cols = identify_column_types(
        df_cleaned, missing_value=missing_value, exclude_columns=exclude_columns
    )
    print(f"[i] 分类列（不插值）共 {len(categorical_cols)} 列")
    print(f"[i] 连续数值列（KNN插值）共 {len(continuous_cols)} 列：{continuous_cols}")

    df_result = knn_impute_continuous_columns(
        df_cleaned, continuous_cols, missing_value=missing_value, n_neighbors=n_neighbors
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
            exclude_columns字段的命名空间对象
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
        "--exclude_columns", type=str, nargs="*", default=[],
        help="强制视为分类列（不参与KNN插值）的列名，空格分隔",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本主入口：解析参数并执行Excel清洗与插值流程

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
        exclude_columns=args.exclude_columns,
    )


if __name__ == "__main__":
    main()
