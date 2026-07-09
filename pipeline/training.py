# -*- coding: utf-8 -*-


# 标准库
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 第三方库
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# 本地模块
from config import RANDOM_SEED, AUTO_VARIANCE_THRESHOLD, DEFAULT_TEST_SIZE
from metrics import (
    compute_weighted_sens_spec, compute_macro_sens_spec,
    compute_weighted_auc, compute_macro_auc,
    plot_cumulative_variance, plot_pc1_vs_pc2, plot_confusion, plot_roc
)


# ========================================================================
#  第一部分：数据加载与建模数据保存
# ========================================================================
def load_raw_vectors(path: str) -> np.ndarray:
    """
    加载原始向量文件（空格或制表符分隔）

    参数:
        path: 向量文件路径

    返回值:
        向量矩阵 (n_samples, n_features)
    """
    vectors = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            vec = np.fromstring(line, sep=' ')
            if vec.size == 0:
                vec = np.fromstring(line, sep='\t')
            vectors.append(vec)
    return np.vstack(vectors)


def load_excel_features(
        path: str,
        sheet_name: str = 'Sheet2'
) -> np.ndarray:
    """
    从 Excel 文件加载特征矩阵（跳过第一列）

    参数:
        path: Excel 文件路径
        sheet_name: 工作表名称

    返回值:
        特征矩阵
    """
    df = pd.read_excel(path, header=0, sheet_name=sheet_name)
    print(df.drop(columns=["文件名", "样本来源", "疾病标签","完整路径"]).values.astype(float))
    return df.drop(columns=["文件名", "样本来源", "疾病标签","完整路径"]).values.astype(float)


def load_labels(path: str) -> np.ndarray:
    """
    加载标签文件

    参数:
        path: 标签文件路径

    返回值:
        整型标签数组
    """
    return np.loadtxt(path).astype(int)


def save_modeling_data(
        X: np.ndarray,
        y: np.ndarray,
        save_path: Path,
        data_type: str = "train"
) -> None:
    """
    将建模数据（特征矩阵 + 标签）保存为 CSV 文件

    参数:
        X: 特征矩阵
        y: 标签数组
        save_path: 保存路径
        data_type: 数据类型描述，用于打印日志
    """
    n_features = X.shape[1]
    feature_cols = [f"feature_{i + 1}" for i in range(n_features)]
    label_col = ["label"]

    df_feat = pd.DataFrame(X, columns=feature_cols)
    df_label = pd.DataFrame(y, columns=label_col)
    df_final = pd.concat([df_feat, df_label], axis=1)

    df_final.to_csv(save_path, index=False, encoding="utf-8")
    print(f"  [✓] {data_type} 建模数据已保存: {save_path.name}")


def get_modeling_data_dir(output_dir: str, fold_idx: Optional[int] = None) -> Path:
    """
    获取建模数据保存目录

    参数:
        output_dir: 输出根目录
        fold_idx: 折索引，None 表示最终训练/测试集

    返回值:
        目标目录的 Path 对象
    """
    base_dir = Path(output_dir) / "modeling_data"
    if fold_idx is None:
        return base_dir / "final_train_test"
    else:
        return base_dir / f"fold_{fold_idx + 1}"


# ========================================================================
#  第二部分：训练/Hold-out 划分
# ========================================================================
def make_split_indices(
        y: np.ndarray,
        split_method: str,
        excel_path: Optional[str],
        excel_sheet: str,
        test_size: float = DEFAULT_TEST_SIZE,
        random_seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据指定方式生成训练+验证集与内部保留验证集的样本索引

    参数:
        y: 全体样本标签（用于 random 模式下的分层）
        split_method: 划分方式，可选:
            - 'time'  : 基于 Excel 中"检验日期"按年份划分
                        (2010-2019 训练+验证 vs 2020-2025 内部保留验证)
            - 'random': 分层随机按 (1-test_size):test_size 划分
        excel_path: Excel 文件路径，time 模式必需，用于读取"检验日期"列
        excel_sheet: Excel 工作表名
        test_size: random 模式下保留验证集所占比例，默认 0.2
        random_seed: random 模式下的随机种子

    返回值:
        train_val_idx: 训练+验证集索引
        test_idx: 内部保留验证集索引

    异常:
        SystemExit: 当 split_method 不被支持、必需文件/列缺失或划分后无样本时
    """
    method = split_method.lower().strip()
    n_samples = len(y)
    indices = np.arange(n_samples)

    if method == 'time':
        # 基于 Excel "检验日期" 列按年份划分
        if excel_path is None:
            sys.exit("错误：split_method='time' 需要提供 excel_path 参数。")

        try:
            df_time = pd.read_excel(excel_path, sheet_name=excel_sheet,
                                     usecols=["检验日期"])
        except KeyError:
            available_cols = pd.read_excel(
                excel_path, sheet_name=excel_sheet, nrows=0
            ).columns.tolist()
            sys.exit(f"Excel 文件中未找到“检验日期”列。可用列名：{available_cols}")

        if len(df_time) != n_samples:
            sys.exit(f"行数不一致！Excel 检验日期行数={len(df_time)} "
                     f"vs labels 行数={n_samples}")

        years = pd.to_datetime(df_time["检验日期"]).dt.year
        if years.isnull().any():
            sys.exit("检验日期列存在缺失值，请先处理。")

        train_mask = ((years >= 2010) & (years <= 2019)).values
        test_mask = ((years >= 2020) & (years <= 2025)).values

        if not train_mask.any() or not test_mask.any():
            sys.exit(f"时间划分后无有效样本：训练集 {train_mask.sum()}，"
                     f"测试集 {test_mask.sum()}")

        train_val_idx = indices[train_mask]
        test_idx = indices[test_mask]
        print(f"✅ 划分方式: 基于检验日期的年份划分 "
              f"(2010-2019 vs 2020-2025)")
        return train_val_idx, test_idx

    elif method == 'random':
        # 分层随机划分（不依赖"检验日期"列）
        if not (0.0 < test_size < 1.0):
            sys.exit(f"错误: test_size 必须在 (0, 1) 区间，当前值: {test_size}")

        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_seed,
            stratify=y
        )
        train_ratio = 1.0 - test_size
        print(f"✅ 划分方式: 分层随机划分 "
              f"({train_ratio:.0%} : {test_size:.0%})")
        return train_val_idx, test_idx

    else:
        sys.exit(f"错误: 不支持的 split_method='{split_method}'，"
                 "仅支持 'time' 或 'random'")


# ========================================================================
#  第三部分：降维（策略 / 缓存 / 诊断信息保存）
# ========================================================================
def get_fold_cache_dir(output_dir: str, fold_idx: int) -> Path:
    """
    获取降维缓存目录路径

    参数:
        output_dir: 输出根目录
        fold_idx: 折索引，-1 表示全量训练集

    返回值:
        缓存目录的 Path 对象
    """
    if fold_idx == -1:
        return Path(output_dir) / "reduction_diagnostics" / "fold0"
    return Path(output_dir) / "reduction_diagnostics" / f"fold{fold_idx + 1}"


def try_load_cached_reduction(
        output_dir: str,
        fold_idx: int
) -> Optional[Tuple[np.ndarray, np.ndarray, Any, Optional[StandardScaler]]]:
    """
    尝试从磁盘加载已缓存的降维结果

    参数:
        output_dir: 输出根目录
        fold_idx: 折索引

    返回值:
        缓存命中时返回 (X_train_reduced, X_val_reduced, reducer, pre_scaler)，
        否则返回 None
    """
    cache_dir = get_fold_cache_dir(output_dir, fold_idx)

    train_csv = cache_dir / "X_train_reduced.csv"
    val_csv = cache_dir / "X_val_reduced.csv"
    bundle_pkl = cache_dir / "reducer_bundle.pkl"

    if not (train_csv.exists() and val_csv.exists()
            and bundle_pkl.exists()):
        return None

    try:
        X_train_reduced = np.loadtxt(train_csv, delimiter=',')
        X_val_reduced = np.loadtxt(val_csv, delimiter=',')

        if X_train_reduced.ndim == 1:
            X_train_reduced = X_train_reduced.reshape(1, -1)
        if X_val_reduced.ndim == 1:
            X_val_reduced = X_val_reduced.reshape(1, -1)

        bundle = joblib.load(bundle_pkl)
        reducer = bundle.get('reducer', None)
        pre_scaler = bundle.get('pre_scaler', None)

        fold_label = ("全量训练集" if fold_idx == -1
                      else f"Fold {fold_idx + 1}")
        print(f"  [缓存命中] {fold_label} 降维数据已从磁盘加载: "
              f"train={X_train_reduced.shape}, "
              f"val={X_val_reduced.shape}")
        return X_train_reduced, X_val_reduced, reducer, pre_scaler

    except Exception as e:
        print(f"  [缓存加载失败] {e}，将重新计算降维")
        return None


def save_reduction_diagnostics(
        reducer: Any,
        pre_scaler: Optional[StandardScaler],
        X_train_reduced: np.ndarray,
        X_val_reduced: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        method: str,
        fold_idx: int,
        output_dir: str,
        class_names: List[str],
        class_label: List[int]
) -> None:
    """
    保存降维诊断信息：降维器、方差解释、散点图、降维后数据等

    参数:
        reducer: 降维器对象
        pre_scaler: 预标准化器（可为 None）
        X_train_reduced: 降维后训练集
        X_val_reduced: 降维后验证集
        y_train: 训练集标签
        y_val: 验证集标签
        method: 降维方式名称
        fold_idx: 折索引（-1 表示全量训练集，存为 fold0）
        output_dir: 输出根目录
        class_names: 类别名称列表
        class_label: 类别标签列表
    """
    fold_num = fold_idx + 1
    diag_dir = Path(output_dir) / "reduction_diagnostics" / f"fold{fold_num}"
    diag_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'reducer': reducer,
        'pre_scaler': pre_scaler,
        'method': method,
        'fold': fold_num
    }, diag_dir / "reducer_bundle.pkl")

    cumvar = None
    evr = None

    if isinstance(reducer, PCA):
        np.savetxt(
            diag_dir / "pca_components.csv",
            reducer.components_,
            delimiter=',',
            header=','.join([f'raw_dim_{i}' for i in range(
                reducer.components_.shape[1])]),
            comments=''
        )

        evr = reducer.explained_variance_ratio_
        cumvar = np.cumsum(evr)
        evr_df = pd.DataFrame({
            'PC': [f'PC{i + 1}' for i in range(len(evr))],
            'Explained_Variance_Ratio': evr,
            'Cumulative_Variance': cumvar
        })
        evr_df.to_csv(diag_dir / "explained_variance_ratio.csv", index=False)

        np.savetxt(
            diag_dir / "pca_mean.csv",
            reducer.mean_.reshape(1, -1),
            delimiter=',',
            header=','.join([f'raw_dim_{i}' for i in range(
                len(reducer.mean_))]),
            comments=''
        )

        plot_cumulative_variance(evr, cumvar, fold_num, method, diag_dir)

    elif isinstance(reducer, LDA):
        if hasattr(reducer, 'explained_variance_ratio_'):
            evr = reducer.explained_variance_ratio_
            cumvar = np.cumsum(evr)
            evr_df = pd.DataFrame({
                'LD': [f'LD{i + 1}' for i in range(len(evr))],
                'Explained_Variance_Ratio': evr,
                'Cumulative_Variance': cumvar
            })
            evr_df.to_csv(
                diag_dir / "lda_explained_variance_ratio.csv", index=False)

        if hasattr(reducer, 'scalings_'):
            np.savetxt(
                diag_dir / "lda_scalings.csv",
                reducer.scalings_,
                delimiter=','
            )

    if X_train_reduced.shape[1] >= 2:
        if fold_idx == -1:
            val_set_label = "Internal Hold-out Validation Set"
        else:
            val_set_label = None
        plot_pc1_vs_pc2(
            X_train_reduced, X_val_reduced,
            y_train, y_val,
            fold_num, method, diag_dir,
            class_names, class_label,
            val_set_label=val_set_label
        )

    np.savetxt(diag_dir / "X_train_reduced.csv",
               X_train_reduced, delimiter=',')
    np.savetxt(diag_dir / "X_val_reduced.csv",
               X_val_reduced, delimiter=',')
    np.savetxt(diag_dir / "y_train.csv", y_train, fmt='%d')
    np.savetxt(diag_dir / "y_val.csv", y_val, fmt='%d')

    with open(diag_dir / "reduction_summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"降维诊断信息 - Fold {fold_num}\n")
        f.write("=" * 50 + "\n")
        f.write(f"降维方式: {method}\n")
        f.write(f"降维后训练集 shape: {X_train_reduced.shape}\n")
        f.write(f"降维后验证集 shape: {X_val_reduced.shape}\n")
        if isinstance(reducer, PCA) and cumvar is not None:
            f.write(f"保留主成分数: {reducer.n_components_}\n")
            f.write(f"累计解释方差: {cumvar[-1]:.6f}\n")
            f.write("各主成分解释方差:\n")
            for i, v in enumerate(evr):
                f.write(f"  PC{i + 1}: {v:.6f} (累计: {cumvar[i]:.6f})\n")

    print(f"  [✓] 降维诊断信息已保存: {diag_dir}")


def fit_and_transform_reduction(
        X_train_raw: np.ndarray,
        X_val_raw: np.ndarray,
        y_train: np.ndarray,
        method: str = 'scaled_pca',
        n_components: int = 64,
        variance_threshold: float = AUTO_VARIANCE_THRESHOLD,
        random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, Any, Optional[StandardScaler]]:
    """
    在训练集上拟合降维器，并变换训练集和验证集

    参数:
        X_train_raw: 原始训练集特征
        X_val_raw: 原始验证集特征
        y_train: 训练集标签（LDA 需要）
        method: 降维方式 ('pca', 'scaled_pca', 'auto', 'lda', 'umap', 'none')
        n_components: 目标降维维度
        variance_threshold: auto 模式的方差阈值
        random_state: 随机种子

    返回值:
        X_train_reduced: 降维后训练集
        X_val_reduced: 降维后验证集
        reducer: 降维器对象（method='none' 时为 None）
        pre_scaler: 预标准化器（仅部分 method 使用，否则为 None）
    """
    pre_scaler = None

    if method == 'none':
        print(f"  降维: none，保留 {X_train_raw.shape[1]} 维")
        return X_train_raw, X_val_raw, None, None

    if method in ('scaled_pca', 'auto', 'lda', 'umap'):
        pre_scaler = StandardScaler()
        X_tr = pre_scaler.fit_transform(X_train_raw)
        X_vl = pre_scaler.transform(X_val_raw)
    else:
        X_tr, X_vl = X_train_raw, X_val_raw

    if method == 'pca':
        nc = min(n_components, X_tr.shape[1], X_tr.shape[0])
        reducer = PCA(n_components=nc, random_state=random_state)
        Xo_tr = reducer.fit_transform(X_tr)
        Xo_vl = reducer.transform(X_vl)
        ev = reducer.explained_variance_ratio_.sum()
        print(f"  降维: PCA({nc}d), 解释方差={ev:.3f}")

    elif method == 'scaled_pca':
        nc = min(n_components, X_tr.shape[1], X_tr.shape[0])
        reducer = PCA(n_components=nc, random_state=random_state)
        Xo_tr = reducer.fit_transform(X_tr)
        Xo_vl = reducer.transform(X_vl)
        ev = reducer.explained_variance_ratio_.sum()
        print(f"  降维: ScaledPCA({nc}d), 解释方差={ev:.3f}")

    elif method == 'auto':
        mx = min(X_tr.shape[1], X_tr.shape[0])
        full = PCA(n_components=mx, random_state=random_state)
        full.fit(X_tr)
        cumvar = np.cumsum(full.explained_variance_ratio_)
        auto_n = int(np.searchsorted(cumvar, variance_threshold) + 1)
        auto_n = min(auto_n, n_components, mx)
        reducer = PCA(n_components=auto_n, random_state=random_state)
        Xo_tr = reducer.fit_transform(X_tr)
        Xo_vl = reducer.transform(X_vl)
        ev = reducer.explained_variance_ratio_.sum()
        print(f"  降维: Auto({auto_n}d, 阈值={variance_threshold}), "
              f"解释方差={ev:.3f}")

    elif method == 'lda':
        nc = min(len(np.unique(y_train)) - 1, X_tr.shape[1])
        reducer = LDA(n_components=nc)
        Xo_tr = reducer.fit_transform(X_tr, y_train)
        Xo_vl = reducer.transform(X_vl)
        print(f"  降维: LDA({nc}d)")

    elif method == 'umap':
        try:
            import umap
        except ImportError:
            sys.exit("需要: pip install umap-learn")
        nc = min(n_components, X_tr.shape[1])
        reducer = umap.UMAP(
            n_components=nc, random_state=random_state,
            n_neighbors=15, min_dist=0.1, metric='euclidean')
        Xo_tr = reducer.fit_transform(X_tr, y=y_train)
        Xo_vl = reducer.transform(X_vl)
        print(f"  降维: UMAP({nc}d)")

    else:
        raise ValueError(f"不支持的降维方式: {method}")

    return Xo_tr, Xo_vl, reducer, pre_scaler


# ========================================================================
#  第四部分：算法配置
# ========================================================================
def get_algorithm_classifiers(
        random_seed: int = RANDOM_SEED,
        num_class: int = 4,
        enable_grid_search: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    获取所有算法的分类器配置及 GridSearch 参数网格

    参数:
        random_seed: 随机种子
        num_class: 类别数量
        enable_grid_search: 是否启用网格搜索

    返回值:
        算法配置字典，键为算法名称
    """
    gs = enable_grid_search
    classifiers = {
        'RF': {
            'classifier': RandomForestClassifier(
                class_weight='balanced',
                random_state=random_seed, n_jobs=-1
            ),
            'needs_scaling': False,
            'use_sample_weight': False,
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 10, 20, 30, 50],
                'min_samples_split': [2, 3, 5, 8],
            } if gs else None,
        },
        'MLP': {
            'classifier': MLPClassifier(
                max_iter=500, random_state=random_seed,
                early_stopping=True, validation_fraction=0.1
            ),
            'needs_scaling': True,
            'use_sample_weight': True,
            'param_grid': {
                'hidden_layer_sizes': [(100, 50), (128, 64), (256, 128)],
                'alpha': [1e-4, 1e-3, 1e-2],
                'learning_rate_init': [0.001, 0.01],
            } if gs else None,
        },
        'LR': {
            'classifier': LogisticRegression(
                max_iter=2000, class_weight='balanced',
                random_state=random_seed, solver='saga'
            ),
            'needs_scaling': True,
            'use_sample_weight': False,
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
            } if gs else None,
        },
        'XGBoost': {
            'classifier': XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss', random_state=random_seed,
                n_jobs=-1
            ),
            'needs_scaling': False,
            'use_sample_weight': True,
            'param_grid': {
                'n_estimators': [300, 500],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.7, 0.9],
                'min_child_weight': [1, 5],
                'reg_alpha': [0, 0.1],
            } if gs else None,
        },
        'SVM': {
            'classifier': SVC(
                probability=True, class_weight='balanced',
                random_state=random_seed
            ),
            'needs_scaling': True,
            'use_sample_weight': False,
            'param_grid': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf'],
                'gamma': ['scale', 0.01, 0.1],
            } if gs else None,
        }
    }

    return classifiers


# ========================================================================
#  第五部分：超参数选择（标准 k 折交叉验证）
# ========================================================================
def score_one_config_on_folds(
        base_clf: Any,
        params: Dict[str, Any],
        fold_feature_sets: List[Dict[str, np.ndarray]],
        needs_scaling: bool,
        use_sample_weight: bool,
        scoring: str
) -> Tuple[float, float, List[float]]:
    """
    在预先按折降维好的若干折上，评估单个超参数配置的平均验证表现

    参数:
        base_clf: 基础分类器实例（用于 clone）
        params: 待评估的一组超参数配置
        fold_feature_sets: 每折的特征字典列表，每个元素含
            'X_train' / 'y_train' / 'X_val' / 'y_val'（均为该折降维/拼接后特征）
        needs_scaling: 该算法是否需要在折内标准化
        use_sample_weight: 是否使用样本权重
        scoring: 评分标准（支持 'f1_weighted'/'f1_macro'/'accuracy'）

    返回值:
        mean_score: 该配置在所有折上的平均验证分数
        std_score: 该配置在所有折上验证分数的标准差
        per_fold_scores: 各折验证分数列表

    异常:
        ValueError: 当 scoring 不受支持时
    """
    per_fold_scores = []

    for fset in fold_feature_sets:
        X_tr = fset['X_train']
        y_tr = fset['y_train']
        X_vl = fset['X_val']
        y_vl = fset['y_val']

        # 折内标准化：在训练子折上拟合，变换验证子折，避免数据泄露
        if needs_scaling:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_vl_s = scaler.transform(X_vl)
        else:
            X_tr_s = X_tr
            X_vl_s = X_vl

        clf = clone(base_clf)
        clf.set_params(**params)

        if use_sample_weight:
            sw = compute_sample_weight('balanced', y_tr)
            clf.fit(X_tr_s, y_tr, sample_weight=sw)
        else:
            clf.fit(X_tr_s, y_tr)

        y_pred = clf.predict(X_vl_s)

        if scoring == 'f1_weighted':
            score = f1_score(y_vl, y_pred, average='weighted',
                             zero_division=0)
        elif scoring == 'f1_macro':
            score = f1_score(y_vl, y_pred, average='macro',
                             zero_division=0)
        elif scoring == 'accuracy':
            score = accuracy_score(y_vl, y_pred)
        else:
            raise ValueError(f"不支持的评分标准: {scoring}")

        per_fold_scores.append(score)

    mean_score = float(np.mean(per_fold_scores))
    std_score = float(np.std(per_fold_scores))
    return mean_score, std_score, per_fold_scores


def select_best_hyperparams_by_cv_mean(
        base_clf: Any,
        param_grid: Optional[Dict[str, List[Any]]],
        fold_feature_sets: List[Dict[str, np.ndarray]],
        algo_name: str,
        output_dir: str,
        needs_scaling: bool = False,
        use_sample_weight: bool = False,
        scoring: str = 'accuracy'
) -> Tuple[Dict[str, Any], Optional[float], pd.DataFrame]:
    """
    采用标准 k 折交叉验证选择超参数配置。

    对**每一个候选超参数配置**，计算它在全部外层折（即用于汇报性能的同一组
    折）上的**平均验证表现**，并选取平均表现最高的配置。这是 k 折交叉验证下
    进行超参数选择的标准做法，可避免不同折各自选出不同超参数、以及“最接近
    均值”这一非常规启发式带来的偏差。

    降维仍在每个外层折内部独立完成（防止数据泄露），fold_feature_sets 中已是
    各折降维/拼接后的特征；本函数仅在此基础上对每个候选配置做折内标准化、训练、
    在该折验证集上评分，再对所有折取平均。所有折共用最终选定的同一组超参数。

    参数:
        base_clf: 基础分类器实例（未拟合，用于 clone）
        param_grid: 候选超参数网格；为 None 时不搜索，直接返回默认配置
        fold_feature_sets: 每折的特征字典列表，每个元素含
            'X_train' / 'y_train' / 'X_val' / 'y_val'
        algo_name: 算法名称（用于日志与文件命名）
        output_dir: 输出根目录（用于保存候选配置排名表）
        needs_scaling: 该算法是否需要折内标准化
        use_sample_weight: 是否使用样本权重
        scoring: 配置评分标准，默认与 CV 汇报口径一致

    返回值:
        best_params: 五折平均验证表现最高的超参数配置
        best_cv_mean_score: 该配置的五折平均验证分数（无搜索时为 None）
        ranking_df: 所有候选配置按五折平均验证分数排序的排名表
    """
    # 未启用网格搜索：使用分类器默认参数，不做配置间比较
    if param_grid is None:
        print(f"    [{algo_name}] 未启用 GridSearch，使用默认超参数配置")
        ranking_df = pd.DataFrame([{
            'rank': 1,
            'mean_cv_score': np.nan,
            'std_cv_score': np.nan,
            'per_fold_scores': str([]),
            'params': str({})
        }])
        return {}, None, ranking_df

    # 展开所有候选超参数配置
    all_configs = list(ParameterGrid(param_grid))

    config_records = []
    for params in all_configs:
        mean_score, std_score, per_fold = score_one_config_on_folds(
            base_clf=base_clf,
            params=params,
            fold_feature_sets=fold_feature_sets,
            needs_scaling=needs_scaling,
            use_sample_weight=use_sample_weight,
            scoring=scoring
        )
        config_records.append({
            'params': params,
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'per_fold_scores': per_fold
        })

    # 按五折平均验证分数降序排名，分数相同则以更小标准差优先
    config_records.sort(
        key=lambda r: (r['mean_cv_score'], -r['std_cv_score']),
        reverse=True
    )

    ranking_df = pd.DataFrame([{
        'rank': i + 1,
        'mean_cv_score': rec['mean_cv_score'],
        'std_cv_score': rec['std_cv_score'],
        'per_fold_scores': str([round(s, 4) for s in rec['per_fold_scores']]),
        'params': str(rec['params'])
    } for i, rec in enumerate(config_records)])

    ranking_dir = Path(output_dir) / "best_params"
    ranking_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = ranking_dir / f"cv_hyperparam_ranking_{algo_name}.csv"
    ranking_df.to_csv(ranking_path, index=False)

    best_record = config_records[0]
    best_params = best_record['params']
    best_cv_mean_score = best_record['mean_cv_score']

    print(f"    [{algo_name}] 候选配置数: {len(all_configs)}, "
          f"评分标准: {scoring}")
    print(f"    [{algo_name}] 选定配置（五折平均验证表现最高的配置）: "
          f"{best_params}")
    print(f"    [{algo_name}] 该配置五折平均验证分数: "
          f"{best_cv_mean_score:.4f} ± {best_record['std_cv_score']:.4f}")
    print(f"    [{algo_name}] 候选配置排名表已保存: {ranking_path}")

    return best_params, best_cv_mean_score, ranking_df


# ========================================================================
#  第六部分：单折固定超参数训练与评估
# ========================================================================
def evaluate_algorithm(
        clf: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        algorithm_name: str,
        class_names: List[str],
        class_label: List[int],
        output_dir: str,
        fold_idx: int,
        selected_params: Dict[str, Any],
        use_sample_weight: bool = False,
        cv_mean_score: Optional[float] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    使用**已选定的固定超参数配置**训练并评估单个算法在一折上的表现。

    注意：超参数已在 select_best_hyperparams_by_cv_mean 中通过标准 k 折
    交叉验证（每个配置取五折平均验证表现，选最高）统一确定。本函数不再在
    折内部进行任何超参数搜索，仅以选定配置在该折上训练并评估，从而保证所有
    折使用一致的超参数配置。

    参数:
        clf: 已设置好选定超参数的分类器实例（未拟合）
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        algorithm_name: 算法名称
        class_names: 类别名称列表
        class_label: 类别标签列表
        output_dir: 输出目录
        fold_idx: 折索引
        selected_params: 已选定的超参数配置（用于记录）
        use_sample_weight: 是否使用样本权重
        cv_mean_score: 选定配置的五折平均验证分数（用于记录）

    返回值:
        clf: 训练后的分类器
        results: 评估结果字典
    """
    results = {}

    print(f"📊 Training {algorithm_name} - Fold {fold_idx + 1} "
          f"(使用选定的固定超参数)...")

    start_time = time.time()

    # 直接使用已选定的固定超参数配置训练，不再进行折内搜索
    if use_sample_weight:
        sw = compute_sample_weight('balanced', y_train)
        clf.fit(X_train, y_train, sample_weight=sw)
    else:
        clf.fit(X_train, y_train)

    results['selected_params'] = selected_params
    results['hyperparam_cv_mean_score'] = cv_mean_score

    training_time = time.time() - start_time

    y_pred = clf.predict(X_val)
    y_score = clf.predict_proba(X_val)
    y_val_bin = label_binarize(y_val, classes=class_label)

    results['accuracy'] = accuracy_score(y_val, y_pred)
    results['training_time'] = training_time

    results['precision_weighted'] = precision_score(y_val, y_pred, average='weighted')
    results['recall_weighted'] = recall_score(y_val, y_pred, average='weighted')
    results['f1_score_weighted'] = f1_score(y_val, y_pred, average='weighted')

    results['precision_macro'] = precision_score(y_val, y_pred, average='macro')
    results['recall_macro'] = recall_score(y_val, y_pred, average='macro')
    results['f1_score_macro'] = f1_score(y_val, y_pred, average='macro')

    results['precision'] = results['precision_weighted']
    results['recall'] = results['recall_weighted']
    results['f1_score'] = results['f1_score_weighted']

    cm = confusion_matrix(y_val, y_pred)
    w_sens, w_spec = compute_weighted_sens_spec(cm)
    m_sens, m_spec = compute_macro_sens_spec(cm)
    results['sensitivity'] = w_sens
    results['specificity'] = w_spec
    results['sensitivity_weighted'] = w_sens
    results['specificity_weighted'] = w_spec
    results['sensitivity_macro'] = m_sens
    results['specificity_macro'] = m_spec

    print(f"\n{algorithm_name} - Fold {fold_idx + 1} Classification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    plot_confusion(cm, class_names, algorithm_name, output_dir, fold_idx + 1)

    auc_dict = plot_roc(
        y_val_bin, y_score, class_names, algorithm_name, output_dir,
        fold_idx + 1)
    results['auc_weighted'] = compute_weighted_auc(auc_dict, y_val_bin)
    results['auc_macro'] = compute_macro_auc(auc_dict)
    results['mean_auc'] = results['auc_weighted']
    results['auc_dict'] = auc_dict

    print(f"✅ {algorithm_name} - Fold {fold_idx + 1} completed")
    print(
        f"   Accuracy: {results['accuracy']:.3f}, "
        f"AUC(w): {results['auc_weighted']:.3f}, "
        f"AUC(m): {results['auc_macro']:.3f}, "
        f"Sensitivity: {results['sensitivity']:.3f}, "
        f"Specificity: {results['specificity']:.3f}, "
        f"Training Time: {training_time:.2f}s")

    return clf, results
