# -*- coding: utf-8 -*-
"""
Copyright: CPMMI
Description:
    File Name   : buildMLModel_onlyembedding.py
    Description : 在全部训练集上拟合 PCA/ScaledPCA，变换到独立测试集，
                  绘制 PC1 vs PC2 散点图及累计解释方差曲线。
                  随机种子、降维逻辑与主脚本 model_classification_pca_cv.py 保持一致。
    Dependency  : sklearn, matplotlib, seaborn, numpy, pandas
History:
    Author : Li, Xinming
    Date   : 2026.04.21
    Version: 1.0
    Summary of Version: 初始版本，硬编码路径在__main__中直接调用main()
    ----------
    Author : Li, Xinming
    Date   : 2026.07.03
    Version: 1.1
    Summary of Version: 将__main__中硬编码的路径改为argparse命令行参数，便于开源仓库中复用
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ---------- matplotlib 全局参数 ----------
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# ---------- 常量 ----------
RANDOM_SEED = 42
AUTO_VARIANCE_THRESHOLD = 0.95


# ========================================================================
#  数据加载（与主脚本完全一致）
# ========================================================================

def load_raw_vectors(path: str) -> np.ndarray:
    """从文本文件读取原始高维向量"""
    vectors = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            vec = np.fromstring(line, sep=' ')
            if vec.size == 0:
                vec = np.fromstring(line, sep='\t')
            vectors.append(vec)
    return np.vstack(vectors)


def load_excel_features(path: str, sheet_name: str = 'Sheet2') -> np.ndarray:
    """从 Excel 文件读取额外特征（第2列起）"""
    df = pd.read_excel(path, header=0, sheet_name=sheet_name)
    return df.iloc[:, 1:].values.astype(float)


def load_labels(path: str) -> np.ndarray:
    """加载整数标签文件"""
    return np.loadtxt(path).astype(int)


# ========================================================================
#  降维策略（与主脚本 fit_and_transform_reduction 一致）
# ========================================================================

def fit_and_transform_reduction(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    method: str = 'scaled_pca',
    n_components: int = 64,
    variance_threshold: float = AUTO_VARIANCE_THRESHOLD,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, Any, Optional[StandardScaler]]:
    """
    在训练集上拟合降维并变换训练/测试集
    返回值: X_train_out, X_test_out, reducer, pre_scaler
    """
    pre_scaler = None

    if method == 'none':
        print(f"  降维: none，保留 {X_train_raw.shape[1]} 维")
        return X_train_raw, X_test_raw, None, None

    # 需要预标准化的方法
    if method in ('scaled_pca', 'auto', 'lda', 'umap'):
        pre_scaler = StandardScaler()
        X_tr = pre_scaler.fit_transform(X_train_raw)
        X_te = pre_scaler.transform(X_test_raw)
    else:
        X_tr, X_te = X_train_raw, X_test_raw

    if method == 'pca':
        nc = min(n_components, X_tr.shape[1], X_tr.shape[0])
        reducer = PCA(n_components=nc, random_state=random_state)
        Xo_tr = reducer.fit_transform(X_tr)
        Xo_te = reducer.transform(X_te)
        ev = reducer.explained_variance_ratio_.sum()
        print(f"  降维: PCA({nc}d), 解释方差={ev:.3f}")

    elif method == 'scaled_pca':
        nc = min(n_components, X_tr.shape[1], X_tr.shape[0])
        reducer = PCA(n_components=nc, random_state=random_state)
        Xo_tr = reducer.fit_transform(X_tr)
        Xo_te = reducer.transform(X_te)
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
        Xo_te = reducer.transform(X_te)
        ev = reducer.explained_variance_ratio_.sum()
        print(f"  降维: Auto({auto_n}d, 阈值={variance_threshold}), "
              f"解释方差={ev:.3f}")

    else:
        raise ValueError(f"本脚本仅支持 pca/scaled_pca/auto，不支持: {method}")

    return Xo_tr, Xo_te, reducer, pre_scaler


# ========================================================================
#  可视化
# ========================================================================

def plot_cumulative_variance(
    evr: np.ndarray,
    cumvar: np.ndarray,
    method: str,
    save_dir: Path
) -> None:
    """绘制累计解释方差曲线图"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    n_pcs = len(evr)
    x = np.arange(1, n_pcs + 1)

    ax1.bar(x, evr, alpha=0.6, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12,
                    fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, cumvar, 'ro-', linewidth=2, markersize=4, label='Cumulative')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12,
                    fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, 1.05])

    for thresh in [0.80, 0.90, 0.95]:
        ax2.axhline(y=thresh, color='gray', linestyle='--',
                     linewidth=0.8, alpha=0.5)
        n_at_thresh = int(np.searchsorted(cumvar, thresh) + 1)
        if n_at_thresh <= n_pcs:
            ax2.annotate(
                f'{thresh*100:.0f}% → PC{n_at_thresh}',
                xy=(n_at_thresh, thresh),
                xytext=(n_at_thresh + max(1, n_pcs * 0.05), thresh - 0.03),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
            )

    plt.title(f'{method.upper()} - Cumulative Explained Variance '
              f'(Full Training Set)',
              fontsize=14, fontweight='bold', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_variance_curve.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] 累计解释方差曲线已保存")


def plot_pc1_vs_pc2(
    X_train_reduced: np.ndarray,
    X_test_reduced: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    method: str,
    save_dir: Path,
    class_names: List[str],
    class_label: List[int]
) -> None:
    """绘制降维后前两个成分的散点图（训练集 + 独立测试集）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_label)))

    if method in ('pca', 'scaled_pca', 'auto'):
        xlabel, ylabel = 'PC1', 'PC2'
    else:
        xlabel, ylabel = 'Dim1', 'Dim2'

    # 左图：全量训练集
    for i, (label, name) in enumerate(zip(class_label, class_names)):
        mask = y_train == label
        axes[0].scatter(
            X_train_reduced[mask, 0], X_train_reduced[mask, 1],
            c=[colors[i]], label=name, alpha=0.6, s=30,
            edgecolors='white', linewidths=0.3
        )
    axes[0].set_xlabel(xlabel, fontsize=12, fontweight='bold')
    axes[0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    axes[0].set_title(f'Training Set (Full Training Set)',
                       fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    # 右图：独立测试集
    for i, (label, name) in enumerate(zip(class_label, class_names)):
        mask = y_test == label
        axes[1].scatter(
            X_test_reduced[mask, 0], X_test_reduced[mask, 1],
            c=[colors[i]], label=name, alpha=0.6, s=30,
            edgecolors='white', linewidths=0.3
        )
    axes[1].set_xlabel(xlabel, fontsize=12, fontweight='bold')
    axes[1].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    axes[1].set_title(f'Independent Test Set',
                       fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f'{method.upper()} - {xlabel} vs {ylabel} '
                 f'(Train vs Independent Test)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "pc1_vs_pc2_scatter.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [✓] PC1 vs PC2 散点图已保存")


def save_pca_diagnostics(
    reducer: PCA,
    pre_scaler: Optional[StandardScaler],
    X_train_reduced: np.ndarray,
    X_test_reduced: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    method: str,
    output_dir: str,
    class_names: List[str],
    class_label: List[int]
) -> None:
    """保存 PCA 诊断信息（复现主脚本 save_reduction_diagnostics 的核心内容）"""
    diag_dir = Path(output_dir) / "pca_train_test_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存降维器对象
    import joblib
    joblib.dump({
        'reducer': reducer,
        'pre_scaler': pre_scaler,
        'method': method,
        'note': 'fitted on full training set, transformed on indep test set'
    }, diag_dir / "reducer_bundle.pkl")

    # 2. PCA 特有属性
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
            'PC': [f'PC{i+1}' for i in range(len(evr))],
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

        plot_cumulative_variance(evr, cumvar, method, diag_dir)

    # 3. PC1 vs PC2 散点图
    if X_train_reduced.shape[1] >= 2:
        plot_pc1_vs_pc2(
            X_train_reduced, X_test_reduced,
            y_train, y_test,
            method, diag_dir,
            class_names, class_label
        )

    # 4. 降维后数据保存
    np.savetxt(diag_dir / "X_train_reduced.csv",
               X_train_reduced, delimiter=',')
    np.savetxt(diag_dir / "X_test_reduced.csv",
               X_test_reduced, delimiter=',')
    np.savetxt(diag_dir / "y_train.csv", y_train, fmt='%d')
    np.savetxt(diag_dir / "y_test.csv", y_test, fmt='%d')

    # 5. 汇总文本
    with open(diag_dir / "reduction_summary.txt", 'w', encoding='utf-8') as f:
        f.write("降维诊断信息 - 全量训练集拟合 / 独立测试集变换\n")
        f.write("=" * 50 + "\n")
        f.write(f"降维方式: {method}\n")
        f.write(f"降维后训练集 shape: {X_train_reduced.shape}\n")
        f.write(f"降维后测试集 shape: {X_test_reduced.shape}\n")
        if isinstance(reducer, PCA):
            f.write(f"保留主成分数: {reducer.n_components_}\n")
            f.write(f"累计解释方差: {cumvar[-1]:.6f}\n")
            f.write(f"各主成分解释方差:\n")
            for i, v in enumerate(evr):
                f.write(f"  PC{i+1}: {v:.6f} (累计: {cumvar[i]:.6f})\n")

    print(f"  [✓] 降维诊断信息已保存: {diag_dir}")


# ========================================================================
#  主流程
# ========================================================================

def main(
    labels_path: str,
    raw_vectors_path: str,
    output_dir: str,
    excel_path: Optional[str] = None,
    excel_sheet: str = 'Sheet2',
    concat_excel: bool = True,
    reduction_method: str = 'scaled_pca',
    n_components: int = 64,
    variance_threshold: float = AUTO_VARIANCE_THRESHOLD
) -> None:
    """
    主流程：加载数据 → 划分训练/测试 → 全量训练集拟合降维 → 变换测试集
            → 保存诊断信息和可视化
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    print("📂 Loading data...")
    X_raw = load_raw_vectors(raw_vectors_path)
    y = load_labels(labels_path)

    X_excel = None
    if concat_excel and excel_path is not None:
        X_excel = load_excel_features(excel_path, sheet_name=excel_sheet)
        if len(X_excel) != len(X_raw):
            raise ValueError(
                f'行数不一致！raw_vectors={len(X_raw)} vs excel={len(X_excel)}'
            )
        print(f"✅ Excel 特征维度: {X_excel.shape[1]}")

    # 类别配置（与主脚本一致）
    if '3cl' in labels_path:
        class_names = ['AILD', 'DILI', 'CHB']
        class_label = [0, 1, 2]
    else:
        class_names = ['AIH', 'PBC', 'DILI', 'CHB']
        class_label = [0, 1, 2, 3]

    print(f"✅ 原始数据 shape: X_raw={X_raw.shape}, y={y.shape}")
    print(f"✅ 降维方式={reduction_method}")

    # 2. 划分训练/测试（与主脚本完全一致：stratify + random_state）
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    X_raw_train = X_raw[train_idx]
    y_train = y[train_idx]
    X_raw_test = X_raw[test_idx]
    y_test = y[test_idx]

    X_excel_train = X_excel[train_idx] if X_excel is not None else None
    X_excel_test = X_excel[test_idx] if X_excel is not None else None

    print(f"✅ 训练集: {X_raw_train.shape[0]} 样本, "
          f"测试集: {X_raw_test.shape[0]} 样本")

    # 3. 全量训练集拟合降维，变换测试集
    print("\n🔧 全量训练集拟合降维器...")
    X_train_reduced, X_test_reduced, reducer, pre_scaler = fit_and_transform_reduction(
        X_raw_train, X_raw_test, y_train,
        method=reduction_method,
        n_components=n_components,
        variance_threshold=variance_threshold,
        random_state=RANDOM_SEED
    )

    # 拼接 Excel 特征（如果启用）
    if X_excel_train is not None and X_excel_test is not None:
        X_train_final = np.hstack([X_train_reduced, X_excel_train])
        X_test_final = np.hstack([X_test_reduced, X_excel_test])
        print(f"  拼接 Excel 特征后维度: {X_train_final.shape[1]}")
    else:
        X_train_final = X_train_reduced
        X_test_final = X_test_reduced

    print(f"  训练集: {X_train_final.shape}, 测试集: {X_test_final.shape}")

    # 4. 保存降维诊断信息
    if reducer is not None:
        print("\n📊 保存降维诊断信息...")
        save_pca_diagnostics(
            reducer=reducer,
            pre_scaler=pre_scaler,
            X_train_reduced=X_train_reduced,   # 保存降维后的原始维度结果
            X_test_reduced=X_test_reduced,
            y_train=y_train,
            y_test=y_test,
            method=reduction_method,
            output_dir=output_dir,
            class_names=class_names,
            class_label=class_label
        )

    # 5. 保存最终特征矩阵（供下游建模使用）
    np.savetxt(Path(output_dir) / "X_train_final.csv",
               X_train_final, delimiter=',')
    np.savetxt(Path(output_dir) / "X_test_final.csv",
               X_test_final, delimiter=',')
    np.savetxt(Path(output_dir) / "y_train.txt", y_train, fmt='%d')
    np.savetxt(Path(output_dir) / "y_test.txt", y_test, fmt='%d')
    np.savetxt(Path(output_dir) / "train_idx.txt", train_idx, fmt='%d')
    np.savetxt(Path(output_dir) / "test_idx.txt", test_idx, fmt='%d')

    # 6. 保存配置
    config = {
        'reduction_method': reduction_method,
        'n_components': n_components,
        'variance_threshold': variance_threshold,
        'random_seed': RANDOM_SEED,
        'labels_path': labels_path,
        'raw_vectors_path': raw_vectors_path,
        'excel_path': excel_path,
        'excel_sheet': excel_sheet,
        'concat_excel': concat_excel,
        'class_names': class_names,
        'class_label': class_label,
        'train_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'train_idx_random_state': RANDOM_SEED,
    }
    config_path = Path(output_dir) / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("🎉 PCA 训练/测试可视化完成！")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 诊断信息: {Path(output_dir) / 'pca_train_test_diagnostics'}")
    print(f"📄 最终特征: X_train_final.csv, X_test_final.csv")
    print(f"📄 标签: y_train.txt, y_test.txt")
    print(f"📄 索引: train_idx.txt, test_idx.txt")
    print(f"{'=' * 60}")


# ========================================================================
#  命令行参数解析
# ========================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，替代原先__main__中的硬编码路径

    参数:
        无

    返回值:
        args: 包含全部运行时配置的命名空间对象，字段与main()参数一一对应
    """
    parser = argparse.ArgumentParser(
        description='在全量训练集上拟合PCA/ScaledPCA降维，并在独立测试集上评估'
    )
    parser.add_argument(
        '--labels_path', type=str, required=True,
        help='标签文件路径，文件名中需包含"3cl"或"4cl"以区分分类任务'
    )
    parser.add_argument(
        '--raw_vectors_path', type=str, required=True,
        help='原始embedding向量文件路径'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='结果与诊断信息输出目录'
    )
    parser.add_argument(
        '--excel_path', type=str, default=None,
        help='临床变量Excel文件路径，仅在--concat_excel开启时使用'
    )
    parser.add_argument(
        '--excel_sheet', type=str, default='Sheet2',
        help='Excel文件中的sheet名称，默认Sheet2'
    )
    parser.add_argument(
        '--concat_excel', action='store_true',
        help='是否将Excel临床变量拼接到降维后的embedding特征上，默认不拼接'
    )
    parser.add_argument(
        '--reduction_method', type=str, default='scaled_pca',
        choices=['pca', 'scaled_pca', 'auto', 'none'],
        help='降维方法，默认scaled_pca'
    )
    parser.add_argument(
        '--n_components', type=int, default=64,
        help='降维目标维度，默认64'
    )
    parser.add_argument(
        '--variance_threshold', type=float, default=AUTO_VARIANCE_THRESHOLD,
        help='reduction_method=auto时使用的累计解释方差阈值，默认0.95'
    )
    return parser.parse_args()


# ========================================================================
#  入口
# ========================================================================

if __name__ == "__main__":
    cli_args = parse_args()

    main(
        labels_path=cli_args.labels_path,
        raw_vectors_path=cli_args.raw_vectors_path,
        output_dir=cli_args.output_dir,
        excel_path=cli_args.excel_path,
        excel_sheet=cli_args.excel_sheet,
        concat_excel=cli_args.concat_excel,
        reduction_method=cli_args.reduction_method,
        n_components=cli_args.n_components,
        variance_threshold=cli_args.variance_threshold
    )
