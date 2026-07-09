# -*- coding: utf-8 -*-

# 标准库
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# 本地模块
from config import (
    N_BOOTSTRAP, BOOTSTRAP_CI_LEVEL, RANDOM_SEED, N_SPLITS,
    HYPERPARAM_SELECTION_SCORING
)

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


# ========================================================================
#  第一部分：指标计算
# ========================================================================
def compute_weighted_sens_spec(cm: np.ndarray) -> Tuple[float, float]:
    """
    基于混淆矩阵计算加权敏感度和特异度

    参数:
        cm: 混淆矩阵

    返回值:
        weighted_sens: 加权敏感度
        weighted_spec: 加权特异度
    """
    num_classes = cm.shape[0]
    sensitivity_per_class = []
    specificity_per_class = []
    support_per_class = []

    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - TP - FN - FP
        support = cm[c, :].sum()

        sens = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) != 0 else 0.0

        sensitivity_per_class.append(sens)
        specificity_per_class.append(spec)
        support_per_class.append(support)

    support_arr = np.array(support_per_class, dtype=float)
    total_support = support_arr.sum()

    if total_support == 0:
        return 0.0, 0.0

    weighted_sens = np.dot(sensitivity_per_class, support_arr) / total_support
    weighted_spec = np.dot(specificity_per_class, support_arr) / total_support

    return weighted_sens, weighted_spec


def compute_macro_sens_spec(cm: np.ndarray) -> Tuple[float, float]:
    """
    基于混淆矩阵计算宏平均敏感度和特异度

    参数:
        cm: 混淆矩阵

    返回值:
        macro_sens: 宏平均敏感度
        macro_spec: 宏平均特异度
    """
    num_classes = cm.shape[0]
    sensitivity_per_class = []
    specificity_per_class = []

    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sens = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) != 0 else 0.0

        sensitivity_per_class.append(sens)
        specificity_per_class.append(spec)

    if num_classes == 0:
        return 0.0, 0.0

    macro_sens = np.mean(sensitivity_per_class)
    macro_spec = np.mean(specificity_per_class)

    return macro_sens, macro_spec


def compute_weighted_auc(
        auc_dict: Dict[Any, float],
        y_true_bin: np.ndarray
) -> float:
    """
    计算加权 AUC

    参数:
        auc_dict: 各类别及 micro AUC 字典
        y_true_bin: 二值化后的真实标签

    返回值:
        加权 AUC 值
    """
    per_class_auc = []
    per_class_support = []

    for key, auc_val in auc_dict.items():
        if key == "micro":
            continue
        per_class_auc.append(auc_val)
        per_class_support.append(np.sum(y_true_bin[:, key]))

    support_arr = np.array(per_class_support, dtype=float)
    total_support = support_arr.sum()

    if total_support == 0 or len(per_class_auc) == 0:
        return auc_dict.get("micro", 0.0)

    return np.dot(per_class_auc, support_arr) / total_support


def compute_macro_auc(auc_dict: Dict[Any, float]) -> float:
    """
    计算宏平均 AUC

    参数:
        auc_dict: 各类别及 micro AUC 字典

    返回值:
        宏平均 AUC 值
    """
    per_class_auc = []

    for key, auc_val in auc_dict.items():
        if key == "micro":
            continue
        per_class_auc.append(auc_val)

    if len(per_class_auc) == 0:
        return auc_dict.get("micro", 0.0)

    return np.mean(per_class_auc)


def bootstrap_ci_95(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        class_label: List[int],
        n_bootstrap: int = N_BOOTSTRAP,
        ci_level: float = BOOTSTRAP_CI_LEVEL,
        random_state: int = RANDOM_SEED
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """
    通过 Bootstrap 重采样计算 95% 置信区间

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 预测概率矩阵
        class_label: 类别标签列表
        n_bootstrap: 重采样次数
        ci_level: 置信水平
        random_state: 随机种子

    返回值:
        按 avg_mode ('macro', 'weighted') 分组的指标置信区间字典，
        每个指标为 (mean, lower, upper) 三元组
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    alpha = 1.0 - ci_level

    avg_modes = ['macro', 'weighted']
    boot_metrics = {
        mode: {
            'Accuracy': [], 'Precision': [], 'Recall': [],
            'F1_Score': [], 'Sensitivity': [], 'Specificity': [],
            'AUC': []
        } for mode in avg_modes
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_s = y_score[idx]

        if len(np.unique(y_t)) < len(class_label):
            continue

        acc = accuracy_score(y_t, y_p)
        cm = confusion_matrix(y_t, y_p, labels=class_label)

        for mode in avg_modes:
            boot_metrics[mode]['Accuracy'].append(acc)

            boot_metrics[mode]['Precision'].append(
                precision_score(y_t, y_p, average=mode, zero_division=0))
            boot_metrics[mode]['Recall'].append(
                recall_score(y_t, y_p, average=mode, zero_division=0))
            boot_metrics[mode]['F1_Score'].append(
                f1_score(y_t, y_p, average=mode, zero_division=0))

            if mode == 'weighted':
                b_sens, b_spec = compute_weighted_sens_spec(cm)
            else:
                b_sens, b_spec = compute_macro_sens_spec(cm)
            boot_metrics[mode]['Sensitivity'].append(b_sens)
            boot_metrics[mode]['Specificity'].append(b_spec)

        y_t_bin = label_binarize(y_t, classes=class_label)
        if y_t_bin.shape[1] == 1:
            y_t_bin = np.hstack([1 - y_t_bin, y_t_bin])
        try:
            n_cls = y_t_bin.shape[1]
            per_class_auc = []
            per_class_support = []
            for i in range(n_cls):
                if np.sum(y_t_bin[:, i]) == 0:
                    continue
                fpr_i, tpr_i, _ = roc_curve(y_t_bin[:, i], y_s[:, i])
                per_class_auc.append(auc(fpr_i, tpr_i))
                per_class_support.append(np.sum(y_t_bin[:, i]))
            if len(per_class_auc) > 0:
                sup = np.array(per_class_support, dtype=float)
                weighted_auc_val = np.dot(per_class_auc, sup) / sup.sum()
                boot_metrics['weighted']['AUC'].append(weighted_auc_val)
                macro_auc_val = np.mean(per_class_auc)
                boot_metrics['macro']['AUC'].append(macro_auc_val)
        except ValueError:
            pass

    results = {}
    for mode in avg_modes:
        results[mode] = {}
        for metric_name, values in boot_metrics[mode].items():
            if len(values) == 0:
                results[mode][metric_name] = (np.nan, np.nan, np.nan)
                continue
            arr = np.array(values)
            mean_val = np.mean(arr)
            lower = np.percentile(arr, alpha / 2 * 100)
            upper = np.percentile(arr, (1 - alpha / 2) * 100)
            results[mode][metric_name] = (mean_val, lower, upper)

    return results


# ========================================================================
#  第二部分：可视化绘图
# ========================================================================
def plot_cumulative_variance(
        evr: np.ndarray,
        cumvar: np.ndarray,
        fold_num: int,
        method: str,
        save_dir: Path
) -> None:
    """
    绘制累计方差解释曲线图

    参数:
        evr: 各主成分方差解释比例
        cumvar: 累计方差解释比例
        fold_num: 折编号
        method: 降维方式名称
        save_dir: 保存目录
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    n_pcs = len(evr)
    x = np.arange(1, n_pcs + 1)

    ax1.bar(x, evr, alpha=0.6, color='steelblue', label='Individual')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12,
                    fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.plot(x, cumvar, 'ro-', linewidth=2, markersize=4,
             label='Cumulative')
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
                f'{thresh * 100:.0f}% → PC{n_at_thresh}',
                xy=(n_at_thresh, thresh),
                xytext=(n_at_thresh + max(1, n_pcs * 0.05), thresh - 0.03),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
            )

    plt.title(f'{method.upper()} - Cumulative Explained Variance '
              f'(Fold {fold_num})',
              fontsize=14, fontweight='bold', pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(save_dir / "cumulative_variance_curve.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_pc1_vs_pc2(
        X_train_reduced: np.ndarray,
        X_val_reduced: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        fold_num: int,
        method: str,
        save_dir: Path,
        class_names: List[str],
        class_label: List[int],
        val_set_label: Optional[str] = None
) -> None:
    """
    绘制降维后前两维散点图（训练集 vs 验证集）

    参数:
        X_train_reduced: 降维后训练集
        X_val_reduced: 降维后验证集
        y_train: 训练集标签
        y_val: 验证集标签
        fold_num: 折编号
        method: 降维方式名称
        save_dir: 保存目录
        class_names: 类别名称列表
        class_label: 类别标签列表
        val_set_label: 验证集子图标题（可选）
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_label)))

    if method in ('pca', 'scaled_pca', 'auto'):
        xlabel, ylabel = 'PC1', 'PC2'
    elif method == 'lda':
        xlabel, ylabel = 'LD1', 'LD2'
    elif method == 'umap':
        xlabel, ylabel = 'UMAP1', 'UMAP2'
    else:
        xlabel, ylabel = 'Dim1', 'Dim2'

    for i, (label, name) in enumerate(zip(class_label, class_names)):
        mask = y_train == label
        axes[0].scatter(
            X_train_reduced[mask, 0], X_train_reduced[mask, 1],
            c=[colors[i]], label=name, alpha=0.6, s=30, edgecolors='white',
            linewidths=0.3
        )
    axes[0].set_xlabel(xlabel, fontsize=12, fontweight='bold')
    axes[0].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    axes[0].set_title(f'Training Set (Fold {fold_num})',
                       fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    for i, (label, name) in enumerate(zip(class_label, class_names)):
        mask = y_val == label
        axes[1].scatter(
            X_val_reduced[mask, 0], X_val_reduced[mask, 1],
            c=[colors[i]], label=name, alpha=0.6, s=30, edgecolors='white',
            linewidths=0.3
        )
    axes[1].set_xlabel(xlabel, fontsize=12, fontweight='bold')
    axes[1].set_ylabel(ylabel, fontsize=12, fontweight='bold')
    if val_set_label is not None:
        right_title = val_set_label
    else:
        right_title = f'Validation Set (Fold {fold_num})'
    axes[1].set_title(right_title, fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f'{method.upper()} - {xlabel} vs {ylabel} (Fold {fold_num})',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "pc1_vs_pc2_scatter.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion(
        cm: np.ndarray,
        classes: List[str],
        algorithm_name: str,
        output_dir: str,
        fold_idx: Optional[Union[int, str]] = None
) -> None:
    """
    绘制混淆矩阵热图并保存

    参数:
        cm: 混淆矩阵
        classes: 类别名称列表
        algorithm_name: 算法名称
        output_dir: 输出目录
        fold_idx: 折编号或 'holdout_val'
    """
    plt.figure(figsize=(8, 6))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized * 100, 1)

    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                     xticklabels=classes, yticklabels=classes,
                     cbar_kws={'label': 'Count'},
                     linewidths=0.5, linecolor='gray')

    for i in range(len(classes)):
        for j in range(len(classes)):
            if not np.isnan(cm_normalized[i, j]):
                text = f"{cm[i, j]}\n({cm_normalized[i, j]}%)"
                ax.text(j + 0.5, i + 0.3, text,
                        ha='center', va='center',
                        fontsize=9,
                        color='black' if cm_normalized[i, j] < 70 else 'white')

    plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')
    plt.ylabel("True Label", fontsize=12, fontweight='bold')

    if fold_idx is not None and fold_idx != 'holdout_val':
        title = f"{algorithm_name} - Confusion Matrix (Fold {fold_idx})"
    else:
        title = (f"{algorithm_name} - Confusion Matrix "
                 f"(Internal Hold-out Validation)")

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    save_dir = Path(f"{output_dir}/confusion_matrices/{algorithm_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if fold_idx is not None and fold_idx != 'holdout_val':
        save_path = save_dir / f"confusion_matrix_fold{fold_idx}.png"
    else:
        save_path = save_dir / "confusion_matrix_holdout_val.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confusion matrix saved: {save_path}")


def plot_roc(
        y_true_bin: np.ndarray,
        y_score: np.ndarray,
        classes: List[str],
        algorithm_name: str,
        output_dir: str,
        fold_idx: Optional[Union[int, str]] = None
) -> Dict[Any, float]:
    """
    绘制 ROC 曲线并保存

    参数:
        y_true_bin: 二值化后的真实标签
        y_score: 预测概率矩阵
        classes: 类别名称列表
        algorithm_name: 算法名称
        output_dir: 输出目录
        fold_idx: 折编号或 'holdout_val'

    返回值:
        各类别及 micro 的 AUC 字典
    """
    if y_score.ndim == 1 or y_score.shape[1] == 1:
        pos_prob = y_score.ravel()
        y_score = np.column_stack([1 - pos_prob, pos_prob])

    n_classes = y_score.shape[1]

    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    valid_classes = []
    for i in range(n_classes):
        if np.sum(y_true_bin[:, i]) == 0:
            print(f"⚠️ {algorithm_name} Fold {fold_idx}: "
                  f"Class {classes[i]} missing, skipping ROC")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid_classes.append(i)

    if len(valid_classes) > 0:
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC=0.5)')

    for idx, i in enumerate(valid_classes):
        plt.plot(fpr[i], tpr[i], color=colors[idx], lw=2.5,
                 label=f'{classes[i]} (AUC={roc_auc[i]:.3f})')

    if "micro" in roc_auc:
        plt.plot(fpr["micro"], tpr["micro"], 'k-', lw=3, alpha=0.8,
                 label=f'Micro-average (AUC={roc_auc["micro"]:.3f})')

    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')

    if fold_idx is not None and fold_idx != 'holdout_val':
        title = f"{algorithm_name} - ROC Curve (Fold {fold_idx})"
    else:
        title = (f"{algorithm_name} - ROC Curve "
                 f"(Internal Hold-out Validation)")

    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()

    save_dir = Path(output_dir) / "roc_curves" / algorithm_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if fold_idx is not None and fold_idx != 'holdout_val':
        save_path = save_dir / f"roc_curve_fold{fold_idx}.png"
    else:
        save_path = save_dir / "roc_curve_holdout_val.png"

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] ROC curve saved: {save_path}")

    return roc_auc


def plot_cv_results(summary_df: pd.DataFrame, output_dir: str) -> None:
    """
    绘制五折交叉验证性能对比图（AUC、Accuracy、F1、训练时间）

    参数:
        summary_df: 算法汇总 DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('5-Fold Cross-Validation Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02)

    algorithm_names = summary_df['Algorithm'].tolist()
    x_pos = np.arange(len(algorithm_names))
    display_names = [n[:8] + '...' if len(n) > 10 else n for n in algorithm_names]

    metric_configs = [
        ('Average_AUC', 'AUC_Std', 'AUC', 'AUC Comparison (Mean ± Std)',
         plt.cm.Set3, [0, 1.2]),
        ('Average_Accuracy', 'Accuracy_Std', 'Accuracy',
         'Accuracy Comparison (Mean ± Std)', plt.cm.Set2, [0, 1.2]),
        ('Average_F1_Score', 'F1_Score_Std', 'F1-Score',
         'F1-Score Comparison (Mean ± Std)', plt.cm.Pastel1, [0, 1.2]),
    ]

    for idx, (val_col, std_col, ylabel, title, cmap, ylim) in enumerate(
            metric_configs):
        ax = axes[idx // 2, idx % 2]
        values = summary_df[val_col]
        stds = summary_df[std_col]
        colors = cmap(np.linspace(0, 1, len(summary_df)))

        bars = ax.bar(x_pos, values, yerr=stds, capsize=8,
                      color=colors, edgecolor='black', linewidth=1.2,
                      alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=30, ha='right',
                           fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')

        for bar, value, std in zip(bars, values, stds):
            label_y = bar.get_height() + std + 0.04
            ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                              edgecolor='gray', alpha=0.9))

    ax4 = axes[1, 1]
    time_values = summary_df['Average_Training_Time(s)']
    colors4 = plt.cm.Pastel2(np.linspace(0, 1, len(summary_df)))
    bars4 = ax4.bar(x_pos, time_values, color=colors4,
                    edgecolor='black', linewidth=1.2, alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(display_names, rotation=30, ha='right', fontsize=10)
    ax4.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Training Time Comparison', fontsize=12,
                  fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.2, axis='y', linestyle='--')
    max_time = max(time_values) if len(time_values) > 0 else 1
    y_limit = max_time * 1.2
    ax4.set_ylim([0, y_limit])

    for bar, value in zip(bars4, time_values):
        label_y = bar.get_height() + y_limit * 0.03
        ax4.text(bar.get_x() + bar.get_width() / 2., label_y,
                 f'{value:.1f}s', ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color='black',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                           edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    save_path = Path(output_dir) / "cv_performance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Cross-validation performance comparison plot saved: "
          f"{save_path}")


def plot_algorithm_comparison(
        results_df: pd.DataFrame,
        output_dir: str
) -> None:
    """
    绘制 Hold-out 验证集上的算法性能对比图

    参数:
        results_df: 算法结果 DataFrame
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Algorithm Performance Comparison (Internal Hold-out Validation)',
        fontsize=16, fontweight='bold', y=1.02)

    algorithm_names = results_df['Algorithm'].tolist()
    x_pos = np.arange(len(algorithm_names))
    display_names = [n[:8] + '...' if len(n) > 10 else n
                     for n in algorithm_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithm_names)))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = results_df[metric].tolist()
        bars = ax.bar(x_pos, values, color=colors,
                      edgecolor='black', linewidth=1.2, alpha=0.8)
        ax.set_ylabel(metric, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12,
                     fontweight='bold', pad=15)
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=30, ha='right',
                           fontsize=10)

        for bar, value in zip(bars, values):
            label_y = bar.get_height() + 0.02
            ax.text(bar.get_x() + bar.get_width() / 2., label_y,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                              edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    save_path = Path(output_dir) / "holdout_val_performance_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Hold-out validation performance comparison plot saved: "
          f"{save_path}")


def plot_auc_comparison(
        results_df: pd.DataFrame,
        output_dir: str
) -> None:
    """
    绘制 CV AUC vs Hold-out AUC 对比图

    参数:
        results_df: 算法结果 DataFrame
        output_dir: 输出目录
    """
    algorithm_names = results_df['Algorithm'].tolist()
    display_names = [n[:8] + '...' if len(n) > 10 else n
                     for n in algorithm_names]
    test_auc = results_df['AUC'].tolist()
    cv_auc = results_df['CV_AUC'].tolist()

    x = np.arange(len(algorithm_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects2 = ax.bar(x + width / 2, test_auc, width,
                    label='Internal Hold-out Validation AUC',
                    color='#FF6B6B', edgecolor='black', linewidth=1.2,
                    alpha=0.8)
    rects1 = ax.bar(x - width / 2, cv_auc, width,
                    label='Cross-Validation Mean AUC',
                    color='#4ECDC4', edgecolor='black', linewidth=1.2,
                    alpha=0.8)

    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title(
        'AUC Performance Comparison: CV vs Internal Hold-out Validation',
        fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha='right', fontsize=11)
    ax.set_ylim([0, 1.2])
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    def _add_value_labels(rects: Any, values: List[float]) -> None:
        """在柱状图上添加数值标签"""
        for rect, value in zip(rects, values):
            height = rect.get_height()
            ax.annotate(
                f'{value:.3f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='black',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor='gray', alpha=0.9, linewidth=0.5))

    _add_value_labels(rects1, cv_auc)
    _add_value_labels(rects2, test_auc)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.5)
    ax.axhline(y=0.8, color='green', linestyle=':', linewidth=0.8,
               alpha=0.5)
    ax.axhline(y=0.9, color='orange', linestyle=':', linewidth=0.8,
               alpha=0.5)

    test_mean = np.mean(test_auc)
    cv_mean = np.mean(cv_auc)
    stats_text = (f'Hold-out Val Mean AUC: {test_mean:.3f}\n'
                  f'CV Mean AUC: {cv_mean:.3f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow',
                      edgecolor='orange', alpha=0.8))

    plt.tight_layout()
    save_path = Path(output_dir) / "auc_comparison_detail.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Detailed AUC comparison plot saved: {save_path}")


# ========================================================================
#  第三部分：报告生成
# ========================================================================
def convert_to_serializable(obj: Any) -> Any:
    """
    将 numpy 类型递归转换为 Python 原生类型，用于 JSON 序列化

    参数:
        obj: 待转换对象

    返回值:
        可 JSON 序列化的对象
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                        np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.complex128, np.complex64)):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def save_best_params_report(
        all_results: Dict[str, Dict[str, Any]],
        selected_hyperparams: Dict[str, Dict[str, Any]],
        algorithms: Dict[str, Dict[str, Any]],
        output_dir: str,
        reduction_method: str,
        n_components: int,
        enable_grid_search: bool
) -> None:
    """
    保存最优参数报告：每折指标 CSV、选定配置参数 JSON、最优方案文本报告。

    本函数采用标准 k 折交叉验证超参数选择：每个算法的超参数由
    select_best_hyperparams_by_cv_mean（见 training.py）统一确定（对每个
    候选配置取五折平均验证表现，选最高），所有折共用同一组超参数。因此报告
    中记录的是该选定配置及其五折平均验证分数，以及该配置在各折上重评估
    得到的指标均值，不再有“代表折/最接近均值折”的概念。

    参数:
        all_results: 所有算法的全部折结果
        selected_hyperparams: 每个算法选定的超参数信息
            （含 params / cv_mean_score）
        algorithms: 算法配置字典
        output_dir: 输出目录
        reduction_method: 降维方式
        n_components: 降维维度
        enable_grid_search: 是否启用网格搜索
    """
    params_dir = Path(output_dir) / "best_params"
    params_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每折指标表：所有折共用同一组选定超参数，记录各折验证指标
    params_rows = []
    for algo_name in algorithms.keys():
        sel = selected_hyperparams[algo_name]
        for fold_idx, res in enumerate(
                all_results[algo_name]['fold_results']):
            row = {
                'Algorithm': algo_name,
                'Fold': fold_idx + 1,
                'Hyperparam_CV_Mean_Score': sel.get('cv_mean_score'),
                'Val_Accuracy': res['accuracy'],
                'Val_AUC': res['mean_auc'],
                'Val_F1': res['f1_score'],
            }
            for k, v in sel['params'].items():
                row[f'param_{k}'] = str(v)
            params_rows.append(row)

    params_df = pd.DataFrame(params_rows)
    params_csv_path = params_dir / "all_folds_selected_params.csv"
    params_df.to_csv(params_csv_path, index=False)
    print(f"[✓] 各折指标（共用选定超参数）已保存: {params_csv_path}")

    # 2) 选定配置 JSON：含五折平均验证分数与五折重评估指标均值
    best_summary = {}
    for algo_name in algorithms.keys():
        sel = selected_hyperparams[algo_name]
        fold_results = all_results[algo_name]['fold_results']

        cv_acc = float(np.mean([r['accuracy'] for r in fold_results]))
        cv_f1 = float(np.mean([r['f1_score'] for r in fold_results]))
        cv_sens = float(np.mean([r['sensitivity'] for r in fold_results]))
        cv_spec = float(np.mean([r['specificity'] for r in fold_results]))
        cv_auc = float(np.mean([r['mean_auc'] for r in fold_results]))

        serializable_params = convert_to_serializable(sel['params'])

        best_summary[algo_name] = {
            'selection_strategy': (
                'standard_kfold_cv_mean (每个候选配置取五折平均验证表现，'
                '选平均最高者)'),
            'hyperparam_scoring': HYPERPARAM_SELECTION_SCORING,
            'hyperparam_cv_mean_score': float(sel['cv_mean_score'])
            if sel.get('cv_mean_score') is not None else None,
            'grid_search_enabled': bool(enable_grid_search),
            'selected_params': serializable_params,
            'reduction_method': str(reduction_method),
            'n_components': int(n_components),
            'cv_metrics_mean_over_folds': {
                'accuracy': cv_acc,
                'f1_score': cv_f1,
                'sensitivity': cv_sens,
                'specificity': cv_spec,
                'auc': cv_auc,
            }
        }

    best_json_path = params_dir / "best_models_params.json"
    with open(best_json_path, 'w', encoding='utf-8') as f:
        json.dump(best_summary, f, indent=2, ensure_ascii=False)
    print(f"[✓] 选定配置参数 JSON 已保存: {best_json_path}")

    # 3) 最优方案文本报告
    report_path = params_dir / "best_scheme_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  最优方案汇总报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("超参数选择策略: 标准 k 折交叉验证\n")
        f.write("  - 对每个候选超参数配置，计算其在全部 "
                f"{N_SPLITS} 折上的平均验证表现\n")
        f.write(f"    （评分标准: {HYPERPARAM_SELECTION_SCORING}），"
                "选取平均表现最高的配置；\n")
        f.write("  - 所有折共用同一组选定超参数，随后逐折重评估各项指标；\n")
        f.write("  - 全量训练集重训练与 Hold-out 评估均使用该选定配置。\n\n")
        f.write(f"降维方式: {reduction_method}\n")
        f.write(f"降维维度: {n_components}\n")
        f.write(f"GridSearch: {'启用' if enable_grid_search else '未启用'}\n")
        f.write(f"CV 折数: {N_SPLITS}\n")
        f.write(f"随机种子: {RANDOM_SEED}\n\n")

        # 以五折平均 AUC 衡量各算法整体表现，选最优算法
        algo_cv_auc = {}
        for algo_name in algorithms.keys():
            fold_results = all_results[algo_name]['fold_results']
            algo_cv_auc[algo_name] = float(
                np.mean([r['mean_auc'] for r in fold_results]))
        best_algo = max(algo_cv_auc, key=algo_cv_auc.get)

        f.write(f"★ 最优算法（CV 平均 AUC 最高）: {best_algo} "
                f"(CV Mean AUC={algo_cv_auc[best_algo]:.3f})\n\n")
        f.write("-" * 70 + "\n")

        for algo_name in algorithms.keys():
            sel = selected_hyperparams[algo_name]
            fold_results = all_results[algo_name]['fold_results']
            cv_acc = np.mean([r['accuracy'] for r in fold_results])
            cv_f1 = np.mean([r['f1_score'] for r in fold_results])
            cv_sens = np.mean([r['sensitivity'] for r in fold_results])
            cv_spec = np.mean([r['specificity'] for r in fold_results])
            cv_auc = np.mean([r['mean_auc'] for r in fold_results])
            marker = " ★" if algo_name == best_algo else ""

            f.write(f"\n📊 {algo_name}{marker}\n")
            if sel.get('cv_mean_score') is not None:
                f.write(f"   超参数选择五折平均验证分数 "
                        f"({HYPERPARAM_SELECTION_SCORING}): "
                        f"{sel['cv_mean_score']:.3f}\n")
            f.write(f"   CV Mean AUC: {cv_auc:.3f}\n")
            f.write(f"   CV Mean Accuracy: {cv_acc:.3f}\n")
            f.write(f"   CV Mean F1-Score: {cv_f1:.3f}\n")
            f.write(f"   CV Mean Sensitivity: {cv_sens:.3f}\n")
            f.write(f"   CV Mean Specificity: {cv_spec:.3f}\n")
            f.write("   选定超参数（五折平均验证表现最高的配置）:\n")
            for k, v in sel['params'].items():
                f.write(f"     {k}: {v}\n")
            f.write("   候选配置完整排名见: "
                    f"best_params/cv_hyperparam_ranking_{algo_name}.csv\n")
            f.write("-" * 70 + "\n")

    print(f"[✓] 最优方案报告已保存: {report_path}")
