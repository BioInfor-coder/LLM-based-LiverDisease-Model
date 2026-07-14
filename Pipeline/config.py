# -*- coding: utf-8 -*-

# ---------- Bootstrap 与交叉验证相关常量 ----------
N_BOOTSTRAP: int = 1000
BOOTSTRAP_CI_LEVEL: float = 0.95
RANDOM_SEED: int = 42
N_SPLITS: int = 5

# ---------- 降维相关常量 ----------
AUTO_VARIANCE_THRESHOLD: float = 0.95

# ---------- 超参数选择相关常量 ----------
# 超参数选择评分标准：与 CV 报告指标口径一致
HYPERPARAM_SELECTION_SCORING: str = 'f1_macro'

# ---------- 数据划分相关常量 ----------
# 随机划分默认测试比例（仅在 split_method='random' 时使用）
DEFAULT_TEST_SIZE: float = 0.2
