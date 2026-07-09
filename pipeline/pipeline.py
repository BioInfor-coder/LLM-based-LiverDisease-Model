# -*- coding: utf-8 -*-

# 标准库
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# 第三方库
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight

# 本地模块
from config import (
    N_SPLITS, RANDOM_SEED, N_BOOTSTRAP, BOOTSTRAP_CI_LEVEL,
    AUTO_VARIANCE_THRESHOLD, HYPERPARAM_SELECTION_SCORING,
    DEFAULT_TEST_SIZE
)
from training import (
    load_raw_vectors, load_excel_features, load_labels,
    save_modeling_data, get_modeling_data_dir,
    make_split_indices,
    try_load_cached_reduction, save_reduction_diagnostics,
    fit_and_transform_reduction,
    get_algorithm_classifiers,
    select_best_hyperparams_by_cv_mean,
    evaluate_algorithm
)
from metrics import (
    compute_weighted_sens_spec, compute_weighted_auc, bootstrap_ci_95,
    plot_cv_results, plot_confusion, plot_roc,
    plot_algorithm_comparison, plot_auc_comparison,
    save_best_params_report
)


# ========================================================================
#  主流程
# ========================================================================
def run_pipeline(
        labels_path: str,
        raw_vectors_path: str,
        output_dir: str,
        excel_path: Optional[str] = None,
        excel_sheet: str = 'Sheet2',
        n_components: int = 64,
        concat_excel: bool = True,
        reduction_method: str = 'scaled_pca',
        enable_grid_search: bool = True,
        variance_threshold: float = AUTO_VARIANCE_THRESHOLD,
        split_method: str = 'time',
        test_size: float = DEFAULT_TEST_SIZE
) -> None:
    """
    主流程：数据加载 → 训练/Hold-out 划分 → 各折降维 → 标准 k 折超参数
    选择（每配置取五折平均，选最高）→ 用选定配置逐折评估 → CV 汇总 →
    全量训练集重训练 → Internal Hold-out 评估 → Bootstrap 95% CI

    超参数选择策略：对每个候选超参数配置，计算其在全部 5 个外层折上的平均
    验证表现，选取平均表现最高的配置；所有折共用该选定配置。降维仍在每折
    内部独立拟合以防止数据泄露。

    参数:
        labels_path: 标签文件路径
        raw_vectors_path: 原始向量文件路径
        output_dir: 输出根目录
        excel_path: Excel 特征文件路径（split_method='time' 时必需，
            因为需要读取"检验日期"列）
        excel_sheet: Excel 工作表名
        n_components: 降维目标维度
        concat_excel: 是否拼接 Excel 特征
        reduction_method: 降维方式
        enable_grid_search: 是否启用网格搜索
        variance_threshold: auto 模式方差阈值
        split_method: 训练/Hold-out 划分方式，可选:
            - 'time'  : 基于 Excel 中"检验日期"按年份划分
                        (2010-2019 训练+验证集 vs 2020-2025 内部保留验证集)
            - 'random': 分层随机按 (1-test_size):test_size 划分
        test_size: random 模式下保留验证集所占比例，默认 0.2
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("📂 Loading data...")
    X_raw = load_raw_vectors(raw_vectors_path)
    y = load_labels(labels_path)

    X_excel = None
    if concat_excel and excel_path is not None:
        X_excel = load_excel_features(excel_path, sheet_name=excel_sheet)
        if len(X_excel) != len(X_raw):
            sys.exit(f'行数不一致！raw_vectors={len(X_raw)} vs excel={len(X_excel)}')
        print(f"✅ Excel 特征维度: {X_excel.shape[1]}")

    if '3cl' in labels_path:
        class_names = ['AILD', 'DILI', 'CHB']
        class_label = [0, 1, 2]
    else:
        class_names = ['AIH', 'PBC', 'DILI', 'CHB']
        class_label = [0, 1, 2, 3]

    num_class = len(class_label)
    print(f"✅ 原始数据 shape: X_raw={X_raw.shape}, y={y.shape}")
    print(f"✅ 降维方式={reduction_method}, GridSearch={enable_grid_search}")

    # --------------------------------------------------------------------
    #  训练+验证 / Internal Hold-out 划分（两种方式可选）
    # --------------------------------------------------------------------
    train_val_idx, test_idx = make_split_indices(
        y=y,
        split_method=split_method,
        excel_path=excel_path,
        excel_sheet=excel_sheet,
        test_size=test_size,
        random_seed=RANDOM_SEED
    )

    X_raw_train_val = X_raw[train_val_idx]
    y_train_val = y[train_val_idx]
    X_raw_test = X_raw[test_idx]
    y_test_holdout = y[test_idx]
    X_excel_train_val = X_excel[train_val_idx] if X_excel is not None else None
    X_excel_test = X_excel[test_idx] if X_excel is not None else None

    print(f"✅ 训练集: {X_raw_train_val.shape[0]} 样本, "
          f"Hold-out验证集: {X_raw_test.shape[0]} 样本")

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    algorithms = get_algorithm_classifiers(
        RANDOM_SEED, num_class=num_class, enable_grid_search=enable_grid_search)

    # ====================================================================
    #  阶段一：对每个外层折独立完成降维（防泄露），收集各折降维后特征。
    #          降维诊断、缓存、Excel 拼接、建模数据与折索引保存均在此阶段完成。
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("🔧 Stage 1: Per-fold dimensionality reduction")
    print(f"{'=' * 60}")

    fold_feature_sets = []          # 每折降维/拼接后的特征字典
    fold_reducers = []              # 每折 reducer
    fold_pre_scalers = []           # 每折 pre_scaler
    fold_indices = []               # 每折 (train_idx, val_idx)

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_raw_train_val, y_train_val)):
        print(f"\n{'-' * 60}")
        print(f"🎯 Fold {fold_idx + 1}/{N_SPLITS} - Reduction")
        print(f"{'-' * 60}")

        X_raw_train = X_raw_train_val[train_idx]
        X_raw_val = X_raw_train_val[val_idx]
        y_train = y_train_val[train_idx]
        y_val = y_train_val[val_idx]

        excel_train = X_excel_train_val[train_idx] if X_excel_train_val is not None else None
        excel_val = X_excel_train_val[val_idx] if X_excel_train_val is not None else None

        cached = try_load_cached_reduction(output_dir, fold_idx)
        if cached is not None:
            X_train_reduced, X_val_reduced, reducer, pre_scaler = cached
        else:
            print(f"  🔧 Fitting reduction on training fold {fold_idx + 1}...")
            X_train_reduced, X_val_reduced, reducer, pre_scaler = \
                fit_and_transform_reduction(
                    X_raw_train, X_raw_val, y_train,
                    method=reduction_method,
                    n_components=n_components,
                    variance_threshold=variance_threshold,
                    random_state=RANDOM_SEED
                )

            if reducer is not None:
                save_reduction_diagnostics(
                    reducer=reducer, pre_scaler=pre_scaler,
                    X_train_reduced=X_train_reduced, X_val_reduced=X_val_reduced,
                    y_train=y_train, y_val=y_val,
                    method=reduction_method, fold_idx=fold_idx,
                    output_dir=output_dir, class_names=class_names, class_label=class_label
                )

        if excel_train is not None and excel_val is not None:
            X_train = np.hstack([X_train_reduced, excel_train])
            X_val = np.hstack([X_val_reduced, excel_val])
            print(f"  拼接 Excel 特征后维度: {X_train.shape[1]}")
        else:
            X_train = X_train_reduced
            X_val = X_val_reduced

        print(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}")

        # ===== 保存当前折 训练集/验证集 建模数据 =====
        fold_data_dir = get_modeling_data_dir(output_dir, fold_idx)
        fold_data_dir.mkdir(parents=True, exist_ok=True)
        save_modeling_data(X_train, y_train, fold_data_dir / "fold_train_modeling_data.csv", "训练集")
        save_modeling_data(X_val, y_val, fold_data_dir / "fold_val_modeling_data.csv", "验证集")

        fold_split_dir = Path(output_dir) / "fold_splits"
        fold_split_dir.mkdir(exist_ok=True)
        np.savetxt(fold_split_dir / f"train_idx_fold{fold_idx + 1}.txt", train_idx, fmt="%d")
        np.savetxt(fold_split_dir / f"val_idx_fold{fold_idx + 1}.txt", val_idx, fmt="%d")

        fold_feature_sets.append({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val
        })
        fold_reducers.append(reducer)
        fold_pre_scalers.append(pre_scaler)
        fold_indices.append((train_idx, val_idx))

    # ====================================================================
    #  阶段二：标准 k 折交叉验证超参数选择。
    #          对每个算法的**每一组候选配置**，计算其在全部 5 折上的平均验证
    #          表现，选取平均最高的配置；所有折共用该选定配置。
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("🔎 Stage 2: Hyperparameter selection by 5-fold mean validation "
          "performance")
    print(f"{'=' * 60}")

    selected_hyperparams = {}
    for algo_name, algo_config in algorithms.items():
        print(f"\n🔧 Selecting hyperparameters for {algo_name}...")
        best_params, cv_mean_score, _ = select_best_hyperparams_by_cv_mean(
            base_clf=algo_config['classifier'],
            param_grid=algo_config.get('param_grid'),
            fold_feature_sets=fold_feature_sets,
            algo_name=algo_name,
            output_dir=output_dir,
            needs_scaling=algo_config['needs_scaling'],
            use_sample_weight=algo_config.get('use_sample_weight', False),
            scoring=HYPERPARAM_SELECTION_SCORING
        )
        selected_hyperparams[algo_name] = {
            'params': best_params,
            'cv_mean_score': cv_mean_score
        }

    # ====================================================================
    #  阶段三：使用每个算法选定的固定超参数配置，在 5 折上逐折训练并评估。
    #          所有折共用同一组超参数，用于汇报 CV 各项指标的均值 ± 标准差。
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("📊 Stage 3: Re-evaluate selected configuration on all folds")
    print(f"{'=' * 60}")

    all_results = {
        algo_name: {
            'fold_results': [],
            'models': [],
            'scalers': [],
            'reducers': [],
            'pre_scalers': [],
            'mean_auc_per_fold': [],
            'accuracy_per_fold': []
        } for algo_name in algorithms.keys()
    }

    for fold_idx, fset in enumerate(fold_feature_sets):
        print(f"\n{'-' * 60}")
        print(f"🎯 Fold {fold_idx + 1}/{N_SPLITS} - Evaluation "
              f"(selected hyperparameters)")
        print(f"{'-' * 60}")

        X_train = fset['X_train']
        y_train = fset['y_train']
        X_val = fset['X_val']
        y_val = fset['y_val']
        reducer = fold_reducers[fold_idx]
        pre_scaler = fold_pre_scalers[fold_idx]

        for algo_name, algo_config in algorithms.items():
            sel = selected_hyperparams[algo_name]
            clf = clone(algo_config['classifier'])
            if sel['params']:
                clf.set_params(**sel['params'])

            if algo_config['needs_scaling']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
            else:
                scaler = None
                X_train_scaled = X_train
                X_val_scaled = X_val

            clf, results = evaluate_algorithm(
                clf, X_train_scaled, y_train, X_val_scaled, y_val,
                algo_name, class_names, class_label, output_dir, fold_idx,
                selected_params=sel['params'],
                use_sample_weight=algo_config.get('use_sample_weight', False),
                cv_mean_score=sel['cv_mean_score']
            )

            all_results[algo_name]['fold_results'].append(results)
            all_results[algo_name]['models'].append(clf)
            all_results[algo_name]['scalers'].append(scaler)
            all_results[algo_name]['reducers'].append(reducer)
            all_results[algo_name]['pre_scalers'].append(pre_scaler)
            all_results[algo_name]['mean_auc_per_fold'].append(results['mean_auc'])
            all_results[algo_name]['accuracy_per_fold'].append(results['accuracy'])

    # ====================================================================
    #  五折 CV 汇总（所有折共用选定超参数）
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("📈 5-Fold Cross-Validation Average Performance")
    print(f"{'=' * 60}")

    for avg_mode in ['weighted', 'macro']:
        prec_key = f'precision_{avg_mode}'
        rec_key = f'recall_{avg_mode}'
        f1_key = f'f1_score_{avg_mode}'
        sens_key = f'sensitivity_{avg_mode}'
        spec_key = f'specificity_{avg_mode}'
        auc_key = f'auc_{avg_mode}'

        summary_data = []
        for algo_name in algorithms.keys():
            fold_results = all_results[algo_name]['fold_results']

            fold_aucs = [r[auc_key] for r in fold_results]
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)

            fold_accs = all_results[algo_name]['accuracy_per_fold']
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)

            fold_precs = [r[prec_key] for r in fold_results]
            mean_prec = np.mean(fold_precs)
            std_prec = np.std(fold_precs)

            fold_recs = [r[rec_key] for r in fold_results]
            mean_rec = np.mean(fold_recs)
            std_rec = np.std(fold_recs)

            fold_f1s = [r[f1_key] for r in fold_results]
            mean_f1 = np.mean(fold_f1s)
            std_f1 = np.std(fold_f1s)

            fold_sens = [r[sens_key] for r in fold_results]
            fold_spec = [r[spec_key] for r in fold_results]
            mean_sens, std_sens = np.mean(fold_sens), np.std(fold_sens)
            mean_spec, std_spec = np.mean(fold_spec), np.std(fold_spec)

            fold_times = [r['training_time'] for r in fold_results]
            mean_time = np.mean(fold_times)

            if avg_mode == 'weighted':
                print(f"\n📊 {algo_name}:")
                print(f"  📊 AUC:        {mean_auc:.3f} ± {std_auc:.3f}")
                print(f"  🎯 Accuracy:   {mean_acc:.3f} ± {std_acc:.3f}")
                print(f"  📏 Precision:  {mean_prec:.3f} ± {std_prec:.3f}")
                print(f"  📐 Recall:     {mean_rec:.3f} ± {std_rec:.3f}")
                print(f"  ⚖️  F1-Score:   {mean_f1:.3f} ± {std_f1:.3f}")
                print(f"  🩺 Sensitivity:{mean_sens:.3f} ± {std_sens:.3f}")
                print(f"  🧪 Specificity:{mean_spec:.3f} ± {std_spec:.3f}")
                print(f"  ⏱️  Avg Training Time: {mean_time:.2f}s")

            summary_data.append({
                'Algorithm': algo_name,
                'Average_AUC': mean_auc, 'AUC_Std': std_auc,
                'Average_Accuracy': mean_acc, 'Accuracy_Std': std_acc,
                'Average_Precision': mean_prec, 'Precision_Std': std_prec,
                'Average_Recall': mean_rec, 'Recall_Std': std_rec,
                'Average_F1_Score': mean_f1, 'F1_Score_Std': std_f1,
                'Average_Sensitivity': mean_sens, 'Sensitivity_Std': std_sens,
                'Average_Specificity': mean_spec, 'Specificity_Std': std_spec,
                'Average_Training_Time(s)': mean_time
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = Path(output_dir) / f"algorithm_comparison_summary_{avg_mode}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[✓] Algorithm comparison results ({avg_mode}) saved to: {summary_path}")

    summary_weighted_path = Path(output_dir) / "algorithm_comparison_summary_weighted.csv"
    summary_df_weighted = pd.read_csv(summary_weighted_path)
    plot_cv_results(summary_df_weighted, output_dir)

    for avg_mode in ['weighted', 'macro']:
        prec_key = f'precision_{avg_mode}'
        rec_key = f'recall_{avg_mode}'
        f1_key = f'f1_score_{avg_mode}'
        sens_key = f'sensitivity_{avg_mode}'
        spec_key = f'specificity_{avg_mode}'
        auc_key = f'auc_{avg_mode}'

        detail_rows = []
        for algo_name in algorithms.keys():
            for fold, res in enumerate(all_results[algo_name]['fold_results'], 1):
                detail_rows.append({
                    "Algorithm": algo_name, "Fold": fold,
                    "Accuracy": res["accuracy"],
                    "Precision": res[prec_key], "Recall": res[rec_key],
                    "F1_Score": res[f1_key],
                    "Sensitivity": res[sens_key], "Specificity": res[spec_key],
                    "AUC": res[auc_key],
                    "Training_Time": res["training_time"],
                    "Hyperparam_CV_Mean_Score": res.get("hyperparam_cv_mean_score"),
                })

        detail_report_path = Path(output_dir) / f"cv_detailed_reports_{avg_mode}.csv"
        pd.DataFrame(detail_rows).to_csv(detail_report_path, index=False)
        print(f"[✓] 5-fold detailed reports ({avg_mode}) saved -> {detail_report_path}")

    # 保存超参数选择报告（基于五折平均验证表现选定的配置）
    save_best_params_report(
        all_results, selected_hyperparams, algorithms,
        output_dir, reduction_method, n_components, enable_grid_search
    )

    # ====================================================================
    #  阶段四：Internal Hold-out Validation
    #          全量训练集拟合降维器 + 用选定超参数重训练 + 评估
    # ====================================================================
    print(f"\n{'=' * 60}")
    print("🧪 Stage 4: Full-training-set retrain & Internal Hold-out "
          "Validation")
    print(f"{'=' * 60}")

    cached_full = try_load_cached_reduction(output_dir, fold_idx=-1)
    if cached_full is not None:
        X_tv_reduced, _, tv_reducer, tv_pre_scaler = cached_full
        if tv_pre_scaler is not None:
            X_test_prescaled = tv_pre_scaler.transform(X_raw_test)
        else:
            X_test_prescaled = X_raw_test
        X_test_reduced = tv_reducer.transform(X_test_prescaled) if tv_reducer is not None else X_test_prescaled
    else:
        print("  🔧 全量训练集拟合降维器...")
        X_tv_reduced, _, tv_reducer, tv_pre_scaler = \
            fit_and_transform_reduction(
                X_raw_train_val, X_raw_train_val, y_train_val,
                method=reduction_method, n_components=n_components,
                variance_threshold=variance_threshold, random_state=RANDOM_SEED
            )
        if tv_pre_scaler is not None:
            X_test_prescaled = tv_pre_scaler.transform(X_raw_test)
        else:
            X_test_prescaled = X_raw_test
        X_test_reduced = tv_reducer.transform(X_test_prescaled) if tv_reducer is not None else X_test_prescaled

        if tv_reducer is not None:
            save_reduction_diagnostics(
                reducer=tv_reducer, pre_scaler=tv_pre_scaler,
                X_train_reduced=X_tv_reduced, X_val_reduced=X_test_reduced,
                y_train=y_train_val, y_val=y_test_holdout,
                method=reduction_method, fold_idx=-1,
                output_dir=output_dir, class_names=class_names, class_label=class_label
            )

    if X_excel_train_val is not None:
        X_tv_features = np.hstack([X_tv_reduced, X_excel_train_val])
        X_test_features = np.hstack([X_test_reduced, X_excel_test])
    else:
        X_tv_features = X_tv_reduced
        X_test_features = X_test_reduced

    # ===== 保存最终全量训练集 + 独立测试集建模数据 =====
    final_data_dir = get_modeling_data_dir(output_dir)
    final_data_dir.mkdir(parents=True, exist_ok=True)
    save_modeling_data(X_tv_features, y_train_val, final_data_dir / "final_full_train_data.csv", "最终全量训练集")
    save_modeling_data(X_test_features, y_test_holdout, final_data_dir / "final_holdout_test_data.csv", "独立测试集")

    # 用字典存储每个算法的训练集指标，避免变量作用域问题
    train_metrics_per_algo = {}

    # best_models：使用选定超参数在全量训练集上重训练的模型
    best_model_dir = Path(output_dir) / "best_models"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    # 全量重训练模型保存目录
    final_model_dir = Path(output_dir) / "final_retrained_models"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    test_predictions = {}
    test_results = []
    for algo_name, algo_config in algorithms.items():
        print(f"\n🔍 Retraining & testing {algo_name} "
              f"(selected hyperparameters)...")
        sel = selected_hyperparams[algo_name]
        clf = clone(algo_config['classifier'])
        if sel['params']:
            clf.set_params(**sel['params'])

        if algo_config['needs_scaling']:
            scaler = StandardScaler()
            X_tv_scaled = scaler.fit_transform(X_tv_features)
            X_test_scaled = scaler.transform(X_test_features)
        else:
            scaler = None
            X_tv_scaled = X_tv_features
            X_test_scaled = X_test_features

        if algo_config.get('use_sample_weight', False):
            sw = compute_sample_weight('balanced', y_train_val)
            clf.fit(X_tv_scaled, y_train_val, sample_weight=sw)
        else:
            clf.fit(X_tv_scaled, y_train_val)

        # CV 五折平均指标（用于报告与对比图的 CV 列）
        fold_results = all_results[algo_name]['fold_results']
        cv_mean_auc = float(np.mean([r['mean_auc'] for r in fold_results]))

        # ===== 保存 best_models（选定配置 + 全量重训练）=====
        best_bundle = {
            'model': clf,
            'reducer': tv_reducer,
            'pre_scaler': tv_pre_scaler,
            'scaler': scaler,
            'reduction_method': reduction_method,
            'n_components': n_components,
            'class_names': class_names,
            'class_label': class_label,
            'selected_params': sel['params'],
            'hyperparam_cv_mean_score': sel['cv_mean_score'],
            'cv_mean_auc': cv_mean_auc,
            'selection_strategy': 'standard_kfold_cv_mean',
        }
        best_model_path = best_model_dir / f'best_{algo_name}_model.pkl'
        joblib.dump(best_bundle, best_model_path)
        print(f"  💾 Best model saved to: {best_model_path}")

        # ===== 保存 final_retrained_models =====
        retrained_bundle = {
            'model': clf,
            'reducer': tv_reducer,
            'pre_scaler': tv_pre_scaler,
            'scaler': scaler,
            'reduction_method': reduction_method,
            'n_components': n_components,
            'class_names': class_names,
            'class_label': class_label,
            'selected_params': sel['params'],
            'hyperparam_cv_mean_score': sel['cv_mean_score'],
            'cv_mean_auc': cv_mean_auc,
        }
        retrained_path = final_model_dir / f'final_{algo_name}_model.pkl'
        joblib.dump(retrained_bundle, retrained_path)
        print(f"  💾 全量重训练模型已保存: {retrained_path}")

        # 训练集评估
        y_train_pred = clf.predict(X_tv_scaled)
        y_train_score = clf.predict_proba(X_tv_scaled)
        train_report_dir = Path(output_dir) / "train_reports"
        train_report_dir.mkdir(parents=True, exist_ok=True)
        train_cls_report = classification_report(
            y_train_val, y_train_pred, digits=4, target_names=class_names)
        train_report_path = train_report_dir / f'{algo_name}_train_report.txt'

        train_acc = accuracy_score(y_train_val, y_train_pred)
        train_prec = precision_score(y_train_val, y_train_pred, average='weighted')
        train_rec = recall_score(y_train_val, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train_val, y_train_pred, average='weighted')
        cm_train = confusion_matrix(y_train_val, y_train_pred)
        train_sens, train_spec = compute_weighted_sens_spec(cm_train)
        y_train_bin = label_binarize(y_train_val, classes=class_label)
        train_auc_dict = {}
        for ci in range(len(class_label)):
            if np.sum(y_train_bin[:, ci]) > 0:
                fpr_i, tpr_i, _ = roc_curve(y_train_bin[:, ci], y_train_score[:, ci])
                train_auc_dict[ci] = auc(fpr_i, tpr_i)
        train_w_auc = compute_weighted_auc(train_auc_dict, y_train_bin)

        # 将训练集指标存入字典，避免变量作用域问题
        train_metrics_per_algo[algo_name] = {
            'accuracy': train_acc,
            'f1': train_f1,
            'auc': train_w_auc,
        }

        with open(train_report_path, 'w', encoding='utf-8') as f:
            f.write(f"{algo_name} - Classification Report on Full Training Set:\n")
            f.write("=" * 50 + "\n")
            f.write(train_cls_report)
            f.write("\n\n--- 训练集指标汇总 ---\n")
            f.write(f"Accuracy:    {train_acc:.3f}\n")
            f.write(f"Precision:   {train_prec:.3f}\n")
            f.write(f"Recall:      {train_rec:.3f}\n")
            f.write(f"F1-Score:    {train_f1:.3f}\n")
            f.write(f"Sensitivity: {train_sens:.3f}\n")
            f.write(f"Specificity: {train_spec:.3f}\n")
            f.write(f"AUC:         {train_w_auc:.3f}\n")
        print(f"  📄 Train report saved: {train_report_path}")

        # 测试集评估
        y_pred = clf.predict(X_test_scaled)
        y_score = clf.predict_proba(X_test_scaled)
        y_test_bin = label_binarize(y_test_holdout, classes=class_label)
        test_predictions[algo_name] = {'y_true': y_test_holdout, 'y_pred': y_pred, 'y_score': y_score}

        accuracy = accuracy_score(y_test_holdout, y_pred)
        precision = precision_score(y_test_holdout, y_pred, average='weighted')
        recall = recall_score(y_test_holdout, y_pred, average='weighted')
        f1 = f1_score(y_test_holdout, y_pred, average='weighted')
        cm_test = confusion_matrix(y_test_holdout, y_pred)
        sensitivity_test, specificity_test = compute_weighted_sens_spec(cm_test)

        print(f"\n{algo_name} Classification Report on Internal Hold-out Validation Set:")
        print(classification_report(y_test_holdout, y_pred, digits=4, target_names=class_names))
        report_dir = Path(output_dir) / "holdout_val_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f'{algo_name}_holdout_val_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{algo_name} - Classification Report on Internal Hold-out Validation Set:\n")
            f.write("=" * 50 + "\n")
            f.write(classification_report(y_test_holdout, y_pred, digits=4, target_names=class_names))

        cm = confusion_matrix(y_test_holdout, y_pred)
        plot_confusion(cm, class_names, algo_name, output_dir, fold_idx='holdout_val')
        auc_dict = plot_roc(y_test_bin, y_score, class_names, algo_name, output_dir, fold_idx='holdout_val')
        mean_auc = compute_weighted_auc(auc_dict, y_test_bin)

        # 使用字典取训练集指标，避免作用域问题
        t_metrics = train_metrics_per_algo[algo_name]
        test_results.append({
            'Algorithm': algo_name, 'Accuracy': accuracy,
            'Sensitivity': sensitivity_test, 'Specificity': specificity_test,
            'F1_Score': f1, 'AUC': mean_auc,
            'CV_AUC': cv_mean_auc,
            'Precision': precision, 'Recall': recall,
            'Train_Accuracy': t_metrics['accuracy'],
            'Train_F1': t_metrics['f1'],
            'Train_AUC': t_metrics['auc']
        })

        print(f"  ✅ Accuracy: {accuracy:.3f}")
        print(f"  ✅ F1-Score: {f1:.3f}")
        print(f"  ✅ AUC: {mean_auc:.3f}")
        print(f"  ✅ Sensitivity: {sensitivity_test:.3f}")
        print(f"  ✅ Specificity: {specificity_test:.3f}")
        print(f"  📄 Report saved: {report_path}")

    # ====================================================================
    #  Bootstrap 95% CI
    # ====================================================================
    print(f"\n{'=' * 60}")
    print(f"📊 Computing 95% CI on Internal Hold-out Validation Set (Bootstrap, n={N_BOOTSTRAP})")
    print(f"{'=' * 60}")

    ci_dir = Path(output_dir) / "confidence_intervals"
    ci_dir.mkdir(exist_ok=True)
    all_ci_results = {}
    for algo_name in algorithms.keys():
        preds = test_predictions[algo_name]
        ci_results = bootstrap_ci_95(
            y_true=preds['y_true'], y_pred=preds['y_pred'], y_score=preds['y_score'],
            class_label=class_label, random_state=RANDOM_SEED
        )
        all_ci_results[algo_name] = ci_results

    for avg_mode in ['weighted', 'macro']:
        ci_rows = []
        for algo_name in algorithms.keys():
            ci_results_mode = all_ci_results[algo_name][avg_mode]
            print(f"\n📊 {algo_name} (Hold-out Validation, Bootstrap 95% CI, {avg_mode}):")
            row = {"Algorithm": algo_name}
            for metric_name, (mean_val, lower, upper) in ci_results_mode.items():
                ci_str = f"{mean_val:.3f} [{lower:.3f}, {upper:.3f}]"
                row[f"{metric_name}_CI95"] = ci_str
                print(f"  {metric_name}: {ci_str}")
            ci_rows.append(row)
        ci_df = pd.DataFrame(ci_rows)
        ci_path = ci_dir / f"best_models_95CI_{avg_mode}.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"\n[✓] 95% CI ({avg_mode}, Bootstrap on hold-out val) saved -> {ci_path}")

    test_results_df = pd.DataFrame(test_results)
    test_results_path = Path(output_dir) / "holdout_val_results_comparison.csv"
    test_results_df.to_csv(test_results_path, index=False)

    plot_algorithm_comparison(test_results_df, output_dir)
    plot_auc_comparison(test_results_df, output_dir)

    config_dict = {
        'reduction_method': reduction_method, 'n_components': n_components,
        'variance_threshold': variance_threshold, 'enable_grid_search': enable_grid_search,
        'hyperparam_selection_strategy': 'standard_kfold_cv_mean',
        'hyperparam_selection_scoring': HYPERPARAM_SELECTION_SCORING,
        'n_splits': N_SPLITS,
        'random_seed': RANDOM_SEED, 'n_bootstrap': N_BOOTSTRAP,
        'bootstrap_ci_level': BOOTSTRAP_CI_LEVEL,
        'labels_path': labels_path, 'raw_vectors_path': raw_vectors_path,
        'excel_path': excel_path, 'excel_sheet': excel_sheet, 'concat_excel': concat_excel,
        'class_names': class_names, 'class_label': class_label,
        'split_method': split_method,
        'test_size': test_size if split_method == 'random' else None,
    }
    config_path = Path(output_dir) / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"[✓] 全局配置已保存: {config_path}")

    print(f"\n{'=' * 60}")
    print("🎉 All algorithm evaluations completed!")
    print(f"📂 建模数据已保存至: {Path(output_dir) / 'modeling_data'}")
    print(f"📂 全量重训练模型已保存至: {final_model_dir}")
    print(f"{'=' * 60}")


# ========================================================================
#  命令行入口（argparse）
# ========================================================================
def _str2bool(value: str) -> bool:
    """
    将命令行字符串参数解析为布尔值

    参数:
        value: 命令行输入的字符串，如 'true'/'false'/'1'/'0'

    返回值:
        解析后的布尔值

    异常:
        argparse.ArgumentTypeError: 当输入无法识别为布尔值时
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析为布尔值: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器，参数项与 run_pipeline 函数签名保持一致

    返回值:
        argparse.ArgumentParser 实例
    """
    parser = argparse.ArgumentParser(
        description="多算法分类建模：折内降维 + 标准 k 折超参数选择 + "
                    "Internal Hold-out 评估 + Bootstrap 95% CI"
    )

    # ---------- 必需参数 ----------
    parser.add_argument(
        "--labels_path", type=str, required=True,
        help="标签文件路径（.txt，整型标签，逐行一个）")
    parser.add_argument(
        "--raw_vectors_path", type=str, required=True,
        help="原始向量文件路径（空格或制表符分隔的向量文件）")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="输出根目录，所有中间结果与最终报告均保存于此")

    # ---------- 可选参数（默认值与原 main() 保持一致） ----------
    parser.add_argument(
        "--excel_path", type=str, default=None,
        help="Excel 特征文件路径。split_method='time' 时必需（读取"
             "\"检验日期\"列）；concat_excel=True 时用于拼接特征。默认: None")
    parser.add_argument(
        "--excel_sheet", type=str, default="Sheet2",
        help="Excel 工作表名，默认: Sheet2")
    parser.add_argument(
        "--n_components", type=int, default=64,
        help="降维目标维度，默认: 64")
    parser.add_argument(
        "--concat_excel", type=_str2bool, default=True,
        help="是否拼接 Excel 特征（true/false），默认: true")
    parser.add_argument(
        "--reduction_method", type=str, default="scaled_pca",
        choices=["pca", "scaled_pca", "auto", "lda", "umap", "none"],
        help="降维方式，默认: scaled_pca")
    parser.add_argument(
        "--enable_grid_search", type=_str2bool, default=True,
        help="是否启用 GridSearch 超参数搜索（true/false），默认: true")
    parser.add_argument(
        "--variance_threshold", type=float, default=AUTO_VARIANCE_THRESHOLD,
        help=f"auto 降维模式的方差阈值，默认: {AUTO_VARIANCE_THRESHOLD}")
    parser.add_argument(
        "--split_method", type=str, default="time",
        choices=["time", "random"],
        help="训练/Hold-out 划分方式：time（基于检验日期年份）或 random"
             "（分层随机），默认: time")
    parser.add_argument(
        "--test_size", type=float, default=DEFAULT_TEST_SIZE,
        help=f"split_method='random' 时保留验证集比例，默认: "
             f"{DEFAULT_TEST_SIZE}")

    return parser


def main() -> None:
    """
    命令行入口主函数：解析参数并调用 run_pipeline 执行完整流水线
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        labels_path=args.labels_path,
        raw_vectors_path=args.raw_vectors_path,
        output_dir=args.output_dir,
        excel_path=args.excel_path,
        excel_sheet=args.excel_sheet,
        n_components=args.n_components,
        concat_excel=args.concat_excel,
        reduction_method=args.reduction_method,
        enable_grid_search=args.enable_grid_search,
        variance_threshold=args.variance_threshold,
        split_method=args.split_method,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
