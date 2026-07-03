# encoding: utf-8

'''
Copyright: EICN RDBIN RDBAI
Description:
    File Name   : 
    Description : 
    Dependency  : 
History:
    Author : Li, Xinming
    Date   : 
    Version: 
    Summary of Version: 
'''


import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score


# save_weights ='OS模型/0708PBC-OS-NORMAL-antiGAI-差异lasso/56'
# save_weights ='OS模型/0619PBC-NOMAL-ANTI原/96'
# save_weights ='OS模型/0620PBC-NOMAL-ANTIGAI/96/'
# save_weights ='OS模型/0624PBC-KANGTIYUAN差异分析/23/'
# save_weights ='OS模型/0626PBC-NORMAL-ANTIGAI-ACCURACY-quanbianlaing/222'
save_weights ='OS模型/0626PBC-NORMAL-ANTIGAI-ACCURACY-quanbianlaing/222'
# save_weights ='OS模型/0629PBC-normal-ANTIGAI-TRAIN差异分析+accuracy/187'

files = os.listdir(save_weights)
for file in files:
    # if 'RandomForestClassifier_weight' in file:
    if 'dataXtrain' in file and 'result' not in file:
        # print(file)
        train_X_LR = np.loadtxt(os.path.join(save_weights,file))
    if  'dataXtest' in file:
        test_X_LR = np.loadtxt(os.path.join(save_weights,file))
    if 'dataYtrain' in file:
        train_Y_LR = np.loadtxt(os.path.join(save_weights,file))
    if  'dataYtest' in file:
        test_Y_LR = np.loadtxt(os.path.join(save_weights,file))

# 确保标签为整数
train_Y_LR = train_Y_LR.astype(int)
test_Y_LR = test_Y_LR.astype(int)

# 模型配置字典
MODEL_CONFIG = [
    {
        'name': 'RF',
        'class': RandomForestClassifier,
        'param_grid': {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'random_state': [0]
        },
        'scale': False
    },
    {
        'name': 'LR',
        'class': LogisticRegression,
        'param_grid': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [1000],
            'random_state': [0]
        },
        'scale': True
    },
    # {
    #     'name': 'XGB',
    #     'class': XGBClassifier,
    #     'param_grid': {
    #         'max_depth': [3, 5, 7],
    #         'n_estimators': [100, 200, 300],
    #         'learning_rate': [0.01, 0.1],
    #         'random_state': [0]
    #     },
    #     'scale': False
    # },
    {
        'name': 'SVM',
        'class': SVC,
        'param_grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'probability': [True],
            # SVC does not support random_state for all kernels, only for some solvers
            'random_state': [0]
        },
        'scale': True
    },
    {
        'name': 'MLP',
        'class': MLPClassifier,
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001],
            'max_iter': [1000],
            'random_state': [0]
        },
        'scale': True
    }
]

# # 定义模型字典
# model_dict = {
#     "RandomForest": RandomForestClassifier(max_depth=3, n_estimators=200, random_state=0),
#     "LogisticRegression": LogisticRegression(max_iter=1000, random_state=0),
#     "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0),
#     "SVM": SVC(probability=True, random_state=0),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)
# }

# 创建保存路径
proba_dir = os.path.join(save_weights, 'proba_output')
os.makedirs(proba_dir, exist_ok=True)

# 五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

print("\n=== Model Evaluation with 5-fold CV on Train Set ===")

for config in MODEL_CONFIG:
    name = config['name']
    model_class = config['class']
    param_grid = config['param_grid']
    scale = config['scale']

    print(f"\n>>> Tuning and Evaluating {name}")

    # 使用 GridSearchCV 找到最优超参数
    grid = GridSearchCV(
        estimator=model_class(),
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        n_jobs=-1
    )
    grid.fit(train_X_LR, train_Y_LR)
    best_params = grid.best_params_
    best_score = grid.best_score_

    print(f"  => Best CV Accuracy: {best_score:.4f}")
    print(f"  => Best Parameters: {best_params}")

    # 使用最佳参数重新训练模型
    best_model = model_class(**best_params)
    best_model.fit(train_X_LR, train_Y_LR)

    # 在训练集上预测并保存概率
    pred_train = best_model.predict(train_X_LR)
    prob_train = best_model.predict_proba(train_X_LR)[:, 1]
    np.savetxt(os.path.join(proba_dir, f"{name}_train_proba"), prob_train, fmt="%.6f")

    # 在测试集上预测并保存概率
    pred_test = best_model.predict(test_X_LR)
    prob_test = best_model.predict_proba(test_X_LR)[:, 1]
    np.savetxt(os.path.join(proba_dir, f"{name}_test_proba"), prob_test, fmt="%.6f")

    # 打印分类报告
    report_train = classification_report(train_Y_LR, pred_train, digits=3)
    report_test = classification_report(test_Y_LR, pred_test, digits=3)

    # 打印并保存分类报告
    print(f"\n{name} - Final Evaluation with Best Params")
    print("Train Classification Report:")
    print(report_train)
    print("Test Classification Report:")
    print(report_test)
    # 保存报告到文件
    with open(os.path.join(proba_dir, f"{name}_classification_report"), 'w') as f:
        f.write(f"{name} - Final Evaluation with Best Params\n\n")
        f.write("Train Classification Report:\n")
        f.write(report_train + "\n")
        f.write("Test Classification Report:\n")
        f.write(report_test + "\n")