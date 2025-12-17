import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from feature_engineering import create_interaction_features, load_and_preprocess


def evaluate_single_dataset(modelname, type="Internal",dataset_type='Mini'):
    print(f"\n{type} Evaluating on : {dataset_type} Dataset ...")

    # 1. 加载数据
    if type == "Internal":
        df = pd.read_csv(f"{modelname}_test_set.csv")
    else:
        df =load_and_preprocess('original_dataset/calories.csv', is_training=False)

    # 2. 生成特征
    X_val = create_interaction_features(df)
    y_val = df['Target']

    # 3. 加载所有模型
    lr_model = joblib.load(modelname + '_lr_model_physics.joblib')
    rf_model = joblib.load(modelname + '_rf_model.joblib')  # <--- 加载 RF
    nn_model = joblib.load(modelname + '_nn_model_interact.joblib')
    scaler = joblib.load('model/scaler_interact.joblib')

    # 4. 进行预测
    # LR 预测
    y_pred_lr = lr_model.predict(X_val)

    # RF 预测 (不需要 Scaler)
    y_pred_rf = rf_model.predict(X_val)  # <--- RF 预测

    # NN 预测 (需要 Scaler)
    X_val_scaled = scaler.transform(X_val)
    y_pred_nn = nn_model.predict(X_val_scaled)

    # 5. 计算指标
    metrics = {
        'Model': ['LR (No Intercept)', 'Random Forest', 'Neural Network'],  # <--- 加入 RF
        'R2 Score': [
            r2_score(y_val, y_pred_lr),
            r2_score(y_val, y_pred_rf),
            r2_score(y_val, y_pred_nn)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_val, y_pred_lr)),
            np.sqrt(mean_squared_error(y_val, y_pred_rf)),
            np.sqrt(mean_squared_error(y_val, y_pred_nn))
        ],
        'MAE': [
            mean_absolute_error(y_val, y_pred_lr),
            mean_absolute_error(y_val, y_pred_rf),
            mean_absolute_error(y_val, y_pred_nn)
        ]
    }
    metrics_df = pd.DataFrame(metrics)

    print(f"--- Results for {type} Evaluation on {dataset_type} Dataset ---")
    print(metrics_df)

    # 6. 可视化
    plot_evaluation(y_val, y_pred_lr, y_pred_rf, y_pred_nn, metrics_df, type,dataset_type)

    return metrics_df


def plot_evaluation(y_true, y_pred_lr, y_pred_rf, y_pred_nn, metrics_df, type,dataset_type):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig.suptitle(f'{type} Evaluation on {dataset_type} Dataset ', fontsize=16)

    # Plot 1: Scatter (True vs Pred)
    max_val = max(y_true.max(), y_pred_nn.max(), y_pred_lr.max(), y_pred_rf.max())

    # 绘制三个模型的散点
    axes.scatter(y_true, y_pred_nn, alpha=0.4, color='red', label='NN', s=20)
    axes.scatter(y_true, y_pred_lr, alpha=0.4, color='green', label='LR', s=20)
    axes.scatter(y_true, y_pred_rf, alpha=0.4, color='blue', label='RF', s=20)  # <--- RF 散点

    axes.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Ideal')
    axes.set_xlabel('Actual')
    axes.set_ylabel('Predicted')
    axes.set_title(f'{type} Prediction Scatter Plot on {dataset_type} Dataset')
    axes.legend()
    plt.tight_layout()
    plt.show()

    # Figure: Metrics Comparison
    fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(f'Model {type} Performance Comparison on {dataset_type} Dataset', fontsize=16)
    # Plot 2: R2 Score Comparison
    # 色板选择 viridis，会自动为3个柱子分配不同颜色
    sns.barplot(x='Model', y='R2 Score', data=metrics_df, ax=axes[0], palette='viridis')
    axes[0].set_title('R2 Score (Higher is Better)')
    axes[0].set_ylim(bottom=min(0, min(metrics_df['R2 Score']) - 0.1), top=1.0)

    # 在柱子上显示数值
    for i, v in enumerate(metrics_df['R2 Score']):
        axes[0].text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

    # RMSE Plot
    sns.barplot(x='Model', y='RMSE', data=metrics_df, ax=axes[1], palette='magma')
    axes[1].set_title('RMSE (Lower is Better)')

    # 在柱子上显示数值
    for i, v in enumerate(metrics_df['RMSE']):
        axes[1].text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

    # MAE Plot
    sns.barplot(x='Model', y='MAE', data=metrics_df, ax=axes[2], palette='rocket')
    axes[2].set_title('MAE (Lower is Better)')
    # 在柱子上显示数值
    for i, v in enumerate(metrics_df['MAE']):
        axes[2].text(i, v, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def run_full_evaluation(modelname,type='Internal',dataset_type='Mini'):
    print(f"Starting evaluation for model: {modelname}")
    if type == 'Internal':
        evaluate_single_dataset(modelname, "Internal",dataset_type)
    elif type == 'External':
        evaluate_single_dataset(modelname, "External",dataset_type)


if __name__ == "__main__":
    print("============================================")
    print("Evaluating Mini Concat Models")
    print("============================================")
    run_full_evaluation('model/mini','Internal','Expanded')
    run_full_evaluation('model/mini','External','Expanded')

    print("\n============================================")
    print("Evaluating Expanded Models")
    print("============================================")
    run_full_evaluation('model/expanded','Internal')
    run_full_evaluation('model/expanded','External')