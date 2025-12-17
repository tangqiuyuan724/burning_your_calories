import pandas as pd
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor  # <--- 新增
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_engineering import load_and_preprocess, create_interaction_features


def train_models(filename, modelname, test_size=0.2):
    print(f"Loading data from {filename}...")

    # 1. 加载和预处理全部数据
    full_df = load_and_preprocess(filename, is_training=True)

    # 2. 切分训练集和测试集
    train_df, test_df = train_test_split(full_df, test_size=test_size, random_state=42)

    print(f"Total samples: {len(full_df)}")
    print(f"Training set shape: {train_df.shape}")
    print(f"Testing set shape: {test_df.shape}")

    # 3. 保存切分后的数据集
    test_file_path = f"{modelname}_test_set.csv"
    test_df.to_csv(test_file_path, index=False)
    print(f"Datasets saved locally.")

    # 4. 生成交互特征
    X_train = create_interaction_features(train_df)
    y_train = train_df['Target']

    # ==========================================
    # 模型 A: 线性回归 (无截距)
    # ==========================================
    print("Training Linear Regression (No Intercept)...")
    lr_model = LinearRegression(fit_intercept=False)
    lr_model.fit(X_train, y_train)

    # ==========================================
    # 模型 B: 随机森林 (Random Forest) <--- 新增部分
    # ==========================================
    print("Training Random Forest...")
    # 随机森林不需要特征标准化，直接用 X_train
    rf_model = RandomForestRegressor(
        n_estimators=100,  # 树的数量
        max_depth=None,  # 深度不限
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心加速
    )
    rf_model.fit(X_train, y_train)

    # ==========================================
    # 模型 C: 神经网络 (需要标准化)
    # ==========================================
    print("Training Neural Network...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)

    # ==========================================
    # 保存模型文件
    # ==========================================
    print("Saving models...")
    joblib.dump(lr_model, modelname + '_lr_model_physics.joblib')
    joblib.dump(rf_model, modelname + '_rf_model.joblib')  # <--- 保存 RF
    joblib.dump(nn_model, modelname + '_nn_model_interact.joblib')
    joblib.dump(scaler, 'model/scaler_interact.joblib')

    print("Done! All 3 models saved.")


if __name__ == "__main__":

    print("--- Processing Mini Concatenation ---")
    train_models('output/concat.csv', 'model/mini')

    print("\n--- Processing Expanded Dataset ---")
    train_models('output/expanded.csv', 'model/expanded')