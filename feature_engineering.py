import pandas as pd
import numpy as np


def load_and_preprocess(filepath, is_training=True):
    """
    加载并清洗数据，统一列名和单位。
    """
    df = pd.read_csv(filepath)
    data = df.copy()

    if is_training:
        # 处理 concat.csv (训练集)
        # 重命名列以匹配统一标准
        rename_map = {
            'Weight (kg)': 'Weight',
            'Avg_BPM': 'Heart_Rate',
            'Calories_Burned': 'Target',
            'Session_Duration (hours)': 'Duration_Hours',
            'Height (m)': 'Height_m',
            'Age': 'Age',
            'Gender': 'Gender'
        }
        # 仅重命名存在的列
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

        # 单位转换：小时 -> 分钟
        if 'Duration_Hours' in data.columns:
            data['Duration_min'] = data['Duration_Hours'] * 60
        # 单位转换：米 -> 厘米
        if 'Height_m' in data.columns:
            data['Height_cm'] = data['Height_m'] * 100

    else:
        # 处理 calories.csv (验证集)
        rename_map = {
            'Calories_Burned': 'Target',
            'Duration': 'Duration_min',  # 已经是分钟
            'Height': 'Height_cm'  # 已经是厘米
        }
        data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})

    # 统一处理性别 (0: Male, 1: Female 或其他)
    # 假设 'Male'/'male' 为 0，其他为 1。根据您的数据调整。
    data['Gender_Male'] = data['Gender'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)

    return data


def create_interaction_features(df):
    """
    构造物理交互特征。
    核心逻辑：能量消耗 = 功率(由体重、心率决定) * 时间
    """
    df_feat = pd.DataFrame()

    # 确保所需列存在
    required_cols = ['Duration_min', 'Heart_Rate', 'Weight', 'Age', 'Gender_Male']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # === 交互特征 (Physics-Informed Interaction Terms) ===
    # 1. 时长 * 心率 (最基础的强度项)
    df_feat['Dur_HR'] = df['Duration_min'] * df['Heart_Rate']

    # 2. 时长 * 心率平方 (模拟高心率下非线性的能量消耗激增)
    df_feat['Dur_HR2'] = df['Duration_min'] * (df['Heart_Rate'] ** 2)

    # 3. 时长 * 体重 (体重越大，做功越多)
    df_feat['Dur_Wt'] = df['Duration_min'] * df['Weight']

    # 4. 时长 * 年龄 (年龄影响代谢率)
    df_feat['Dur_Age'] = df['Duration_min'] * df['Age']

    # 5. 时长 * 性别
    df_feat['Dur_Gen'] = df['Duration_min'] * df['Gender_Male']

    # 6. 原始时长 (作为基础缩放因子)
    df_feat['Duration'] = df['Duration_min']

    return df_feat