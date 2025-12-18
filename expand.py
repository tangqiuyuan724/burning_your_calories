import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
# load data
df=pd.read_csv('output/concat.csv')

target_rows = 20000
current_rows = df.shape[0]
rows_to_add = target_rows - current_rows

print(f"Current shape: {df.shape}")

df_large = pd.concat([df] * (rows_to_add // current_rows + 1), ignore_index=True)
df_large = df_large.sample(n=target_rows, replace=True, random_state=42).reset_index(drop=True)

numeric_cols = df_large.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    noise = np.random.normal(0, df_large[col].std() * 0.05, len(df_large))
    df_large[col] = np.round(df_large[col] + noise, 2)

categorical_cols = df_large.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    df_large[col] = np.random.choice(unique_vals, size=len(df_large), replace=True)

# 我们将 20% 的数据的时长强制修改为 0.1 - 0.5 小时 (6-30分钟)，以覆盖真实场景
mask_short = np.random.rand(len(df_large)) < 0.2
df_large.loc[mask_short, 'Session_Duration (hours)'] = np.random.uniform(0.1, 0.5, size=sum(mask_short))

df_large['BMI'] = np.round(df_large['Weight (kg)'] / (df_large['Height (m)'] ** 2), 2)

# 公式: Calories (kcal) ≈ MET * Weight (kg) * Duration (hr) * Intensity_Factor
def calc_calories(row):
    # MET 参考值: HIIT=8.0, Cardio=7.0, Strength=5.0, Yoga=3.0
    # 获取心率用于修正强度 (假设平均活跃心率为 120 BPM)
    # 物理逻辑：相同运动下，心率越高，实际消耗越大
    level_factor = max(1,row['Avg_BPM'] / 120)
    duration = row['Session_Duration (hours)']
    workout = row['Workout_Type']
    base = {
        'HIIT': 8.0,
        'Strength': 5.0,
        'Cardio': 7.0,
        'Yoga': 3.0
    }.get(workout, 7.0)
    weight = row['Weight (kg)']
    calories = base * weight * duration * level_factor
    # 添加 10% 的随机波动模拟个体代谢差异 (肌肉率等)
    return np.round(calories*np.random.uniform(0.9,1.1),2)
df_large['Calories_Burned'] = df_large.apply(calc_calories, axis=1)

df_large['Fat_Percentage'] = np.clip(
    25 + (df_large['BMI'] - 22) * 0.8 - df_large['Experience_Level'] * 0.5 +
    np.random.normal(0, 2, len(df_large)), 8, 35
)

#  Calories intake
df_large['Calories'] = np.round(
    (df_large['Weight (kg)'] * 25) +
    (df_large['Workout_Frequency (days/week)'] * 50) +
    (df_large['Physical exercise'] * 20) +
    np.random.normal(0, 100, len(df_large))
)

df_large['Age'] = np.clip(df_large['Age'], 18, 70)
df_large['Water_Intake (liters)'] = np.clip(df_large['Water_Intake (liters)'], 1, 5)
df_large['Experience_Level'] = np.clip(df_large['Experience_Level'], 1, 5)
# 扩大duration范围
df_large['Session_Duration (hours)'] = np.clip(df_large['Session_Duration (hours)'], 0.1, 3.0)

print(f"Final realistic shape: {df_large.shape}")
print(f"Information of expanded dataset: \n{df_large.info()}")
print(f"Description of expanded dataset: \n{df_large.describe().T}")

df_large.to_csv('output/expanded.csv', index=False)