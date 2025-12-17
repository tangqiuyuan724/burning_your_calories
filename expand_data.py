import pandas as pd
import numpy as np

# load data
df=pd.read_csv('output/after_filling.csv')

target_rows = 20000
current_rows = df.shape[0]
rows_to_add = target_rows - current_rows

print(f"Current shape: {df.shape}")

df_large = pd.concat([df] * (rows_to_add // current_rows + 1), ignore_index=True)
df_large = df_large.sample(n=target_rows, replace=True, random_state=42).reset_index(drop=True)

numeric_cols = df_large.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    noise = np.random.normal(0, df_large[col].std() * 0.02, len(df_large))
    df_large[col] = np.round(df_large[col] + noise, 2)

categorical_cols = df_large.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    df_large[col] = np.random.choice(unique_vals, size=len(df_large), replace=True)

df_large['BMI'] = np.round(df_large['Weight (kg)'] / (df_large['Height (m)'] ** 2), 2)

def calc_calories(row):
    level_factor = 1 + (row['Experience_Level'] / 10)
    duration = row['Session_Duration (hours)']
    workout = row['Workout_Type']
    base = {
        'HIIT': 11,
        'Strength': 9,
        'Cardio': 8,
        'Yoga': 6
    }.get(workout, 8)
    return np.round(base * 100 * duration * level_factor, 2)

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
df_large['Session_Duration (hours)'] = np.clip(df_large['Session_Duration (hours)'], 0.3, 2.5)

print(f"Final realistic shape: {df_large.shape}")

df.to_csv('output/expanded_dataset.csv', index=False)