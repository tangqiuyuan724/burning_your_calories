import pandas as pd

# load main dataset
df = pd.read_csv('original_dataset/gym_members_exercise_tracking.csv')

# 1. merge user_nutritional_data.csv
df2 = pd.read_csv('original_dataset/user_nutritional_data.csv')
print("=" * 80)
print("Starting concat user_nutritional_data.csv")
print("-" * 80)
print(f"Before dropping, columns of user_nutritional_data.csv:\n{df2.columns}")

print("-" * 80)
# drop unneeded columns
df2.drop(columns=['Gender', 'Age', 'Height', 'Weight', 'BMR','Daily meals frequency'
    ,'Carbs', 'Proteins', 'Fats'], inplace=True)
print(f"After dropping, columns of user_nutritional_data.csv:\n{df2.columns}")

# concatenation
df = pd.concat([df, df2], axis=1)
print("-" * 80)
print(f"After concatenation, columns of main dataset:\n{df.columns}")
print("-" * 80)
print(f"Shape of dataset: {df.shape}")

# Data Handling after merging
print("-" * 80)
# show NULL value
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"percentage of null data before dropping :\n{data_null}")
print("-" * 80)
#  Drop heavily null rows
df = df.dropna(thresh=df.shape[1] * 0.5)
# show NULL value
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"percentage of null data after dropping :\n{data_null}")
print("-" * 80)
# show the information about data
df.info()
print("-" * 80)
# show missing value in data
print(f"Missing Value = {df.isnull().sum()}")
print("-" * 80)
# show duplicated value
print(f'Duplicated = {df.duplicated().sum()}')
print("-" * 80)
# the shape of data
print(f'Shape = {df.shape}')
print("-" * 80)
# describe data
df.describe().T
print("=" * 80)

df.to_csv('output/concat.csv', index=False)
