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
df2.drop(columns=['Gender', 'Age', 'Height', 'Weight', 'BMR'], inplace=True)
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

# 2. merge healthy_eating_dataset.csv
print("Starting concat healthy_eating_dataset.csv")
print("-" * 80)
df3 = pd.read_csv('original_dataset/healthy_eating_dataset.csv')
print(f"Columns of healthy_eating_dataset.csv:\n{df3.columns}")
print("-" * 80)
# show NULL value
data_null = round(df3.isna().sum() / df3.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"percentage of null data in healthy_eating_dataset.csv:\n{data_null}")
print("-" * 80)
df3.drop(columns=['meal_id', 'cuisine', 'calories', 'protein_g', 'carbs_g', 'fat_g', 'fiber_g', 'image_url'],
         inplace=True)
print(f"After dropping unneeded features, columns of healthy_eating_dataset.csv:\n{df3.columns}")
print("-" * 80)
# concatination
df = pd.concat([df, df3], axis=1)
print(f"After concatenation, columns of main dataset:\n{df.columns}")
print("-" * 80)
print(f"Shape of dataset: {df.shape}")
print("-" * 80)
# show NULL value
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"percentage of null data before dropping :\n{data_null}")
print("-" * 80)
#  Drop heavily null rows
df = df.dropna(thresh=df.shape[1] * 0.5)
# check again after remove Null Value
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"percentage of null data after dropping :\n{data_null}")
print("-" * 80)
print(f"Shape of dataset: {df.shape}")
print("-" * 80)
print(df.info())
print("=" * 80)

# 3. merge Top 50 Excerice for your body.csv and Workout.csv
print("Starting concat Top 50 Excerice for your body.csv and Workout.csv")
df4 = pd.read_csv('original_dataset/Top 50 Excerice for your body.csv')
df5 = pd.read_csv('original_dataset/Workout.csv')
print("-" * 80)
print(f"Shape of Top 50 Excerice for your body.csv: {df4.shape}")
print("-" * 80)
print(f"Columns of Top 50 Excerice for your body.csv:\n{df4.columns}")
print("-" * 80)
print(f"Information of Top 50 Excerice for your body.csv:\n{df4.info()}")
print("-" * 80)
# concatenation
df = pd.concat([df, df4], axis=1)
print(f"Shape of main dataset after concatenation of Top 50 Excerice for your body.csv: \n{df.shape}")
print("-" * 80)
print(f"Shape of Workout.csv: {df5.shape}")
print("-" * 80)
print(f"Columns of Workout.csv before dropping:\n{df5.columns}")
print("-" * 80)
# Drop Unneeded Columns
df5.drop(columns=['Sets', 'Reps per Set'], inplace=True)
print(f"Columns of Workout.csv after dropping:\n{df5.columns}")
print("-" * 80)

# concatenation Data 5 with Needed Data
df = pd.concat([df, df5], axis=1)
print(f"Shape of main dataset after concatenation of Workout.csv: \n{df.shape}")
print("-" * 80)
# show missing value in data
print(f"Missing Value = {df.isnull().sum()}")
print("-" * 80)
# show duplicated value
print(f'Duplicated = {df.duplicated().sum()}')
print("-" * 80)
# check again after remove Null Value
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(f"Percentage of null data after concatenation:\n{data_null}")

df.to_csv('output/after_concatenation.csv', index=False)
print("=" * 80)
