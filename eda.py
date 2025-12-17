import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
df = pd.read_csv('original_dataset/gym_members_exercise_tracking.csv')

print("=" * 50)
# 1. describe data
# show missing value in data
print(f"Missing Value = {df.isnull().sum()}")
print("=" * 50)
# show duplicated value
print(f'Duplicated = {df.duplicated().sum()}')
print("=" * 50)
# the shape of data
print(f'Shape = {df.shape}')
print("=" * 50)
# describe data
print("Description of data:")
df.describe().T
print("=" * 50)
# data info to know numerical and categorical values
print("Information of data:")
df.info()
print("=" * 50)
# show NULL value
print("Percentage of null data:")
data_null = round(df.isna().sum() / df.shape[0] * 100, 2)
data_null.to_frame(name='percent NULL data (%)')
print(data_null)

print("=" * 50)
# 2. data visualization
# show Numerical columns before analysis
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("Numerical Columns:")
print(numerical_cols)
print("=" * 50)
# show Categorical columns before analysis
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print("Categorical Columns:")
print(categorical_cols)
print("=" * 50)

# A. Fat_Percentage vs Calories Burned
# the higher the percentage of body fat, the lower the number of calories burned by the body
sns.lineplot(data=df, x="Calories_Burned", color="skyblue", y="Fat_Percentage")
plt.title("Fat_Percentage vs Calories Burned")
plt.show()

# B. Workout_Frequency (days/week) vs Calories Burned
# exercising more than once per week leads to a consistent increase in overall energy expenditure.
sns.set_theme(style="whitegrid")
custom_palette = {
    "Male": "#1f77b4",  # Soft blue
    "Female": "#ff7f0e"  # Soft orange
}
plt.figure(figsize=(10, 6))
bar = sns.barplot(
    x="Workout_Frequency (days/week)",
    y="Calories_Burned",
    data=df,
    hue="Gender",
    palette=custom_palette,
    edgecolor="black",
    linewidth=1.5,
    saturation=0.85,
    alpha=0.9
)
plt.title("Average Calories Burned by Workout Frequency and Gender",
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Workout Frequency (days/week)", fontsize=13)
plt.ylabel("Calories Burned (kcal)", fontsize=13)
for container in bar.containers:
    bar.bar_label(container, fmt='%.0f', label_type='edge', fontsize=10, padding=3)
plt.legend(
    title="Gender",
    title_fontsize=12,
    fontsize=11,
    loc="upper left",
    frameon=True,
    fancybox=True,
    shadow=True,
    borderpad=1
)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

# C. Session Duration distribution by gender
sns.histplot(data=df, x="Session_Duration (hours)", kde=True, hue="Gender", multiple="stack")
plt.title("Stacked Histogram by Gender")
plt.show()
print("=" * 50)

# D. Experience Level Analysis
# Set professional style
sns.set_theme(style="whitegrid")
num_cols = ['Age', 'Weight (kg)', 'Session_Duration (hours)', 'Calories_Burned',
            'Fat_Percentage', 'BMI']

# Define color palette
palette = sns.color_palette("viridis", len(df["Experience_Level"].unique()))

# Create subplots
plt.figure(figsize=(18, 16))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(
        data=df,
        x="Experience_Level",
        y=col,
        palette=palette,
        width=0.6,
        fliersize=3,
        linewidth=1.2
    )
    plt.title(f"{col} vs Experience Level", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("")
    plt.ylabel(col, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

# Global title
plt.suptitle("Relationship Between Experience Level and Other Attributes", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
sns.despine(left=True, bottom=True)
plt.show()

# Workout_Type vs Experience Level
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    x="Experience_Level",
    hue="Workout_Type",
    palette="viridis",
    edgecolor="black"
)
plt.title("Workout Type Distribution by Experience Level", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Experience Level")
plt.ylabel("Count of Participants")
plt.legend(title="Workout Type", title_fontsize=11)
plt.tight_layout()
plt.show()
