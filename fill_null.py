import pandas as pd
import numpy as np
import random
# load data
df = pd.read_csv('output/after_concatenation.csv')
print("="*80)
print("Starting filling null value in 'Equipment Needed'")
print("-"*80)
print("Before filling null value in 'Equipment Needed'")
print("-"*80)
print(df['Equipment Needed'].isnull())
print("-"*80)
equipment_list = [
    "Dumbbells", "Barbell", "Resistance Bands", "Kettlebell", "Pull-up Bar",
    "Medicine Ball", "Bench", "Cable Machine", "Treadmill", "No Equipment"
]
df['Equipment Needed'] = df['Equipment Needed'].fillna(np.random.choice(equipment_list))
print("After filling null value in 'Equipment Needed':")
print(f"Number of nan in 'Equipment Needed': {df['Equipment Needed'].isnull().sum()}")
print("="*80)

# Define realistic value lists
print("Starting filling 'Benefit', 'Target Muscle Group', 'Difficulty Level', 'Body Part', 'Type of Muscle' and 'Workout'")
print("-"*80)
benefit_list = [
    "Builds upper body strength", "Improves flexibility", "Enhances endurance",
    "Burns fat quickly", "Increases muscle mass", "Improves posture",
    "Boosts cardiovascular fitness", "Improves balance and coordination"
]

target_muscle_list = [
    "Chest", "Back", "Arms", "Legs", "Core", "Shoulders",
    "Glutes", "Full Body", "Quadriceps", "Triceps"
]

difficulty_list = ["Beginner", "Intermediate", "Advanced"]

body_part_list = [
    "Chest", "Back", "Legs", "Arms", "Shoulders", "Core", "Full Body"
]

type_of_muscle_list = [
    "Upper Chest", "Lower Chest", "Biceps", "Triceps",
    "Quads", "Hamstrings", "Abs", "Deltoids", "Lats"
]

workout_list = [
    "Bench Press", "Push Ups", "Pull Ups", "Squats", "Deadlift",
    "Lunges", "Plank", "Bicep Curls", "Overhead Press", "Lat Pulldown"
]
df['Benefit'] = df['Benefit'].fillna(np.random.choice(benefit_list))
df['Target Muscle Group'] = df['Target Muscle Group'].fillna(np.random.choice(target_muscle_list))
df['Difficulty Level'] = df['Difficulty Level'].fillna(np.random.choice(difficulty_list))
df['Body Part'] = df['Body Part'].fillna(np.random.choice(body_part_list))
df['Type of Muscle'] = df['Type of Muscle'].fillna(np.random.choice(type_of_muscle_list))
df['Workout'] = df['Workout'].fillna(np.random.choice(workout_list))

df.loc[df['Body Part'] == 'Chest', 'Type of Muscle'] = np.random.choice(["Upper Chest", "Lower Chest"])
df.loc[df['Body Part'] == 'Legs', 'Type of Muscle'] = np.random.choice(["Quads", "Hamstrings"])
df.loc[df['Body Part'] == 'Arms', 'Type of Muscle'] = np.random.choice(["Biceps", "Triceps"])
df.loc[df['Body Part'] == 'Back', 'Type of Muscle'] = np.random.choice(["Lats", "Deltoids"])
df.loc[df['Body Part'] == 'Core', 'Type of Muscle'] = "Abs"

# Verify result

print("Missing values after filling:")
print(df.isnull().sum().sort_values(ascending=False).head(10))
print("="*80)

print("Starting filling remains")
exercise_names = [
    "Push Ups", "Squats", "Lunges", "Plank", "Deadlift",
    "Bicep Curls", "Bench Press", "Burpees", "Mountain Climbers", "Leg Press"
]

target_muscles = [
    "Chest", "Legs", "Back", "Core", "Arms",
    "Shoulders", "Glutes", "Full Body"
]

equipments = [
    "Dumbbells", "Barbell", "Resistance Band", "Bodyweight",
    "Kettlebell", "Machine", "Bench", "Cable"
]

difficulty_levels = ["Beginner", "Intermediate", "Advanced"]

# Name of Exercise
df['Name of Exercise'] = df['Name of Exercise'].fillna(
    pd.Series(np.random.choice(exercise_names, size=len(df)), index=df.index)
)

# Reps and Sets
def generate_reps(difficulty):
    if difficulty == 'Beginner':
        return random.randint(10, 15)
    elif difficulty == 'Intermediate':
        return random.randint(12, 20)
    else:
        return random.randint(15, 25)

def generate_sets(difficulty):
    if difficulty == 'Beginner':
        return random.randint(2, 3)
    elif difficulty == 'Intermediate':
        return random.randint(3, 4)
    else:
        return random.randint(4, 5)

df['Reps'] = df['Reps'].fillna(df['Difficulty Level'].apply(generate_reps))
df['Sets'] = df['Sets'].fillna(df['Difficulty Level'].apply(generate_sets))

#  Burns Calories (per 30 min)
def generate_calories(row):
    base = {'Beginner': 150, 'Intermediate': 250, 'Advanced': 350}
    wt_factor = 0.2 * (row['Weight (kg)'] - 70) if 'Weight (kg)' in df.columns else 0
    return base.get(row['Difficulty Level'], 200) + wt_factor + random.randint(-20, 20)

df['Burns Calories (per 30 min)'] = df['Burns Calories (per 30 min)'].fillna(df.apply(generate_calories, axis=1))

#  Target Muscle Group
df['Target Muscle Group'] = df['Target Muscle Group'].fillna(
    pd.Series(np.random.choice(target_muscles, size=len(df)), index=df.index)
)

#  Equipment Needed
df['Equipment Needed'] = df['Equipment Needed'].fillna(
    pd.Series(np.random.choice(equipments, size=len(df)), index=df.index)
)

#  Difficulty Level
df['Difficulty Level'] = df['Difficulty Level'].fillna(
    pd.Series(np.random.choice(difficulty_levels, size=len(df)), index=df.index)
)
print('-'*80)
print(f"number of missing values in df: \n{df.isnull().sum()}")
print("="*80)

df.to_csv('output/after_filling.csv', index=False)
