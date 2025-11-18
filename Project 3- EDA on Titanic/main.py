#Project 3-EDA on Titanic

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sns.set(style="whitegrid")

df = sns.load_dataset("titanic")

print("✅ Dataset Loaded Successfully!")
print(df.head())


print("\n✅ Dataset Info:")
print(df.info())

print("\n✅ Summary Statistics:")
print(df.describe(include="all"))

print("\n✅ Missing Values:")
print(df.isnull().sum())


df['age'] = df['age'].fillna(df['age'].median())


df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])


df = df.drop(columns=['deck'])

print("\n✅ After Cleaning Missing Values:")
print(df.isnull().sum())


plt.figure(figsize=(6,4))
sns.histplot(df['age'], kde=True, color='skyblue')
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df)
plt.title("Count of Male vs Female")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='class', data=df)
plt.title("Passenger Class Count")
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x='class', y='survived', data=df)
plt.title("Survival Rate by Class")
plt.show()



plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='fare', hue='survived', data=df)
plt.title("Age vs Fare (colored by Survival)")
plt.show()


numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap (Numeric Features Only)")
plt.show(block=True)




print("\n✅ Key Insights:")
print("""
1️⃣ Females had a higher survival rate than males  
2️⃣ 1st class passengers survived more than 2nd & 3rd  
3️⃣ Younger passengers had slightly higher survival  
4️⃣ Fare has positive correlation with survival  
5️⃣ Many columns had missing data (deck, age, embarked)
""")


input("Press ENTER to exit...")

