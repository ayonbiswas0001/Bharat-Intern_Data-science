import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic data from CSV
data = pd.read_csv('titanic.csv')  # Replace 'titanic.csv' with your actual file name

# 1. Survival Comparison
survival_counts = data['Survived'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(survival_counts.index, survival_counts.values, tick_label=['Did Not Survive', 'Survived'])
plt.title('Survival Comparison')
plt.xlabel('Survival')
plt.ylabel('Count')
plt.show()

# 2. Survival by Pclass
survival_by_pclass = data.groupby('Pclass')['Survived'].mean()
plt.figure(figsize=(6, 4))
survival_by_pclass.plot(kind='bar')
plt.title('Survival by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

# 3. Sex Ratio Comparison
sex_ratio = data['Sex'].value_counts()
survived_sex_ratio = data.groupby('Sex')['Survived'].mean()
plt.figure(figsize=(8, 4))
sns.barplot(x=sex_ratio.index, y=sex_ratio.values, color='blue', label='Total')
sns.barplot(x=survived_sex_ratio.index, y=survived_sex_ratio.values, color='orange', label='Survived')
plt.title('Sex Ratio Comparison')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend()
plt.show()

# 4. Survival Rate by Age
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='Age', hue='Survived', bins=20, multiple='stack', palette='pastel')
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['Did Not Survive', 'Survived'])
plt.show()

# 5. Survival Rate of Siblings/Spouses
siblings_survival = data.groupby('Siblings/Spouses Aboard')['Survived'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=siblings_survival.index, y=siblings_survival.values)
plt.title('Survival Rate of Siblings/Spouses')
plt.xlabel('Number of Siblings/Spouses')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

# 6. Survival Rate of Parents/Children
parents_survival = data.groupby('Parents/Children Aboard')['Survived'].mean()
plt.figure(figsize=(10, 6))
sns.barplot(x=parents_survival.index, y=parents_survival.values)
plt.title('Survival Rate of Parents/Children')
plt.xlabel('Number of Parents/Children')
plt.ylabel('Survival Rate')
plt.xticks(rotation=0)
plt.show()

# 7. Comparison by "Can Swim"
can_swim_survival = data.groupby('Can Swim')['Survived'].value_counts().unstack().fillna(0)
plt.figure(figsize=(8, 5))
can_swim_survival.plot(kind='bar', stacked=True)
plt.title('Survival Comparison by "Can Swim"')
plt.xlabel('Can Swim')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['Did Not Survive', 'Survived'])
plt.show()
