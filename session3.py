# ------------------ DAY3 - Titanic EDA ------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


titanic = sns.load_dataset("titanic")

# ------------------ DISPLAYING DATA ---------------------
print("Head of dataset:")
print(titanic.head(), "\n")

print("Shape of dataset:", titanic.shape, "\n")

print("Info:")
print(titanic.info(), "\n")

print("Summary OF DATA:")
print(titanic.describe(), "\n")


# ------------------ DISPLAYING DATA GRAPHICALLY ---------------------
plt.figure(figsize=(6,4))
sns.countplot(x="sex", data=titanic)
plt.title("Passengers Gender")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="class", data=titanic)
plt.title("Passenger Class")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="embark_town", data=titanic)
plt.title("Passengers Town")
plt.show()

# ------------------ ANALYSIS ---------------------
plt.figure(figsize=(6,4))
sns.countplot(x="survived", data=titanic)
plt.title("Overall Survival Distribution (0 = Died, 1 = Survived)")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="sex", hue="survived", data=titanic)
plt.title("Survival by Sex")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="class", hue="survived", data=titanic)
plt.title("Survival by Class")
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(x="age", hue="survived", data=titanic)
plt.title("Survival by Age Distribution")
plt.show()

# ------------------ BOXPLOT---------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="sex", y="age", hue="survived", data=titanic)
plt.title("Age Distribution by Sex and Survival")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x="class", y="fare", data=titanic)
plt.title("Fare Distribution by Class")
plt.show()

# ------------------ CORRELATION HEATMAP ---------------------
plt.figure(figsize=(8,6))
corr = titanic.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()


print("Titanic EDA Completed")
