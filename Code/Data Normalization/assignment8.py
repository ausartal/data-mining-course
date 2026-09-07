import pandas as pd
import numpy as np

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Age", "Fare"]].copy()
kelas = dataset["Survived"]

# isi dulu missing value Age dengan mean per class
for val in kelas.unique():
    mean_age = dataset[kelas == val]["Age"].mean()
    idx = (kelas == val) & (data["Age"].isnull())
    data.loc[idx, "Age"] = mean_age

print("=" * 50)
print("NORMALISASI SIGMOIDAL")
print("=" * 50)
print()
print("Rumus : 1 / (1 + e^(-x))")
print()

# normalisasi Sigmoidal untuk kolom Age
data["Age_Sigmoid"] = 1 / (1 + np.exp(-data["Age"]))

# normalisasi Sigmoidal untuk kolom Fare
data["Fare_Sigmoid"] = 1 / (1 + np.exp(-data["Fare"]))

print("Hasil Normalisasi Sigmoidal :")
print(data[["Age", "Fare", "Age_Sigmoid", "Fare_Sigmoid"]])
