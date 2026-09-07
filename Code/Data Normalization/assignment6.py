import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Age", "Fare"]].copy()
kelas = dataset["Survived"]

# isi dulu missing value Age dengan mean per class
for val in kelas.unique():
    mean_age = dataset[kelas == val]["Age"].mean()
    idx = (kelas == val) & (data["Age"].isnull())
    data.loc[idx, "Age"] = mean_age

print("=" * 50)
print("NORMALISASI MIN-MAX (0 - 1)")
print("=" * 50)
print()
print("Rumus : (x - min) / (max - min)")
print()

# normalisasi Min-Max untuk kolom Age
min_age = data["Age"].min()
max_age = data["Age"].max()

print("Age  -> min :", min_age, ", max :", max_age)

data["Age_MinMax"] = (data["Age"] - min_age) / (max_age - min_age)

# normalisasi Min-Max untuk kolom Fare
min_fare = data["Fare"].min()
max_fare = data["Fare"].max()

print("Fare -> min :", min_fare, ", max :", max_fare)
print()

data["Fare_MinMax"] = (data["Fare"] - min_fare) / (max_fare - min_fare)

print("Hasil Normalisasi Min-Max :")
print(data[["Age", "Fare", "Age_MinMax", "Fare_MinMax"]])
