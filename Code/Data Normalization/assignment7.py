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
print("NORMALISASI Z-SCORE")
print("=" * 50)
print()
print("Rumus : (x - mean) / std")
print()

# normalisasi Z-Score untuk kolom Age
mean_age = data["Age"].mean()
std_age = data["Age"].std()

print("Age  -> mean :", round(mean_age, 2), ", std :", round(std_age, 2))

data["Age_ZScore"] = (data["Age"] - mean_age) / std_age

# normalisasi Z-Score untuk kolom Fare
mean_fare = data["Fare"].mean()
std_fare = data["Fare"].std()

print("Fare -> mean :", round(mean_fare, 2), ", std :", round(std_fare, 2))
print()

data["Fare_ZScore"] = (data["Fare"] - mean_fare) / std_fare

print("Hasil Normalisasi Z-Score :")
print(data[["Age", "Fare", "Age_ZScore", "Fare_ZScore"]])
