import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Age", "Fare"]].copy()
kelas = dataset["Survived"]

# cek missing value sebelum diisi
print("Missing value sebelum diisi :")
print(data.isnull().sum())
print()

# isi missing value pada kolom Age dengan mean per class
# artinya, rata-rata Age dihitung berdasarkan Survived (0 atau 1)
for val in kelas.unique():
    mean_age = dataset[kelas == val]["Age"].mean()
    print("Mean Age untuk Survived =", val, ":", round(mean_age, 2))

# isi missing value
for val in kelas.unique():
    mean_age = dataset[kelas == val]["Age"].mean()
    # cari baris yang Survived == val dan Age-nya kosong
    idx = (kelas == val) & (data["Age"].isnull())
    data.loc[idx, "Age"] = mean_age

print()
print("Missing value setelah diisi :")
print(data.isnull().sum())
print()
print(data)
