import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 1. Load dataset dan tampilkan

print("=" * 50)
print("ASSIGNMENT 1 : Load Dataset Titanic")
print("=" * 50)

dataset = pd.read_csv("Titanic-Dataset.csv")
print(dataset)
print()


# 2. Jumlah baris dan kolom

print("=" * 50)
print("ASSIGNMENT 2 : Jumlah Baris dan Kolom")
print("=" * 50)

rows = dataset.shape[0]
cols = dataset.shape[1]

print("Jumlah baris :", rows)
print("Jumlah kolom :", cols)
print()


# 3. Ambil kolom fitur

print("=" * 50)
print("ASSIGNMENT 3 : Data Fitur")
print("=" * 50)

data = dataset[["Name", "Sex", "Age", "Pclass", "Fare"]]
print(data)
print()


# 4. Ambil kolom kelas

print("=" * 50)
print("ASSIGNMENT 4 : Data Kelas (Survived)")
print("=" * 50)

kelas = dataset["Survived"]
print(kelas)
print()


# 5. Tambah kolom Relatives

print("=" * 50)
print("ASSIGNMENT 5 : Tambah Fitur Relatives")
print("=" * 50)

data["Relatives"] = dataset["SibSp"] + dataset["Parch"]
print(data)
print()


# 6. Hitung penumpang per Pclass

print("=" * 50)
print("ASSIGNMENT 6 : Jumlah Penumpang per Pclass")
print("=" * 50)

pclass_count = dataset["Pclass"].value_counts()
pclass_count = pclass_count.sort_index()

for i in pclass_count.index:
    print("Pclass", i, ":", pclass_count[i], "penumpang")
print()


# 7. Hitung penumpang per Sex

print("=" * 50)
print("ASSIGNMENT 7 : Jumlah Penumpang per Sex")
print("=" * 50)

sex_count = dataset["Sex"].value_counts()

for i in sex_count.index:
    print(i, ":", sex_count[i], "penumpang")
print()


# 8. Selamat dan tidak selamat per Pclass

print("=" * 50)
print("ASSIGNMENT 8 : Selamat/Tidak Selamat per Pclass")
print("=" * 50)

for pclass_val in sorted(dataset["Pclass"].unique()):
    data_kelas = dataset[dataset["Pclass"] == pclass_val]
    tidak_selamat = len(data_kelas[data_kelas["Survived"] == 0])
    selamat = len(data_kelas[data_kelas["Survived"] == 1])
    print("Pclass", pclass_val, "-> Tidak Selamat:", tidak_selamat, ", Selamat:", selamat)
print()


# 9. Visualisasi Sex vs urutan data

print("=" * 50)
print("ASSIGNMENT 9 : Visualisasi Sex")
print("=" * 50)

fig, ax = plt.subplots(figsize=(10, 5))

# ubah sex jadi angka biar bisa diplot
# male = 0, female = 1
sex_angka = dataset["Sex"].map({"male": 0, "female": 1})

# bagi berdasarkan survived
for val in [0, 1]:
    idx = dataset["Survived"] == val
    if val == 0:
        warna = "red"
        label = "Tidak Selamat"
    else:
        warna = "blue"
        label = "Selamat"

    # tambah sedikit noise biar titik tidak tumpang tindih
    noise = np.random.uniform(-0.1, 0.1, size=idx.sum())
    ax.scatter(dataset.index[idx], sex_angka[idx] + noise,
               c=warna, label=label, alpha=0.5, s=15)

ax.set_yticks([0, 1])
ax.set_yticklabels(["male", "female"])
ax.set_xlabel("Urutan Data")
ax.set_ylabel("Sex")
ax.set_title("Visualisasi Sex berdasarkan Survived")
ax.legend()
plt.tight_layout()
plt.savefig("assignment_9_sex.png", dpi=150)
plt.close()

print("Plot berhasil disimpan : assignment_9_sex.png")
print()


# 10. Visualisasi Age vs urutan data

print("=" * 50)
print("ASSIGNMENT 10 : Visualisasi Age")
print("=" * 50)

# hapus baris yang Age-nya kosong
dataset_bersih = dataset.dropna(subset=["Age"])

fig, ax = plt.subplots(figsize=(10, 5))

for val in [0, 1]:
    idx = dataset_bersih["Survived"] == val
    if val == 0:
        warna = "red"
        label = "Tidak Selamat"
    else:
        warna = "blue"
        label = "Selamat"

    ax.scatter(dataset_bersih.index[idx], dataset_bersih["Age"][idx],
               c=warna, label=label, alpha=0.5, s=15)

ax.set_xlabel("Urutan Data")
ax.set_ylabel("Age")
ax.set_title("Visualisasi Age berdasarkan Survived")
ax.legend()
plt.tight_layout()
plt.savefig("assignment_10_age.png", dpi=150)
plt.close()

print("Plot berhasil disimpan : assignment_10_age.png")
print()

print("=" * 50)
print("SELESAI")
print("=" * 50)
