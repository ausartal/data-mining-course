import os
import numpy as np
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
mean_age_setiap_kelas = dataset.groupby("Survived")["Age"].mean()

for kelas in mean_age_setiap_kelas.index:
    posisi_kosong = (dataset["Survived"] == kelas) & (dataset["Age"].isnull())
    dataset.loc[posisi_kosong, "Age"] = mean_age_setiap_kelas[kelas]

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
dataset = dataset.sample(frac=1, random_state=10).reset_index(drop=True)

bagian = np.array_split(dataset.index, 10)
posisi_test = bagian[0]
posisi_train = dataset.index.difference(posisi_test)
train_data = dataset.loc[posisi_train, kolom_fitur].copy()
train_label = dataset.loc[posisi_train, "Survived"].copy()
test_data = dataset.loc[posisi_test, kolom_fitur].copy()
test_label = dataset.loc[posisi_test, "Survived"].copy()

nilai_min = train_data.min()
nilai_max = train_data.max()
train_data = (train_data - nilai_min) / (nilai_max - nilai_min)

print("=" * 50)
print("NORMALISASI DATA LATIH FOLD PERTAMA")
print("=" * 50)
print("Nilai minimum dan maksimum setiap atribut :")
print(pd.DataFrame({"Minimum": nilai_min, "Maksimum": nilai_max}))
print()
print(train_data)
