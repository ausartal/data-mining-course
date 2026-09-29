import os
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

batas = int(len(dataset) * 0.7)
train_data = dataset.loc[:batas - 1, kolom_fitur].copy()
train_label = dataset.loc[:batas - 1, "Survived"].copy()
test_data = dataset.loc[batas:, kolom_fitur].copy()
test_label = dataset.loc[batas:, "Survived"].copy()

nilai_min = train_data.min()
nilai_max = train_data.max()
train_data = (train_data - nilai_min) / (nilai_max - nilai_min)

print("=" * 50)
print("NORMALISASI DATA LATIH HOLD-OUT")
print("=" * 50)
print("Nilai minimum dan maksimum setiap atribut :")
print(pd.DataFrame({"Minimum": nilai_min, "Maksimum": nilai_max}))
print()
print(train_data)
