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

batas = int(len(dataset) * 0.7)
train_data = dataset.loc[:batas - 1, kolom_fitur].copy()
train_label = dataset.loc[:batas - 1, "Survived"].copy()
test_data = dataset.loc[batas:, kolom_fitur].copy()
test_label = dataset.loc[batas:, "Survived"].copy()

nilai_min = train_data.min()
nilai_max = train_data.max()
train_data = (train_data - nilai_min) / (nilai_max - nilai_min)
test_data = (test_data - nilai_min) / (nilai_max - nilai_min)

hasil_prediksi = []

for i in test_data.index:
    jarak = np.sqrt(((train_data - test_data.loc[i]) ** 2).sum(axis=1))
    posisi_terdekat = np.argsort(jarak.to_numpy())[:3]
    label_terdekat = train_label.iloc[posisi_terdekat]
    prediksi = np.bincount(label_terdekat.to_numpy()).argmax()
    hasil_prediksi.append(prediksi)

jumlah_error = (np.array(hasil_prediksi) != test_label.to_numpy()).sum()
error_ratio = jumlah_error / len(test_label)

print("=" * 50)
print("K-NN HOLD-OUT DENGAN K = 3")
print("=" * 50)
print("Jumlah data latih :", len(train_data))
print("Jumlah data uji   :", len(test_data))
print("Jumlah error      :", jumlah_error)
print("Error ratio       :", round(error_ratio, 4))
print("Error percentage  :", round(error_ratio * 100, 2), "%")
