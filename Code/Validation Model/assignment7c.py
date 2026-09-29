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

total_error = 0
catatan_min_max = []

for i in dataset.index:
    train_data = dataset.drop(index=i)[kolom_fitur].copy()
    train_label = dataset.drop(index=i)["Survived"].copy()
    test_data = dataset.loc[[i], kolom_fitur].copy()
    test_label = dataset.loc[i, "Survived"]
    nilai_min = train_data.min()
    nilai_max = train_data.max()
    train_data = (train_data - nilai_min) / (nilai_max - nilai_min)
    test_data = (test_data - nilai_min) / (nilai_max - nilai_min)
    jarak = np.sqrt(((train_data - test_data.loc[i]) ** 2).sum(axis=1))
    posisi_terdekat = np.argsort(jarak.to_numpy())[:3]
    label_terdekat = train_label.iloc[posisi_terdekat]
    prediksi = np.bincount(label_terdekat.to_numpy()).argmax()

    if prediksi != test_label:
        total_error = total_error + 1

    catatan_min_max.append([i + 1, nilai_min["Sex"], nilai_max["Sex"], nilai_min["Age"], nilai_max["Age"], nilai_min["Pclass"], nilai_max["Pclass"], nilai_min["Fare"], nilai_max["Fare"]])

nilai_normalisasi = pd.DataFrame(catatan_min_max, columns=["Data", "Sex Minimum", "Sex Maksimum", "Age Minimum", "Age Maksimum", "Pclass Minimum", "Pclass Maksimum", "Fare Minimum", "Fare Maksimum"])
error_ratio = total_error / len(dataset)

print("=" * 50)
print("K-NN LOO DENGAN K = 3")
print("=" * 50)
print("Nilai minimum dan maksimum data latih setiap LOO :")
print(nilai_normalisasi.to_string(index=False))
print("Jumlah percobaan :", len(dataset))
print("Total error      :", total_error)
print("Error ratio      :", round(error_ratio, 4))
print("Error percentage :", round(error_ratio * 100, 2), "%")
