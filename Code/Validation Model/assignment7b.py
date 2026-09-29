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
total_error = 0
total_data = 0

print("=" * 50)
print("K-NN 10-FOLD DENGAN K = 3")
print("=" * 50)

for nomor_fold in range(10):
    posisi_test = bagian[nomor_fold]
    posisi_train = dataset.index.difference(posisi_test)
    train_data = dataset.loc[posisi_train, kolom_fitur].copy()
    train_label = dataset.loc[posisi_train, "Survived"].copy()
    test_data = dataset.loc[posisi_test, kolom_fitur].copy()
    test_label = dataset.loc[posisi_test, "Survived"].copy()
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
    total_error = total_error + jumlah_error
    total_data = total_data + len(test_label)
    print("Fold", nomor_fold + 1, "-> error :", jumlah_error, ", error ratio :", round(error_ratio, 4))

error_ratio_kfold = total_error / total_data
print("Total data diuji    :", total_data)
print("Total error         :", total_error)
print("Error ratio 10-Fold :", round(error_ratio_kfold, 4))
print("Error percentage    :", round(error_ratio_kfold * 100, 2), "%")
