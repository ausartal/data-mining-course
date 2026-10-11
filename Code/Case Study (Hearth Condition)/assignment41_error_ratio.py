import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "Heart_dataset.csv"))

mapping = {
    "sex": {"female": 0, "male": 1},
    "chest_pain_type": {
        "typical_angina": 0,
        "atypical_angina": 1,
        "non_anginal_pain": 2,
        "asymptomatic": 3
    },
    "resting_ecg_result": {
        "normal": 0,
        "wave_abnormality": 1,
        "ventricular_hypertrophy": 2
    },
    "exercise_induced_angina": {"no": 0, "yes": 1},
    "st_slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
    "thal": {"normal": 0, "fixed_defect": 1, "reversable_defect": 2},
    "target": {"Absence": 0, "Presence": 1}
}

for kolom, nilai_mapping in mapping.items():
    dataset[kolom] = dataset[kolom].map(nilai_mapping)

fitur = dataset.drop(columns=["target"])
target = dataset["target"]

daftar_model = [
    ("K-NN K=1", "knn", 1),
    ("K-NN K=5", "knn", 5),
    ("K-NN K=9", "knn", 9),
    ("Decision Tree", "tree", 0)
]
daftar_normalisasi = ["Min-Max", "Z-Score", "Sigmoidal"]
daftar_validasi = ["K-Fold K=4", "K-Fold K=10", "Leave-One-Out"]

ringkasan = []
nomor = 5

for nama_model, jenis_model, nilai_k in daftar_model:
    for normalisasi in daftar_normalisasi:
        for nama_validasi in daftar_validasi:
            if nama_validasi == "K-Fold K=4":
                validasi = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            elif nama_validasi == "K-Fold K=10":
                validasi = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            else:
                validasi = LeaveOneOut()

            prediksi_benar = 0
            prediksi_salah = 0

            for indeks_latih, indeks_uji in validasi.split(fitur, target):
                data_latih = fitur.iloc[indeks_latih].copy()
                data_uji = fitur.iloc[indeks_uji].copy()
                target_latih = target.iloc[indeks_latih]
                target_uji = target.iloc[indeks_uji]

                if normalisasi == "Min-Max":
                    scaler = MinMaxScaler()
                    data_latih = scaler.fit_transform(data_latih)
                    data_uji = scaler.transform(data_uji)
                elif normalisasi == "Z-Score":
                    scaler = StandardScaler()
                    data_latih = scaler.fit_transform(data_latih)
                    data_uji = scaler.transform(data_uji)
                else:
                    data_latih = 1 / (1 + np.exp(-data_latih))
                    data_uji = 1 / (1 + np.exp(-data_uji))

                if jenis_model == "knn":
                    model = KNeighborsClassifier(n_neighbors=nilai_k)
                else:
                    model = DecisionTreeClassifier(random_state=42)

                model.fit(data_latih, target_latih)
                hasil_prediksi = model.predict(data_uji)
                prediksi_benar += np.sum(hasil_prediksi == target_uji.to_numpy())
                prediksi_salah += np.sum(hasil_prediksi != target_uji.to_numpy())

            jumlah_data_uji = prediksi_benar + prediksi_salah
            accuracy = prediksi_benar / jumlah_data_uji
            error_ratio = prediksi_salah / jumlah_data_uji

            ringkasan.append({
                "nomor": nomor,
                "klasifikasi": nama_model,
                "normalisasi": normalisasi,
                "validasi": nama_validasi,
                "prediksi_benar": prediksi_benar,
                "prediksi_salah": prediksi_salah,
                "jumlah_data_uji": jumlah_data_uji,
                "accuracy": round(accuracy, 4),
                "accuracy_percentage": round(accuracy * 100, 2),
                "error_ratio": round(error_ratio, 4),
                "error_percentage": round(error_ratio * 100, 2)
            })
            nomor += 1

tabel = pd.DataFrame(ringkasan)
tabel = tabel.sort_values(
    by=["error_ratio", "prediksi_salah", "nomor"]
).reset_index(drop=True)

print("=" * 50)
print("RINGKASAN ERROR RATIO")
print("=" * 50)
print(tabel.to_string(index=False))

print("=" * 50)
print("TIGA PERCOBAAN DENGAN ERROR RATIO TERKECIL")
print("=" * 50)
print(tabel.head(3).to_string(index=False))

terbaik = tabel.iloc[0]

print("=" * 50)
print("KESIMPULAN")
print("=" * 50)
print("Metode klasifikasi    :", terbaik["klasifikasi"])
print("Metode normalisasi    :", terbaik["normalisasi"])
print("Metode validasi       :", terbaik["validasi"])
print("Jumlah data uji       :", terbaik["jumlah_data_uji"])
print("Jumlah prediksi benar :", terbaik["prediksi_benar"])
print("Jumlah prediksi salah :", terbaik["prediksi_salah"])
print("Accuracy              :", terbaik["accuracy"])
print("Accuracy percentage   :", terbaik["accuracy_percentage"], "%")
print("Error ratio           :", terbaik["error_ratio"])
print("Error percentage      :", terbaik["error_percentage"], "%")
