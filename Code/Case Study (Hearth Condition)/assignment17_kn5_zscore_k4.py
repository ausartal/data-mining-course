import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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

validasi = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

prediksi_benar = 0
prediksi_salah = 0

for indeks_latih, indeks_uji in validasi.split(fitur, target):
    data_latih = fitur.iloc[indeks_latih].copy()
    data_uji = fitur.iloc[indeks_uji].copy()
    target_latih = target.iloc[indeks_latih]
    target_uji = target.iloc[indeks_uji]

    scaler = StandardScaler()
    data_latih = scaler.fit_transform(data_latih)
    data_uji = scaler.transform(data_uji)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(data_latih, target_latih)
    hasil_prediksi = model.predict(data_uji)

    prediksi_benar += np.sum(hasil_prediksi == target_uji.to_numpy())
    prediksi_salah += np.sum(hasil_prediksi != target_uji.to_numpy())

jumlah_data_uji = prediksi_benar + prediksi_salah
accuracy = prediksi_benar / jumlah_data_uji
error_ratio = prediksi_salah / jumlah_data_uji

print("=" * 50)
print("K-NN K=5 | Z-SCORE | K-FOLD K=4")
print("=" * 50)
print("Metode klasifikasi    : K-NN K=5")
print("Nilai k                :", 5)
print("Metode normalisasi    : Z-SCORE")
print("Metode validasi       : K-FOLD K=4")
print("Jumlah data uji        :", jumlah_data_uji)
print("Jumlah prediksi benar  :", prediksi_benar)
print("Jumlah prediksi salah  :", prediksi_salah)
print("Accuracy               :", round(accuracy, 4))
print("Accuracy percentage    :", round(accuracy * 100, 2), "%")
print("Error ratio            :", round(error_ratio, 4))
print("Error percentage       :", round(error_ratio * 100, 2), "%")
