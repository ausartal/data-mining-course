import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "Heart_dataset.csv"))

print("Data awal :")
print(dataset)
print()
print("Informasi kolom dan tipe data awal :")
dataset.info()

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

print()
print("Mapping kategori :")
for kolom, nilai_mapping in mapping.items():
    print(kolom, ":", nilai_mapping)

for kolom, nilai_mapping in mapping.items():
    dataset[kolom] = dataset[kolom].map(nilai_mapping)

hasil_csv = os.path.join(folder, "Heart_Dataset_encoded.csv")
dataset.to_csv(hasil_csv, index=False)

print()
print("Data setelah encoding :")
print(dataset)
print()
print("Tipe data setelah encoding :")
print(dataset.dtypes)
print()
print("Jumlah baris :", dataset.shape[0])
print("Jumlah kolom :", dataset.shape[1])
print("File hasil   :", hasil_csv)
