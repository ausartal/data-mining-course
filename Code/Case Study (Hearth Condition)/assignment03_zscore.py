import os
import pandas as pd
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

scaler = StandardScaler()
fitur_normal = pd.DataFrame(
    scaler.fit_transform(fitur),
    columns=fitur.columns,
    index=fitur.index
)

print("=" * 50)
print("NORMALISASI Z-SCORE")
print("=" * 50)
print()
print("Rumus : (x - mean) / std")
print()
print("Mean dan standar deviasi :")
print(pd.DataFrame({"mean": fitur.mean(), "standar_deviasi": fitur.std()}))
print()
print("Data sebelum normalisasi :")
print(fitur.head())
print()
print("Hasil normalisasi Z-Score :")
print(fitur_normal.head())
print()
print("Target tidak dinormalisasi :")
print(target.head())
