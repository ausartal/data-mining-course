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

posisi_test = 0
train_data = dataset.drop(index=posisi_test)[kolom_fitur].copy()
train_label = dataset.drop(index=posisi_test)["Survived"].copy()
test_data = dataset.loc[[posisi_test], kolom_fitur].copy()
test_label = dataset.loc[posisi_test, "Survived"]

print("=" * 50)
print("LABEL DATA LATIH LOO PERTAMA")
print("=" * 50)
print(train_label)
