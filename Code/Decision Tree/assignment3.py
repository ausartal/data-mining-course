import pandas as pd

dataset = pd.read_csv("titanic.csv")

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
train_data = dataset[kolom_fitur].copy()

mean_age_setiap_class = dataset.groupby("Survived")["Age"].mean()

for kelas in mean_age_setiap_class.index:
    posisi_kosong = (dataset["Survived"] == kelas) & (train_data["Age"].isnull())
    train_data.loc[posisi_kosong, "Age"] = mean_age_setiap_class[kelas]

train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})

print("=" * 50)
print("TRAIN DATA")
print("=" * 50)
print()
print("Mean Age setiap class :")
print(mean_age_setiap_class)
print()
print("Missing value setelah diisi :")
print(train_data.isnull().sum())
print()
print(train_data)
