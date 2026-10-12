import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

data = dataset[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

posisi_train = data[data["Age"].notnull()].index
posisi_kosong = data[data["Age"].isnull()].index

train_data = data.loc[posisi_train, ["Sex", "Pclass", "Fare", "Survived"]].copy()
train_label = data.loc[posisi_train, "Age"].copy()
test_data = data.loc[posisi_kosong, ["Sex", "Pclass", "Fare", "Survived"]].copy()

kolom_fitur = ["Sex", "Pclass", "Fare", "Survived"]
scaler = MinMaxScaler()
train_data_normal = pd.DataFrame(
    scaler.fit_transform(train_data),
    columns=kolom_fitur,
    index=train_data.index
)

print("=" * 50)
print("NORMALISASI MIN-MAX (0 - 1)")
print("=" * 50)
print()
print("Nilai minimum dan maksimum :")
print(pd.DataFrame({"minimum": scaler.data_min_, "maksimum": scaler.data_max_}, index=kolom_fitur))
print()
print("Train data setelah normalisasi :")
print(train_data_normal)
