import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))
test_dataset = pd.read_csv(os.path.join(folder, "titanic_test.csv"))
test_label_dataset = pd.read_csv(os.path.join(folder, "titanic_testlabel.csv"))

data = dataset[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

posisi_train = data[data["Age"].notnull()].index
posisi_kosong = data[data["Age"].isnull()].index

train_data = data.loc[posisi_train, ["Sex", "Pclass", "Fare", "Survived"]].copy()
train_label = data.loc[posisi_train, "Age"].copy()
test_data = data.loc[posisi_kosong, ["Sex", "Pclass", "Fare", "Survived"]].copy()

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

hasil_prediksi = []

for i in range(len(test_data)):
    jarak = np.sqrt(((train_data - test_data[i]) ** 2).sum(axis=1))
    posisi_terdekat = np.argsort(jarak)[:3]
    label_terdekat = train_label.iloc[posisi_terdekat].to_numpy()

    if label_terdekat[0] == label_terdekat[1] or label_terdekat[0] == label_terdekat[2]:
        prediksi = label_terdekat[0]
    elif label_terdekat[1] == label_terdekat[2]:
        prediksi = label_terdekat[1]
    else:
        prediksi = label_terdekat[0]

    hasil_prediksi.append(prediksi)

data.loc[posisi_kosong, "Age"] = hasil_prediksi

train_data = data[["Sex", "Age", "Pclass", "Fare"]].copy()
train_label = data["Survived"].copy()

test_data = test_dataset[["Sex", "Age", "Pclass", "Fare"]].copy()
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})

posisi_kosong = test_data[test_data.isnull().any(axis=1)].index
test_data = test_data.drop(index=posisi_kosong)
test_label = test_label_dataset["Survived"].drop(index=posisi_kosong)

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
scaler = MinMaxScaler()
train_data_normal = pd.DataFrame(
    scaler.fit_transform(train_data),
    columns=kolom_fitur,
    index=train_data.index
)
test_data_normal = pd.DataFrame(
    scaler.transform(test_data),
    columns=kolom_fitur,
    index=test_data.index
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
print()
print("Test data setelah normalisasi :")
print(test_data_normal)
