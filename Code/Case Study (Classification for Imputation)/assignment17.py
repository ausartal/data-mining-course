import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

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

scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_data, train_label)
hasil_prediksi = model.predict(test_data)

print("=" * 50)
print("K-NN K=3 HASIL KLASIFIKASI")
print("=" * 50)
print()
print("Jumlah data uji        :", len(test_data))
print("Hasil prediksi         :", hasil_prediksi)
