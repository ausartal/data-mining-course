import pandas as pd
import numpy as np

dataset = pd.read_csv("titanic.csv")
test_dataset = pd.read_csv("titanic_test.csv")
test_label_dataset = pd.read_csv("titanic_testlabel.csv")

train_data = dataset[["Age", "Fare"]].copy()
pos_missing_train = train_data[train_data.isnull().any(axis=1)].index
train_data = train_data.drop(index=pos_missing_train)
train_label = dataset["Survived"].drop(index=pos_missing_train)

test_data = test_dataset[["Age", "Fare"]].copy()
pos_missing_test = test_data[test_data.isnull().any(axis=1)].index
test_data = test_data.drop(index=pos_missing_test)
test_label = test_label_dataset["Survived"].drop(index=pos_missing_test)

min_data = train_data.min()
max_data = train_data.max()

train_data = (train_data - min_data) / (max_data - min_data)
test_data = (test_data - min_data) / (max_data - min_data)

class_result = pd.DataFrame(index=test_data.index)

for k in range(1, 16):
    hasil_k = []

    for i in test_data.index:
        jarak = np.sqrt(((train_data - test_data.loc[i]) ** 2).sum(axis=1))
        pos_terdekat = np.argsort(jarak.to_numpy())[:k]
        label_terdekat = train_label.iloc[pos_terdekat]
        jumlah_label = np.bincount(label_terdekat.to_numpy())
        hasil = np.argmax(jumlah_label)
        hasil_k.append(hasil)

    class_result["k=" + str(k)] = hasil_k

print("Error ratio setiap nilai k :")

for k in range(1, 16):
    jumlah_error = (class_result["k=" + str(k)] != test_label).sum()
    error_ratio = jumlah_error / len(test_label)

    print("k =", k,
          "-> jumlah error :", jumlah_error,
          ", error ratio :", round(error_ratio, 4),
          "(", round(error_ratio * 100, 2), "%)")
