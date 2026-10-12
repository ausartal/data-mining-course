import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

data = dataset[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

posisi_train = data[data["Age"].notnull()].index
posisi_kosong = data[data["Age"].isnull()].index

train_data = data.loc[posisi_train, ["Sex", "Pclass", "Fare", "Survived"]].copy()
train_label = data.loc[posisi_train, "Age"].copy()
test_data = data.loc[posisi_kosong, ["Sex", "Pclass", "Fare", "Survived"]].copy()

print("=" * 50)
print("TRAIN DATA")
print("=" * 50)
print()
print(train_data)
print()
print("=" * 50)
print("TRAIN LABEL")
print("=" * 50)
print()
print(train_label)
print()
print("=" * 50)
print("TEST DATA")
print("=" * 50)
print()
print(test_data)
