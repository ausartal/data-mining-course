import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

data = dataset[["Sex", "Age", "Pclass", "Fare", "Survived"]].copy()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

posisi_train = data[data["Age"].notnull()].index
train_data = data.loc[posisi_train, ["Sex", "Pclass", "Fare", "Survived"]].copy()

print("=" * 50)
print("TRAIN DATA")
print("=" * 50)
print()
print(train_data)
