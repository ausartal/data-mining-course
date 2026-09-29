import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

dataset = dataset.sample(frac=1, random_state=10).reset_index(drop=True)
batas = int(len(dataset) * 0.7)
train_dataset = dataset.loc[:batas - 1].copy()
test_dataset = dataset.loc[batas:].copy()

print("=" * 50)
print("HOLD-OUT METHOD 70% - 30%")
print("=" * 50)
print("Jumlah seluruh data :", len(dataset))
print("Jumlah data latih   :", len(train_dataset))
print("Jumlah data uji     :", len(test_dataset))
