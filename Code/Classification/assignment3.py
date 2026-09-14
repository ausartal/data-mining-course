import pandas as pd

dataset = pd.read_csv("titanic.csv")

train_data = dataset[["Age", "Fare"]].copy()

# catat posisi baris yang memiliki missing value
pos_missing_train = train_data[train_data.isnull().any(axis=1)].index

# hapus baris yang memiliki missing value
train_data = train_data.drop(index=pos_missing_train)

print("Posisi missing value train :")
print(pos_missing_train.tolist())
print()
print("Train data setelah missing value dihapus :")
print(train_data)
