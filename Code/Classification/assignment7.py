import pandas as pd

dataset = pd.read_csv("titanic.csv")

train_data = dataset[["Age", "Fare"]].copy()
pos_missing_train = train_data[train_data.isnull().any(axis=1)].index
train_data = train_data.drop(index=pos_missing_train)

# simpan nilai minimum dan maksimum setiap atribut
min_data = train_data.min()
max_data = train_data.max()

# normalisasi Min-Max 0 sampai 1
train_data = (train_data - min_data) / (max_data - min_data)

print("Nilai minimum data train :")
print(min_data)
print()
print("Nilai maksimum data train :")
print(max_data)
print()
print("Train data setelah normalisasi :")
print(train_data)
