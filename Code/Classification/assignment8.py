import pandas as pd

dataset = pd.read_csv("titanic.csv")
test_dataset = pd.read_csv("titanic_test.csv")

train_data = dataset[["Age", "Fare"]].copy()
pos_missing_train = train_data[train_data.isnull().any(axis=1)].index
train_data = train_data.drop(index=pos_missing_train)

test_data = test_dataset[["Age", "Fare"]].copy()
pos_missing_test = test_data[test_data.isnull().any(axis=1)].index
test_data = test_data.drop(index=pos_missing_test)

# nilai minimum dan maksimum diambil dari train data
min_data = train_data.min()
max_data = train_data.max()

# normalisasi test data memakai min dan max dari train data
test_data = (test_data - min_data) / (max_data - min_data)

print("Nilai minimum data train :")
print(min_data)
print()
print("Nilai maksimum data train :")
print(max_data)
print()
print("Test data setelah normalisasi :")
print(test_data)
