import pandas as pd

test_dataset = pd.read_csv("titanic_test.csv")

test_data = test_dataset[["Age", "Fare"]].copy()

# catat posisi baris yang memiliki missing value
pos_missing_test = test_data[test_data.isnull().any(axis=1)].index

# hapus baris yang memiliki missing value
test_data = test_data.drop(index=pos_missing_test)

print("Posisi missing value test :")
print(pos_missing_test.tolist())
print()
print("Test data setelah missing value dihapus :")
print(test_data)
