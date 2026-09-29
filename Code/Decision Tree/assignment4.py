import pandas as pd

test_dataset = pd.read_csv("titanic_test.csv")

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
test_data = test_dataset[kolom_fitur].copy()

# isi missing value Age dan Fare dengan mean
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())

test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})

print("=" * 50)
print("TEST DATA")
print("=" * 50)
print()
print("Missing value setelah diisi :")
print(test_data.isnull().sum())
print()
print(test_data)
