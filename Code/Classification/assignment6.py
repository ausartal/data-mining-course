import pandas as pd

test_dataset = pd.read_csv("titanic_test.csv")
test_label_dataset = pd.read_csv("titanic_testlabel.csv")

test_data = test_dataset[["Age", "Fare"]].copy()
pos_missing_test = test_data[test_data.isnull().any(axis=1)].index

# ambil label selain baris yang memiliki missing value
test_label = test_label_dataset["Survived"].drop(index=pos_missing_test)

print("Test label :")
print(test_label)
