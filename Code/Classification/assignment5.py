import pandas as pd

dataset = pd.read_csv("titanic.csv")

train_data = dataset[["Age", "Fare"]].copy()
pos_missing_train = train_data[train_data.isnull().any(axis=1)].index

# ambil label selain baris yang memiliki missing value
train_label = dataset["Survived"].drop(index=pos_missing_train)

print("Train label :")
print(train_label)
