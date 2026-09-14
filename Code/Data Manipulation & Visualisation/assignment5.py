import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Name", "Sex", "Age", "Pclass", "Fare"]]
data["Relatives"] = dataset["SibSp"] + dataset["Parch"]
print(data)
