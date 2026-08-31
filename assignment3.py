import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Name", "Sex", "Age", "Pclass", "Fare"]]
print(data)
