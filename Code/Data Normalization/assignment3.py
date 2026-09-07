import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

data = dataset[["Age", "Fare"]]
print(data)
