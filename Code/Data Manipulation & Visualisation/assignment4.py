import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

kelas = dataset["Survived"]
print(kelas)
