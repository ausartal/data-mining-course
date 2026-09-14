import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

rows = dataset.shape[0]
cols = dataset.shape[1]

print("Jumlah baris :", rows)
print("Jumlah kolom :", cols)
