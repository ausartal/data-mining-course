# Tugas Data Mining - Assignment 4
# Nama  : Ahmad Nabah Falah
# NRP   : (isi NRP)

import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

kelas = dataset["Survived"]
print(kelas)
