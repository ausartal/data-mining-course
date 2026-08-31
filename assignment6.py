# Tugas Data Mining - Assignment 6
# Nama  : Ahmad Nabah Falah
# NRP   : (isi NRP)

import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

pclass_count = dataset["Pclass"].value_counts()
pclass_count = pclass_count.sort_index()

for i in pclass_count.index:
    print("Pclass", i, ":", pclass_count[i], "penumpang")
