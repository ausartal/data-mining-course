import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

sex_count = dataset["Sex"].value_counts()

for i in sex_count.index:
    print(i, ":", sex_count[i], "penumpang")
