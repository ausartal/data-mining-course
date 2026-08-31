import pandas as pd

dataset = pd.read_csv("Titanic-Dataset.csv")

for pclass_val in sorted(dataset["Pclass"].unique()):
    data_kelas = dataset[dataset["Pclass"] == pclass_val]
    tidak_selamat = len(data_kelas[data_kelas["Survived"] == 0])
    selamat = len(data_kelas[data_kelas["Survived"] == 1])
    print("Pclass", pclass_val, "-> Tidak Selamat:", tidak_selamat, ", Selamat:", selamat)
