import os
import numpy as np
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

dataset = dataset.sample(frac=1, random_state=10).reset_index(drop=True)
bagian = np.array_split(dataset.index, 10)

print("=" * 50)
print("K-FOLD CROSS VALIDATION DENGAN K = 10")
print("=" * 50)
for nomor_fold in range(10):
    jumlah_test = len(bagian[nomor_fold])
    jumlah_train = len(dataset) - jumlah_test
    print("Fold", nomor_fold + 1, "-> data latih :", jumlah_train, ", data uji :", jumlah_test)
