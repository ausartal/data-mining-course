import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Titanic-Dataset.csv")

fig, ax = plt.subplots(figsize=(10, 5))

# ubah sex jadi angka biar bisa diplot
# male = 0, female = 1
sex_angka = dataset["Sex"].map({"male": 0, "female": 1})

# bagi berdasarkan survived
for val in [0, 1]:
    idx = dataset["Survived"] == val
    if val == 0:
        warna = "red"
        label = "Tidak Selamat"
    else:
        warna = "blue"
        label = "Selamat"

    # tambah sedikit noise biar titik tidak tumpang tindih
    noise = np.random.uniform(-0.1, 0.1, size=idx.sum())
    ax.scatter(dataset.index[idx], sex_angka[idx] + noise,
               c=warna, label=label, alpha=0.5, s=15)

ax.set_yticks([0, 1])
ax.set_yticklabels(["male", "female"])
ax.set_xlabel("Urutan Data")
ax.set_ylabel("Sex")
ax.set_title("Visualisasi Sex berdasarkan Survived")
ax.legend()
plt.tight_layout()
plt.savefig("assignment_9_sex.png", dpi=150)
plt.close()

print("Plot berhasil disimpan : assignment_9_sex.png")
