import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dataset = pd.read_csv("Titanic-Dataset.csv")

# hapus baris yang Age-nya kosong
dataset_bersih = dataset.dropna(subset=["Age"])

fig, ax = plt.subplots(figsize=(10, 5))

for val in [0, 1]:
    idx = dataset_bersih["Survived"] == val
    if val == 0:
        warna = "red"
        label = "Tidak Selamat"
    else:
        warna = "blue"
        label = "Selamat"

    ax.scatter(dataset_bersih.index[idx], dataset_bersih["Age"][idx],
               c=warna, label=label, alpha=0.5, s=15)

ax.set_xlabel("Urutan Data")
ax.set_ylabel("Age")
ax.set_title("Visualisasi Age berdasarkan Survived")
ax.legend()
plt.tight_layout()
plt.savefig("assignment_10_age.png", dpi=150)
plt.close()

print("Plot berhasil disimpan : assignment_10_age.png")
