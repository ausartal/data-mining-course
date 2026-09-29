import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text

dataset = pd.read_csv("titanic.csv")

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
train_data = dataset[kolom_fitur].copy()

mean_age_setiap_class = dataset.groupby("Survived")["Age"].mean()

for kelas in mean_age_setiap_class.index:
    posisi_kosong = (dataset["Survived"] == kelas) & (train_data["Age"].isnull())
    train_data.loc[posisi_kosong, "Age"] = mean_age_setiap_class[kelas]

train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
train_label = dataset["Survived"].copy()

dtc = DecisionTreeClassifier(random_state=10)
dtc.fit(train_data, train_label)

hirarki = export_text(dtc, feature_names=list(train_data.columns))

print("=" * 50)
print("HIRARKI DECISION TREE")
print("=" * 50)
print()
print(hirarki)

fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(dtc, feature_names=kolom_fitur, class_names=["Tidak Selamat", "Selamat"],
               filled=True, rounded=True, max_depth=4, fontsize=8, ax=ax)
plt.tight_layout()
plt.savefig("decision_tree_hierarchy.png", dpi=150)
plt.close()

print()
print("Gambar hirarki disimpan : decision_tree_hierarchy.png")
