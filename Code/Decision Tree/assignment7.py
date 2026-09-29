import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("titanic.csv")
test_dataset = pd.read_csv("titanic_test.csv")
test_label_dataset = pd.read_csv("titanic_testlabel.csv")

kolom_fitur = ["Sex", "Age", "Pclass", "Fare"]
train_data = dataset[kolom_fitur].copy()

mean_age_setiap_class = dataset.groupby("Survived")["Age"].mean()

for kelas in mean_age_setiap_class.index:
    posisi_kosong = (dataset["Survived"] == kelas) & (train_data["Age"].isnull())
    train_data.loc[posisi_kosong, "Age"] = mean_age_setiap_class[kelas]

train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
train_label = dataset["Survived"].copy()

test_data = test_dataset[kolom_fitur].copy()
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())
test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
test_label = test_label_dataset["Survived"].copy()

dtc = DecisionTreeClassifier(random_state=10)
dtc.fit(train_data, train_label)

class_result = dtc.predict(test_data)

jumlah_error = (class_result != test_label.to_numpy()).sum()
error_ratio = jumlah_error / len(test_label)

print("=" * 50)
print("KLASIFIKASI DECISION TREE")
print("=" * 50)
print()
print("Hasil prediksi test data :")
print(class_result)
print()
print("Jumlah data uji :", len(test_label))
print("Jumlah error    :", jumlah_error)
print("Error ratio     :", round(error_ratio, 4))
print("Error percentage:", round(error_ratio * 100, 2), "%")
