import pandas as pd

test_label_dataset = pd.read_csv("titanic_testlabel.csv")

test_label = test_label_dataset["Survived"].copy()

print("=" * 50)
print("TEST LABEL")
print("=" * 50)
print()
print(test_label)
