import pandas as pd

dataset = pd.read_csv("titanic.csv")

train_label = dataset["Survived"].copy()

print("=" * 50)
print("TRAIN LABEL")
print("=" * 50)
print()
print(train_label)
