import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

print("=" * 50)
print("DATASET")
print("=" * 50)
print()
print(dataset)
