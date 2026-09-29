import os
import pandas as pd

folder = os.path.dirname(os.path.abspath(__file__))
dataset = pd.read_csv(os.path.join(folder, "titanic.csv"))

print("=" * 50)
print("LEAVE-ONE-OUT CROSS VALIDATION")
print("=" * 50)
print("Jumlah seluruh data :", len(dataset))
print("Jumlah percobaan    :", len(dataset))
print("Data latih setiap percobaan :", len(dataset) - 1)
print("Data uji setiap percobaan   : 1")
