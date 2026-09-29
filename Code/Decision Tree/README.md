# Decision Tree

Tugas mata kuliah Data Mining - Decision Tree.

Klasifikasi data Titanic menggunakan Decision Tree.

## Dataset

Dataset yang digunakan:

- `titanic.csv`
- `titanic_test.csv`
- `titanic_testlabel.csv`

## Daftar Assignment

| No | File | Deskripsi |
|----|------|-----------|
| 1 | `assignment1.py` | Load dataset dan tampilkan |
| 2 | `assignment2.py` | Load test dataset dan tampilkan |
| 3 | `assignment3.py` | Ambil kolom fitur (Sex, Age, Pclass, Fare) dan isi missing value Age dengan mean per class |
| 4 | `assignment4.py` | Ambil kolom fitur test (Sex, Age, Pclass, Fare) |
| 5 | `assignment5.py` | Ambil kolom kelas (Survived) sebagai train label |
| 6 | `assignment6.py` | Load test label dari titanic_testlabel.csv |
| 7 | `assignment7.py` | Klasifikasi Decision Tree dan hitung error ratio |
| 8 | `assignment8.py` | Tampilkan hirarki Decision Tree |

## Cara Menjalankan

```bash
python assignment1.py
python assignment2.py
...
python assignment8.py
```

## Hasil

- Error ratio klasifikasi Decision Tree: **0.2368 (23.68%)**
- Gambar hirarki: `decision_tree_hierarchy.png`

## Library yang Digunakan

- pandas
- scikit-learn
- matplotlib

## Penulis

Ahmad Nabah Falah
