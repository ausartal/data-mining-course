# Data Mining Course

Repository ini dibuat untuk mempelajari mata kuliah **Data Mining**.

## Dataset

Dataset yang digunakan: **Titanic-Dataset.csv**

## Struktur Repository

```
Code/
├── Data Manipulation & Visualisation/   # Tugas 1 - Manipulasi & Visualisasi Data
└── Data Normalization/                  # Tugas 2 - Normalisasi Data
```

## Tugas 1 - Data Manipulation & Visualisation

| No | File | Deskripsi |
|----|------|-----------|
| 1 | `assignment1.py` | Load dataset dan tampilkan |
| 2 | `assignment2.py` | Jumlah baris dan kolom |
| 3 | `assignment3.py` | Ambil kolom fitur (Name, Sex, Age, Pclass, Fare) |
| 4 | `assignment4.py` | Ambil kolom kelas (Survived) |
| 5 | `assignment5.py` | Tambah fitur Relatives (SibSp + Parch) |
| 6 | `assignment6.py` | Hitung penumpang per Pclass |
| 7 | `assignment7.py` | Hitung penumpang per Sex |
| 8 | `assignment8.py` | Selamat/Tidak Selamat per Pclass |
| 9 | `assignment9.py` | Visualisasi Sex vs urutan data |
| 10 | `assignment10.py` | Visualisasi Age vs urutan data |

File `titanic_assign.py` berisi semua assignment dalam satu file.

### Cara Menjalankan

```bash
cd "Code/Data Manipulation & Visualisation"
python assignment1.py
python assignment2.py
...
python assignment10.py
```

### Output Visualisasi

- `assignment_9_sex.png` - Scatter plot Sex berdasarkan Survived
- `assignment_10_age.png` - Scatter plot Age berdasarkan Survived

## Tugas 2 - Data Normalization

| No | File | Deskripsi |
|----|------|-----------|
| 1 | `assignment1.py` | Load dataset dan tampilkan |
| 2 | `assignment2.py` | Jumlah baris dan kolom |
| 3 | `assignment3.py` | Ambil kolom fitur (Age, Fare) |
| 4 | `assignment4.py` | Ambil kolom kelas (Survived) |
| 5 | `assignment5.py` | Isi missing value Age dengan mean per class |
| 6 | `assignment6.py` | Normalisasi Min-Max (0-1) |
| 7 | `assignment7.py` | Normalisasi Z-Score |
| 8 | `assignment8.py` | Normalisasi Sigmoidal |

### Cara Menjalankan

```bash
cd "Code/Data Normalization"
python assignment1.py
python assignment2.py
...
python assignment8.py
```

## Penulis

Ahmad Nabah Falah
