# Case Study (Classification for Imputation)

Tugas mata kuliah Data Mining - Case Study (Classification for Imputation).

Imputasi missing value kolom Age dengan K-NN, lalu klasifikasi Survived pada data uji Titanic.

## Dataset

Dataset yang digunakan: **titanic.csv**, **titanic_test.csv**, **titanic_testlabel.csv**

## Narasi

a. Dataset yang digunakan berjumlah 891 data penumpang Titanic dengan fitur Sex, Age, Pclass, Fare, dan Survived.

b. Kolom Age memiliki missing value sebanyak 177 data. Data yang Age-nya tersedia digunakan sebagai training, sedangkan data yang Age-nya kosong digunakan sebagai testing.

c. Pengisian missing value dilakukan dengan K-NN K=3, di mana Age diperlakukan sebagai class dan fitur yang digunakan adalah Sex, Pclass, Fare, dan Survived.

d. Sebelum klasifikasi, data dinormalisasi menggunakan Min-Max 0-1. Nilai minimum dan maksimum setiap atribut diambil dari data latih, lalu dipakai kembali untuk mengubah data uji.

e. Setelah seluruh missing value Age terisi, data digunakan kembali untuk memprediksi kolom Survived pada titanic_test.csv dengan K-NN K=3.

f. Data uji yang mengandung missing value dihapus, sehingga tersisa 331 dari 418 data pengujian.

g. Hasil prediksi dibandingkan dengan label sebenarnya dari titanic_testlabel.csv, lalu dihitung jumlah error dan error ratio.

## Hasil

| Metrik | Nilai |
|---|---|
| Jumlah data uji | 331 |
| Jumlah error | 61 |
| Error ratio | 0.1843 |
| Error percentage | 18.43 % |

## Daftar Assignment

| No | File | Deskripsi |
|----|------|-----------|
| 1 | `assignment01.py` | Load dataset dan tampilkan |
| 2 | `assignment02.py` | Ambil fitur Sex, Age, Pclass, Fare, Survived |
| 3 | `assignment03.py` | Ambil train data dari data yang Age tersedia |
| 4 | `assignment04.py` | Ambil train label Age |
| 5 | `assignment05.py` | Ambil test data dari data yang Age kosong |
| 6 | `assignment06.py` | Normalisasi Min-Max train data |
| 7 | `assignment07.py` | Normalisasi Min-Max test data |
| 8 | `assignment08.py` | K-NN K=3 prediksi Age |
| 9 | `assignment09.py` | Isi missing value Age |
| 10 | `assignment10.py` | Load test dataset |
| 11 | `assignment11.py` | Ambil train data fitur Sex, Age, Pclass, Fare |
| 12 | `assignment12.py` | Ambil train label Survived |
| 13 | `assignment13.py` | Ambil test data dan hapus missing value |
| 14 | `assignment14.py` | Ambil test label |
| 15 | `assignment15.py` | Normalisasi Min-Max train data |
| 16 | `assignment16.py` | Normalisasi Min-Max test data |
| 17 | `assignment17.py` | K-NN K=3 hasil klasifikasi Survived |
| 18 | `assignment18.py` | Hitung jumlah error |
| 19 | `assignment19.py` | Hitung error ratio |

## Cara Menjalankan

```bash
python assignment01.py
python assignment02.py
...
python assignment19.py
```

## Library yang Digunakan

- pandas
- numpy
- scikit-learn

## Penulis

Ahmad Nabah Falah
