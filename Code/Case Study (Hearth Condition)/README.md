# Case Study (Hearth Condition)

Tugas mata kuliah Data Mining - Case Study (Hearth Condition).

Klasifikasi penyakit jantung menggunakan K-NN dan Decision Tree dengan berbagai kombinasi normalisasi dan validasi.

## Dataset

Dataset yang digunakan: **Heart_dataset.csv**

## Narasi

a. Dataset yang digunakan berjumlah 270 data dengan 13 fitur.

b. Target dibagi menjadi dua kondisi, yaitu:
   - Absence sebanyak 150 data
   - Presence sebanyak 120 data

c. Pengujian dilakukan menggunakan K-NN dengan nilai K=1, K=5, dan K=9, serta Decision Tree.

d. Sebelum klasifikasi, data dinormalisasi menggunakan Min-Max, Z-Score, dan Sigmoidal.

e. Setiap kombinasi diuji dengan K-Fold K=4, K-Fold K=10, dan Leave-One-Out.

f. Total terdapat 36 kombinasi pengujian.

g. Normalisasi dilakukan berdasarkan data latih agar data uji tidak ikut memengaruhi proses pelatihan.

h. Hasil setiap percobaan dibandingkan berdasarkan jumlah prediksi benar, accuracy, dan error ratio.

## Daftar Assignment

| No | File | Deskripsi |
|----|------|-----------|
| 1 | `assignment01.py` | Load dataset dan tampilkan |
| 2 | `assignment01_encode.py` | Encoding fitur kategorikal |
| 3 | `assignment02_minmax.py` | Normalisasi Min-Max |
| 4 | `assignment03_zscore.py` | Normalisasi Z-Score |
| 5 | `assignment04_sigmoid.py` | Normalisasi Sigmoidal |
| 6 | `assignment05` s.d. `assignment40` | 36 kombinasi model, normalisasi, dan validasi |
| 7 | `assignment41_error_ratio.py` | Ringkasan error ratio dan kesimpulan |

## Keterangan suffix

Pola nama: `assignment{NN}_{model}_{normalisasi}_{validasi}.py`

| Suffix | Keterangan |
|---|---|
| `kn1` / `kn5` / `kn9` | K-NN dengan K=1 / K=5 / K=9 |
| `tree` | Decision Tree |
| `minmax` / `zscore` / `sigmoid` | Metode normalisasi |
| `k4` / `k10` / `loo` | K-Fold K=4 / K-Fold K=10 / Leave-One-Out |

## Cara Menjalankan

```bash
python assignment01.py
python assignment01_encode.py
python assignment02_minmax.py
...
python assignment41_error_ratio.py
```

## Library yang Digunakan

- pandas
- numpy
- matplotlib
- scikit-learn

## Penulis

Ahmad Nabah Falah
