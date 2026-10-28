# Format Laporan Evaluasi Model Regresi: ML dan DL

Dokumen ini menetapkan format laporan untuk model Machine Learning (ML) dan Deep Learning (DL) yang memprediksi densitas mikroalga. Gunakan nama model yang konsisten di semua tabel dan CSV.

---

## Format laporan ML

Bagian ini digunakan untuk model seperti `LASSO`, `DTS`, `Random Forest`, `Gradient Boosting`, `KNN`, dan `XGBoost`.

## Protokol evaluasi random untuk model baru

Gunakan protokol ini ketika melatih model baru yang hasilnya akan dibandingkan dengan skrip random ML, DL, dan LASSO. Urutan evaluasinya harus tetap: **random split 80/20 → random KFold 5-fold pada train → fit final pada seluruh train → evaluasi final pada test**.

### 1. Siapkan input dan target

Siapkan `X` sebagai matriks input dan `y` sebagai target. Pastikan jumlah baris sama, setiap baris input cocok dengan targetnya, dan tidak ada nilai kosong atau tak hingga. Kolom identitas seperti nama file tidak dimasukkan ke `X`, kecuali memang merupakan fitur yang disengaja. Cara memperoleh input boleh berbeda antar model; protokol evaluasi dimulai setelah `X` dan `y` siap.

### 2. Lakukan satu random split 80/20

Gunakan 80% sampel untuk train/pengembangan dan 20% untuk test. Tetapkan seed `42` agar split dapat direproduksi untuk urutan dan jumlah sampel yang sama.

```python
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
TEST_RATIO = 0.20

indices = np.arange(len(y))
train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_RATIO,
    shuffle=True,
    random_state=SEED,
)

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]
```

Simpan indeks atau nama sampel test untuk tabel prediksi. Setelah split ini, jangan gunakan test untuk memilih fitur, preprocessing, hyperparameter, epoch, atau model.

### 3. Jalankan random KFold 5-fold hanya pada train

Gunakan fold yang sama untuk evaluasi CV model-model yang dibandingkan:

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
```

Pada setiap putaran, empat fold menjadi data fit dan satu fold menjadi validasi. Ulangi sampai semua fold pernah menjadi validasi. Hitung MAE, RMSE, dan R² pada fold validasi, kemudian laporkan skor per fold serta mean dan standar deviasi sampel antar-fold (`ddof=1`).

Semua preprocessing yang belajar parameter dari data (misalnya standardisasi atau imputasi) harus di-fit hanya pada bagian fit fold. Untuk scikit-learn, tempatkan preprocessing dan estimator dalam `Pipeline`. Untuk model lain, hitung parameter preprocessing dari bagian fit fold lalu terapkan parameter yang sama ke fold validasi.

### 4. Lakukan tuning hanya dengan data train

Jika model membutuhkan pemilihan fitur atau hyperparameter, jangan melibatkan test. Untuk memperoleh skor CV yang tidak optimistis akibat tuning, lakukan tuning di dalam bagian fit pada setiap fold luar (nested CV), lalu evaluasi konfigurasi terpilih pada fold validasi luar. Aturan tuning harus ditetapkan sebelum melihat skor test.

Untuk model yang memakai jumlah epoch tetap, tetapkan jumlah epoch sebelum evaluasi. Jika memakai early stopping atau memilih epoch, gunakan validation yang berasal dari data train dan jangan menggunakan test untuk menentukan epoch.

### 5. Fit ulang pada seluruh train

Setelah CV dan tuning selesai, buat/reset model dengan konfigurasi final, lalu fit pada seluruh 80% data train. Jangan memakai model yang hanya dilatih pada bagian fit dari salah satu fold.

### 6. Evaluasi final pada test

Prediksi test menggunakan model final dan hitung MAE, RMSE, serta R². Evaluasi test dilakukan setelah keputusan model dan konfigurasi final ditetapkan. Simpan nama sampel, nilai aktual, prediksi, dan residual (`actual - predicted`). Skor test adalah evaluasi akhir, bukan dasar untuk memilih model.

### Checklist keseragaman

| Pemeriksaan | Standar |
|---|---|
| Split | Acak, 80% train dan 20% test |
| Seed split | `random_state=42` |
| CV | `KFold(n_splits=5, shuffle=True, random_state=42)` pada train saja |
| Preprocessing dan tuning | Tidak menggunakan test; preprocessing di-fit ulang dalam setiap fold |
| Fit final | Seluruh 80% train |
| Evaluasi final | Test 20%, setelah fit final |
| Metrik | MAE, RMSE, dan R² untuk CV dan test |
| Hasil tersimpan | Skor per fold, ringkasan CV, metrik test, aktual, prediksi, residual |

> Seed yang sama membuat split dapat direproduksi jika urutan dan jumlah sampel sama. Jika sampel atau urutannya berbeda, seed yang sama tidak menjamin test berisi observasi yang sama.

## Protokol evaluasi kronologis

- Urutkan sampel berdasarkan timestamp gambar sebelum membagi data.
- Gunakan 80% data paling awal sebagai data train/pengembangan dan 20% data paling akhir sebagai data test.
- Jalankan 5-fold `TimeSeriesSplit` hanya pada data train.
- Untuk tiap fold, hitung MAE, RMSE, dan R² pada bagian validasi fold.
- Ringkas skor lima fold dengan mean ± standar deviasi sampel (`ddof=1`). Simpan mean dan std di kolom numerik terpisah.
- Latih model final pada seluruh data train, lalu evaluasi pada data train dan test.
- Simpan prediksi, nilai aktual, serta residual test. Residual dihitung sebagai `actual - predicted`.

> Jika model memiliki hyperparameter yang dipilih melalui cross-validation, lakukan pemilihannya hanya menggunakan data train. Untuk menghindari bias pada skor CV, pilih hyperparameter di bagian train masing-masing fold (nested CV), lalu nilai pada bagian validasi fold tersebut.

## Tabel laporan

### Skor CV setiap fold

| Model | Fold | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| NAMA_MODEL | 1 | ... | ... | ... |
| NAMA_MODEL | 2 | ... | ... | ... |
| NAMA_MODEL | 3 | ... | ... | ... |
| NAMA_MODEL | 4 | ... | ... | ... |
| NAMA_MODEL | 5 | ... | ... | ... |

### Ringkasan CV

Tampilkan ringkasan untuk dibaca dengan format `mean ± std`, misalnya `-0.264 ± 0.273`. Mean dan std adalah dua angka terpisah: mean dapat positif atau negatif, sedangkan std tidak negatif.

| Model | MAE (mean ± std) | RMSE (mean ± std) | R2 (mean ± std) |
|---|---:|---:|---:|
| NAMA_MODEL | ... ± ... | ... ± ... | ... ± ... |

### Evaluasi train dan test

| Model | Data | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| NAMA_MODEL | Train | ... | ... | ... |
| NAMA_MODEL | Test | ... | ... | ... |

### Prediksi data test

| nama_file | actual | predicted_NAMA_MODEL | residual_NAMA_MODEL |
|---|---:|---:|---:|
| nama_gambar_1 | ... | ... | ... |
| nama_gambar_2 | ... | ... | ... |

## Format empat CSV utama

Semua model menggunakan nama file CSV yang sama. Tambahkan baris hasil masing-masing model ke CSV yang sama agar model dapat dibandingkan langsung.

### `hasil_cv_per_fold.csv`

Satu baris untuk setiap kombinasi model dan fold.

| Kolom | Isi |
|---|---|
| `Model` | Nama model |
| `Fold` | Nomor fold, 1 sampai 5 |
| `MAE` | Mean Absolute Error fold validasi |
| `RMSE` | Root Mean Squared Error fold validasi |
| `R2` | Koefisien determinasi fold validasi |

### `hasil_cv_ringkasan.csv`

Satu baris untuk setiap model. Simpan angka mean dan std di kolom terpisah agar tetap numerik dan mudah divisualisasikan. Format `mean ± std` digunakan ketika menampilkan tabel.

| Kolom | Isi |
|---|---|
| `Model` | Nama model |
| `MAE_mean` | Rata-rata MAE antar-fold |
| `MAE_std` | Simpangan baku sampel MAE antar-fold |
| `RMSE_mean` | Rata-rata RMSE antar-fold |
| `RMSE_std` | Simpangan baku sampel RMSE antar-fold |
| `R2_mean` | Rata-rata R² antar-fold |
| `R2_std` | Simpangan baku sampel R² antar-fold |
| `n_folds` | Jumlah fold yang diringkas, biasanya 5 |

### `hasil_evaluasi_train_test_semua_model.csv`

Dua baris untuk setiap model: satu untuk train dan satu untuk test.

| Kolom | Isi |
|---|---|
| `Model` | Nama model |
| `Data` | `Train` atau `Test` |
| `MAE` | MAE pada subset tersebut |
| `RMSE` | RMSE pada subset tersebut |
| `R2` | R² pada subset tersebut |

### `prediksi_test_semua_model.csv`

Satu baris untuk setiap sampel test. Gunakan satu pasang kolom prediksi dan residual untuk setiap model.

| Kolom | Isi |
|---|---|
| `nama_file` | Nama atau identitas gambar test |
| `actual` | Densitas aktual |
| `predicted_NAMA_MODEL` | Prediksi model; ganti `NAMA_MODEL` dengan nama model |
| `residual_NAMA_MODEL` | `actual - predicted` untuk model tersebut |

Contoh kolom untuk dua model: `predicted_LASSO`, `residual_LASSO`, `predicted_Random Forest`, `residual_Random Forest`.

## Interpretasi singkat

- MAE dan RMSE yang lebih kecil menunjukkan kesalahan prediksi yang lebih kecil.
- R² yang lebih besar menunjukkan kecocokan yang lebih baik. R² dapat negatif jika model lebih buruk daripada prediksi menggunakan rata-rata target.
- Standar deviasi merangkum sebaran skor antar-fold; nilainya tidak menunjukkan arah dan tidak negatif.
- Nilai metrik test adalah evaluasi pada periode yang ditahan dan tidak digunakan untuk memilih model atau hyperparameter.

---

## Format laporan DL

Bagian ini mengikuti skrip DL multi-model yang menggunakan random split dan 5-fold `KFold`. Pipeline CNN moving-window juga menggunakan format CV dan train/test ini, dengan split serta evaluasi di tingkat gambar sumber. Rincian khususnya dijelaskan pada bagian “Format CNN moving-window”.

### Protokol evaluasi DL

- Gunakan pembagian acak 80% data pengembangan dan 20% data test.
- Pada 80% data pengembangan, gunakan 5-fold `KFold` dengan `shuffle=True` dan seed tetap.
- Catat metrik validasi setiap fold: MAE, RMSE, dan R².
- Ringkas metrik fold per model dengan mean ± std sampel (`ddof=1`). Di CSV, simpan mean dan std sebagai kolom numerik terpisah.
- Catat train loss, validation loss, train MAE, dan validation MAE setiap epoch pada setiap fold. Ringkas riwayat per model dan epoch dengan mean ± std antar-fold.
- Latih ulang setiap model pada seluruh 80% data pengembangan dan evaluasi pada data train serta test.
- Simpan prediksi CV, prediksi test, residual, metrik, riwayat epoch, dan checkpoint model terbaik.
- Pilih model terbaik berdasarkan MAE rata-rata CV terkecil. Evaluasi test hanya dilakukan untuk pelaporan akhir.

### Tabel laporan DL

#### Informasi split

| Item | Nilai |
|---|---|
| Jumlah pasangan gambar-label | ... |
| Data pengembangan/train | 80% (... sampel) |
| Data test | 20% (... sampel) |
| Metode split | Random, seed tetap |
| Jumlah fold CV | 5 (`KFold`, shuffle) |
| Model yang dievaluasi | ... |

#### Skor CV setiap fold

| Model | Fold | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| NAMA_MODEL | 1 | ... | ... | ... |
| NAMA_MODEL | 2 | ... | ... | ... |
| NAMA_MODEL | 3 | ... | ... | ... |
| NAMA_MODEL | 4 | ... | ... | ... |
| NAMA_MODEL | 5 | ... | ... | ... |

#### Ringkasan CV

Tampilkan hasil sebagai mean ± std antar-fold. CSV tetap menyimpan mean dan std terpisah dalam bentuk numerik.

| Model | MAE (mean ± std) | RMSE (mean ± std) | R2 (mean ± std) |
|---|---:|---:|---:|
| NAMA_MODEL | ... ± ... | ... ± ... | ... ± ... |

#### Evaluasi final pada train dan test

| Model | Data | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| NAMA_MODEL | Train | ... | ... | ... |
| NAMA_MODEL | Test | ... | ... | ... |

#### Prediksi test

| Nama_Gambar | actual | predicted_nama_model | residual_nama_model |
|---|---:|---:|---:|
| nama_gambar_1 | ... | ... | ... |
| nama_gambar_2 | ... | ... | ... |

Residual dihitung sebagai `actual - predicted`. Untuk banyak model, satu CSV prediksi berisi satu pasang kolom prediksi dan residual untuk setiap model.

### Format CSV DL

Simpan semua CSV DL di folder output run yang sama. Nama file berikut mengikuti skrip DL yang dilampirkan.

#### `dl_cv_metrics_per_fold.csv`

Satu baris per model dan fold; berisi `Model`, `Fold`, `MAE`, `RMSE`, dan `R2`.

#### `dl_cv_model_summary.csv`

Satu baris per model; kolomnya `Model`, `MAE_mean`, `MAE_std`, `RMSE_mean`, `RMSE_std`, `R2_mean`, `R2_std`, dan `n_folds`. Untuk laporan yang dibaca, tampilkan pasangan masing-masing sebagai `mean ± std`.

#### `dl_cv_loss_per_fold_per_epoch.csv`

Riwayat pelatihan dan validasi untuk setiap model, fold, dan epoch. Kolom meliputi `Model`, `Fold`, `epoch`, `train_loss`, `train_mae`, `validation_loss`, dan `validation_mae`.

#### `dl_cv_loss_mean_per_epoch.csv`

Ringkasan riwayat per model dan epoch. Kolom meliputi `Model`, `epoch`, `train_loss_mean`, `train_loss_std`, `validation_loss_mean`, `validation_loss_std`, `train_mae_mean`, `train_mae_std`, `validation_mae_mean`, dan `validation_mae_std`.

#### `dl_cv_predictions.csv`

Prediksi validasi CV per model, fold, dan gambar. Kolom: `Model`, `Fold`, `Nama_Gambar`, `actual`, `predicted`, `residual`.

#### `dl_final_train_loss_per_epoch.csv`

Riwayat pelatihan model final pada seluruh data pengembangan. Kolom: `Model`, `epoch`, `train_loss`, dan `train_mae`.

#### `dl_final_metrics_train_test.csv`

Dua baris per model (`Train` dan `Test`). Kolom: `Model`, `Data`, `MAE`, `RMSE`, `R2`.

#### `dl_test_predictions_all_models.csv`

Satu baris per gambar test. Kolom identitas dan target: `Nama_Gambar`, `actual`. Berikutnya, satu pasangan `predicted_nama_model` dan `residual_nama_model` untuk setiap model.

#### `dl_test_metrics_all_models.csv`

Satu baris per model. Kolom: `Model`, `MAE`, `RMSE`, `R2` pada test.

Checkpoint model yang dipilih berdasarkan MAE CV disimpan sebagai `dl_best_model_by_cv.pth`.

### Visualisasi DL

- **Kurva MAE CV per model**: epoch pada sumbu x; mean train MAE dan validation MAE dari lima fold pada sumbu y; pita di sekitar kurva menunjukkan ± std antar-fold. Simpan satu gambar per model, misalnya `mae_cv_nama_model.png`.
- **Evaluasi test per model**: satu gambar memuat plot residual terhadap prediksi, plot actual versus predicted dengan garis prediksi sempurna, dan MAE/RMSE/R² test. Simpan sebagai `nama_model_evaluasi_test.png`.
- Visualisasi dan CSV harus berasal dari run, pembagian data, serta prediksi yang sama.

### Cara membaca hasil DL

- MAE dan RMSE lebih kecil berarti kesalahan prediksi lebih kecil.
- R² lebih besar biasanya lebih baik; R² negatif berarti hasil model lebih buruk daripada baseline rata-rata target pada subset yang dievaluasi.
- Pada ringkasan CV, mean R² dapat negatif atau positif. Std selalu tidak negatif dan menunjukkan variasi skor antar-fold.
- Pita ± std pada kurva epoch menunjukkan variasi antar-fold pada setiap epoch, bukan confidence interval.
- Hasil DL dengan random split tidak boleh dibandingkan seolah-olah memakai pembagian yang sama dengan ML chronological. Untuk perbandingan langsung, kedua pendekatan perlu memakai sampel test dan protokol split yang sama.

---

