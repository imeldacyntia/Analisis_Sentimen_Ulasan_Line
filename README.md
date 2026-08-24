# Analisis Sentimen Ulasan Aplikasi LINE Berbasis NLP Menggunakan Komparasi Model Deep Learning

## Ringkasan Proyek
Proyek ini mengimplementasikan pipeline Natural Language Processing (NLP) dan melakukan studi komparasi tiga model Deep Learning (LSTM, CNN, GRU) untuk mengklasifikasikan sentimen pengguna aplikasi LINE dari data ulasan Google Play Store ke dalam tiga kategori: Positif, Netral, dan Negatif.

* Dataset: 30.000 ulasan riil dari Google Play Store.
* Metode Pelabelan: Leksikon sentimen Bahasa Indonesia (>10.000 kata).
* Model yang Diuji: LSTM, CNN, GRU.

---

## Akuisisi dan Preprocessing Data

Pengambilan data dilakukan secara otomatis menggunakan pustaka `google-play-scraper` pada aplikasi LINE (`jp.naver.line.android`) dengan parameter wilayah Indonesia (`lang='id'`, `country='id'`).

Tahapan preprocessing teks:
1. Data Cleaning: Menghapus data kosong (missing value) dan data duplikat pada kolom teks.
2. Regex Cleaning: Menghilangkan mention (`@`), hashtag (`#`), retweet (`RT`), tautan URL, angka, dan tanda baca.
3. Case Folding: Menyeragamkan seluruh karakter teks menjadi huruf kecil (lowercase).
4. Normalisasi Kata: Mengubah kata-kata tidak baku atau singkatan ke dalam bentuk baku Bahasa Indonesia berdasarkan kamus kustom.
5. Tokenisasi: Memisahkan teks menjadi token kata menggunakan `nltk.tokenize.word_tokenize`.
6. Filtering: Menghapus stopwords umum Bahasa Indonesia dan Bahasa Inggris menggunakan korpus NLTK serta daftar kata sambung kustom.

---

## Eksplorasi dan Visualisasi Data

### Distribusi Polaritas Sentimen
Pelabelan sentimen dilakukan secara otomatis menggunakan aturan skor leksikon (3.609 kata positif dan 6.607 kata negatif):
* Positif (Skor >= 0): 18.949 ulasan (63,2%)
* Netral (-7 < Skor < 0): 7.751 ulasan (25,8%)
* Negatif (Skor <= -7): 3.300 ulasan (11,0%)

<p align="center">
  <img src="img/Distribusi%20Polaritas.png" width="450" alt="Distribusi Polaritas Sentimen"/>
</p>

### Distribusi Panjang Teks dan Kata Dominan
<p align="center">
  <img src="img/Distribusi%20Panjang%20Text.png" width="48%" alt="Distribusi Panjang Teks"/>
  <img src="img/Kata%20yang%20Paling%20Sering%20Muncul%20dalam%20Dataset.png" width="48%" alt="Kata Paling Sering Muncul"/>
</p>

* Panjang Teks: Mayoritas ulasan memiliki panjang kalimat antara 4 hingga 15 kata.
* Kata Kunci Dominan: Berdasarkan bobot TF-IDF, kata *line*, *login*, *aplikasi*, *tolong*, dan *update* memiliki frekuensi kemunculan tertinggi.

---

## Analisis Word Cloud

<p align="center">
  <img src="img/Word%20Cloud%20dari%20Tweet%20Positif.png" width="48%" alt="Word Cloud Tweet Positif"/>
  <img src="img/Word%20Cloud%20dari%20Tweet%20Negatif.png" width="48%" alt="Word Cloud Tweet Negatif"/>
</p>

<p align="center">
  <img src="img/Word%20Cloud%20dari%20Tweet%20Netral.png" width="48%" alt="Word Cloud Tweet Netral"/>
</p>

---

## Eksperimen dan Evaluasi Model

Data teks direpresentasikan ke dalam bentuk sekuens menggunakan `Tokenizer` (maksimum 2.500 kosakata) dan `pad_sequences`. Tiga model Deep Learning dilatih menggunakan optimizer Adam dan fungsi loss `categorical_crossentropy` dengan skema pembagian data dan arsitektur sebagai berikut:

| Model | Arsitektur Utama | Pembagian Data (Train / Val / Test) | Akurasi Train | Akurasi Test |
| :--- | :--- | :---: | :---: | :---: |
| LSTM | `Embedding(256)` + `LSTM(128, L2)` + `LSTM(64, L2)` + `Dense(128)` + `Dense(64)` | 70% / 20% / 10% | 96,86% | 90,50% |
| CNN | `Embedding(512)` + `Conv1D(64, k=5)` + `MaxPooling1D(2)` + `Flatten` + `Dense(64)` + `Dropout(0.5)` | 80% / 10% / 10% | 99,77% | 87,70% |
| GRU | `Embedding(512)` + `SpatialDropout1D(0.3)` + `Bi-GRU(64)` + `Bi-GRU(128)` + `Dense(128, L2)` + `Dense(64, L2)` | 90% / 5% / 5% | 93,74% | 92,20% |

---

## Uji Inferensi Sampel

Ketiga model diuji terhadap sampel teks ulasan baru untuk membandingkan hasil klasifikasi masing-masing model:

```text
Teks: "Saya sangat menyukai fitur terbaru dari LINE, tampilannya lebih menarik dan performanya lebih cepat dari sebelumnya!"
Label Asli: positive
Prediksi (LSTM): positive
Prediksi (CNN): positive
Prediksi (GRU): positive

Teks: "Aplikasi ini sering error, ribet saat login, kode verifikasi tidak masuk, akun susah diakses, dan update malah membuatnya semakin lemot."
Label Asli: negative
Prediksi (LSTM): negative
Prediksi (CNN): negative
Prediksi (GRU): negative

Teks: "Aplikasi LINE sudah saya update, tetapi saat login masih ada sedikit error, mohon perbaikan lebih lanjut."
Label Asli: neutral
Prediksi (LSTM): neutral
Prediksi (CNN): neutral
Prediksi (GRU): neutral
