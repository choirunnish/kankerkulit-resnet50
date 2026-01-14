# Klasifikasi Penyakit Kanker Kulit pada Citra Dermoskopi Menggunakan Convolutional Neural Network dengan Arsitektur Residual Network 50

## Abstrak
Kanker kulit merupakan pertumbuhan sel tidak normal pada kulit yang disebabkan oleh paparan sinar UV. Kanker kulit yang tidak mendapatkan penanganan dengan benar dapat menyebar ke dalam jaringan lain dan dapat mengakibatkan kematian. Oleh sebab itu, penelitian ini melakukan deteksi kanker kulit menggunakan metode Convolutional Neural Network (CNN) dengan model Residual Network 50 (ResNet50). CNN dipilih karena dapat mengelola, menganalisis data yang besar dan kompleks serta mengambil bentuk gambar 2D sebagai masukan. ResNet50 dipilih karena dapat mengenali pola-pola detail pada citra dermoskopi kanker kulit dan menggunakan struktur bottleneck untuk mengurangi jumlah parameter sehingga model lebih ringan, risiko overfitting lebih kecil dan dapat menghasilkan akurasi yang baik. Pemodelan CNN yang terbentuk adalah model CNN ResNet50 yang terdiri dari 152 lapisan. Berdasarkan Learning Curve, model menunjukkan akurasi yang meningkat dan loss yang menurun pada data latih tetapi terjadi fluktuasi pada data uji yang artinya model overfitting. Berdasarkan Confusion
Matrix, model menunjukkan performa yang baik pada kelas BCC, MEL, dan NV tetapi masih perlu perbaikan dalam mengidentifikasi kelas AK, SCC, dan BKL. Berdasarkan Classification Report, diperoleh nilai akurasi keseluruhan sebesar 77%. Sebuah aplikasi web berbasis model ResNet50 berhasil dibuat dan dapat berjalan dengan normal

## Dataset
Data yang digunakan dalam penelitian ini adalah data sekunder yang berbentuk citra dermoskopi kanker kulit yang diambil melalui situs International Imaging Skin Collaboration (ISIC) tahun 2019. ISIC adalah organisasi internasional yang menghimpun data citra dermoskopi untuk mengurangi kanker kulit. Data ISIC 2019 berjumlah 25331 data citra yang terbagi menjadi 8 kelas kanker kulit antara lain Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis Lesion, Dermatofibroma, Melanoma, Melanocytic Nevus, Squamous Cell Carcinoma, dan Vascular Lesion. Rinciannya dipaparkan pada Tabel 1
| Label Kelas | Singkatan | Kelas                       | Jumlah Gambar |
|-------------|-----------|-----------------------------| ------------- |
| 0           | AK        | *Actinic Keratosis*         | 867           |
| 1           | BCC       | *Basal Cell Carcinoma*      | 3323          |
| 2           | BKL       | *Benign Keratosis Lesion*   | 2624          |
| 3           | DF        | *Dermatofibroma*            | 239           |
| 4           | MEL       | *Melanoma*                  | 4522          |
| 5           | NV        | *Melanocytic Nevus*         | 12875         |
| 6           | SCC       | *Squamous Cell Carcinoma*   | 628           |
| 7           | VASC      | *Vascular Lesion*           | 253           |
|                  Total                                |  25331        |

*Tabel 1: Distribusi Citra.*

## Langkah-Langkah Penelitian
Berikut tahapan analisis yang dilakukan pada penelitian ini:
1. Mengumpulkan data berbentuk citra dermoskopi kanker kulit yang diperoleh dari situs International Imaging Skin Collaboration (ISIC)
tahun 2019.
2. Memasukkan data citra dermoskopi kanker kulit ke Google Colab.
3. Melakukan preprocessing data
Citra dermoskopi memiliki banyak noise yang dapat memengaruhi hasil klasifikasi sehingga diperlukan preprocessing data untuk memperbaiki masalah pada citra dermoskopi. Langkah-langkah preprocessing data dijelaskan sebagai berikut:
a. Melakukan resize data citra dari ukuran asli 1024 × 1024 piksel menjadi ukuran 224 × 224 piksel sesuai ketentuan pada arsitektur      ResNet50 secara otomatis.
  b. Menghilangkan hair features menggunakan dull razor filtering.
  c. Melakukan normalisasi data citra dengan cara melakukan pembagian terhadap nilai RGB dari 0 hingga 255 dengan 255, sehingga
     didapatkan nilai RGB pada rentang antara 0 hingga 1.
  d. Melakukan labelisasi citra sesuai dengan kelasnya
5. Membagi data menjadi dua bagian dengan perincian pembagian data yaitu 80% data train dan 20% data test.
6. Melakukan augmentasi data.
7. Merancang model CNN dengan arsitektur Residual Network 50 (ResNet50) yang dimana menerapkan metode transfer learning dalam            pembuatan model yang siap pakai (pre-trained model) untuk melakukan klasifikasi jenis penyakit kanker kulit.
8. Melakukan pelatihan model dengan menggunakan data train.
9. Melakukan evaluasi model berdasarkan Learning Curve.
10. Melakukan visualisasi hasil evaluasi menggunakan Confusion Matrix
   untuk didapatkan nilai akurasi, presisi, recall, dan F1-score.
11. Hasil model disimpan ke dalam format h5 untuk diimplementasikan ke dalam tampilan sebuah website menggunakan framework Flask
12. Melakukan pengujian model. Langkah-langkah pengujian model sebagai berikut:
    a. Pada tahap awal, menghidupkan server menggunakan Flask sebagai back-end dan menampilkan tampilan web pada front-end melalui           URL: http://127.0.0.1:5000.
b. Setelah server aktif, file model .h5 diload ke memori agar dapat digunakan dalam proses deteksi.
c. Pengguna mengunggah gambar dengan format file .jpg, .jpeg, atau .png melalui formulir yang tersedia di halaman utama. Gambar
   tersebut akan digunakan untuk memprediksi jenis kanker kulit.
d. Setelah pengguna mengunggah gambar, gambar akan disimpan ke dalam folder uploads pada server dan file tersebut akan diberi nama
   sesuai file asli yang diunggah.
e. Setelah gambar disimpan, gambar akan diresize menjadi ukuran 224 x 224 piksel sesuai dengan ukuran yang dibutuhkan untuk model
   ResNet50 untuk prediksi.
f. Setelah gambar diresize, gambar juga dinormalisasi dengan mengubah nilai piksel ke rentang [0, 1].
g. Setelah itu, website melakukan prediksi terhadap gambar yang telah dimasukkan oleh pengguna menggunakan model yang telah diload
   sebelumnya. Model ini akan mengidentifikasi jenis kanker kulit yang terdapat pada gambar tersebut.
h. Muncul segmen halaman website berikutnya dimana pengguna dapat melihat jenis kanker kulit beserta tingkat akurasi prediksinya. 

## Kesimpulan
Berikut adalah kesimpulan dari hasil penelitian ini:
- Pemodelan yang terbentuk adalah model CNN ResNet50 yang terdiri dari 152 lapisan yaitu 1 lapisan Input, 3 lapisan ZeroPadding2D, 52 lapisan Conv2D, 16 lapisan BatchNormalization, 52 lapisan Activation, 1 lapisan MaxPooling2D, 8 lapisan Add, 1 lapisan GlobalAveragePooling2D, 1 lapisan Dropout, dan 2 lapisan Dense.
- Berdasarkan Learning Curve, model menunjukkan akurasi meningkat dan loss menurun pada data latih tetapi terjadi fluktuasi pada data uji yang artinya model overfitting. Berdasarkan Confusion Matrix, model menunjukkan performa yang baik pada kelas BCC, MEL, dan NV tetapi masih perlu perbaikan pada kelas AK, SCC, dan BKL yang banyak kesalahan klasifikasi. Berdasarkan Classification Report, model ini diperoleh nilai akurasi keseluruhan sebesar 77%.
- Aplikasi website ini dibuat menggunakan framework Flask dengan Python untuk back-end dan HTML serta CSS untuk front-end. Model yang dilatih disimpan dalam format file .h5 untuk memproses gambar yang diunggah pengguna kemudian menampilkan hasil deteksi jenis kanker kulit dan nilai akurasi. 
