# Klasifikasi Penyakit Kanker Kulit pada Citra Dermoskopi Menggunakan Convolutional Neural Network dengan Arsitektur Residual Network 50

## Abstrak
Kanker kulit merupakan pertumbuhan sel tidak normal pada kulit yang disebabkan oleh paparan sinar UV. Kanker kulit yang tidak mendapatkan penanganan dengan benar dapat menyebar ke dalam jaringan lain dan dapat mengakibatkan kematian. Oleh sebab itu, penelitian ini melakukan deteksi kanker kulit menggunakan metode Convolutional Neural Network (CNN) dengan model Residual Network 50 (ResNet50). CNN dipilih karena dapat mengelola, menganalisis data yang besar dan kompleks serta mengambil bentuk gambar 2D sebagai masukan. ResNet50 dipilih karena dapat mengenali pola-pola detail pada citra dermoskopi kanker kulit dan menggunakan struktur bottleneck untuk mengurangi jumlah parameter sehingga model lebih ringan, risiko overfitting lebih kecil dan dapat menghasilkan akurasi yang baik. Pemodelan CNN yang terbentuk adalah model CNN ResNet50 yang terdiri dari 152 lapisan. Berdasarkan Learning Curve, model menunjukkan akurasi yang meningkat dan loss yang menurun pada data latih tetapi terjadi fluktuasi pada data uji yang artinya model overfitting. Berdasarkan Confusion
Matrix, model menunjukkan performa yang baik pada kelas BCC, MEL, dan NV tetapi masih perlu perbaikan dalam mengidentifikasi kelas AK, SCC, dan BKL. Berdasarkan Classification Report, diperoleh nilai akurasi keseluruhan sebesar 77%. Sebuah aplikasi web berbasis model ResNet50 berhasil dibuat dan dapat berjalan dengan normal

## Dataset
Data yang digunakan dalam penelitian ini adalah data sekunder yang berbentuk citra dermoskopi kanker kulit yang diambil melalui situs International Imaging Skin Collaboration (ISIC) tahun 2019. ISIC adalah organisasi internasional yang menghimpun data citra dermoskopi untuk mengurangi kanker kulit. Data ISIC 2019 berjumlah 25331 data citra yang terbagi menjadi 8 kelas kanker kulit antara lain Actinic Keratosis, Basal Cell Carcinoma, Benign Keratosis Lesion, Dermatofibroma, Melanoma, Melanocytic Nevus, Squamous Cell Carcinoma, dan Vascular Lesion. 
| Label Kelas | Singkatan                                                              | Target  |
|-------------|-----------------------------------------------------------------------------|---------|
| 0           | AK                                                                      | NV      |
| 1           | BCC                                                                    | MEL     |
| 2           | BKL                                                                            | BCC*    |
| 3           | DF      |
| 4           | MEL                                                                            | AK*     |
| 5           | NV                                                                            | SCC*    |
| 6           | SCC                                                                             | VASC*   |
| 7           | VASC                                                                            | DF*     |
*Table 1: Mapping from diagnosis to targets.*


## Kesimpulan
Berikut adalah kesimpulan dari hasil penelitian ini:
- Pemodelan yang terbentuk adalah model CNN ResNet50 yang terdiri dari 152 lapisan yaitu 1 lapisan Input, 3 lapisan ZeroPadding2D, 52 lapisan Conv2D, 16 lapisan BatchNormalization, 52 lapisan Activation, 1 lapisan MaxPooling2D, 8 lapisan Add, 1 lapisan GlobalAveragePooling2D, 1 lapisan Dropout, dan 2 lapisan Dense.
- Berdasarkan Learning Curve, model menunjukkan akurasi meningkat dan loss menurun pada data latih tetapi terjadi fluktuasi pada data uji yang artinya model overfitting. Berdasarkan Confusion Matrix, model menunjukkan performa yang baik pada kelas BCC, MEL, dan NV tetapi masih perlu perbaikan pada kelas AK, SCC, dan BKL yang banyak kesalahan klasifikasi. Berdasarkan Classification Report, model ini diperoleh nilai akurasi keseluruhan sebesar 77%.
- Aplikasi website ini dibuat menggunakan framework Flask dengan Python untuk back-end dan HTML serta CSS untuk front-end. Model yang dilatih disimpan dalam format file .h5 untuk memproses gambar yang diunggah pengguna kemudian menampilkan hasil deteksi jenis kanker kulit dan nilai akurasi. 
