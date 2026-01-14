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
| Total       |           |                             |  25331        |

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

## Pembuatan Model Arsitektur Residual Network 50 
ResNet50 adalah model deep learning yang digunakan dalam penelitian ini untuk mendeteksi kanker kulit dari citra dermoskopi. Model ini memanfaatkan transfer learning dengan bobot pra-latih dari dataset ImageNet yang terdiri dari sekitar 1,2 juta gambar dengan 1000 kategori kelas. Dengan transfer learning, model tidak perlu melatih bobot dari awal, sehingga prosesnya menjadi lebih cepat. Struktur model ResNet50 dapat dilihat pada Gambar 1.

![Gambar 1](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture6.png)

*Gambar 1: Struktur Model Residual Network 50*

Model ini terdiri dari 7 lapisan yaitu lapisan input, zero padding, blok 1, residual block 1, residual block 2 hingga residual block 4, dan lapisan klasifikasi. Lapisan input berupa data citra kanker kulit berukuran 224x224 piksel dan 3 RGB sebagai data masukan. Block 1 terdiri dari lapisan konvolusi, batchnormalization, aktivasi ReLU, dan maxpooling. Residual block 1 hingga 4 terdiri dari beberapa lapisan konvolusi yang dilengkapi dengan shortcut connections dan fitur yang dihasilkan akan diproses dibagian klasifikasi. Bagian klasifikasi terdiri dari lapisan GlobalAveragePooling2D untuk meminimalisir terjadinya overfitting dengan cara mengurangi jumlah total parameter, lapisan Dropout dengan tingkat dropout 0.3 digunakan untuk mengurangi terjadinya overfitting pada model dengan cara mematikan 30% node secara acak selama pelatihan, lapisan fully connected dengan 1024 neuron, menggunakan aktivasi ReLU untuk mengganti nilai negatif pada citra dengan nilai 0, dan regularisasi L2 sebesar 0.01 untuk memberikan penalti pada nilai loss apabila terdapat nilai bobot yang besar. Penalti diberikan dengan cara menambahkan nilai regularisasi dengan nilai loss. Lapisan batch normalization untuk mempercepat, meningkatkan pelatihan dan stabilitas model serta lapisan Dense sebagai output layer dengan jumlah neuron yang sesuai dengan jumlah kelas yaitu 8 dan fungsi aktivasi softmax untuk menghasilkan output kelas, dimana setiap unit menghasilkan probabilitas untuk kelas tertentu.

Model ini juga dilakukan pembekuan 20 lapisan terakhir (stage 5) artinya bobot pada 20 lapisan terakhir, tetap mempertahankan bobot yang sudah dipelajari dari dataset ImageNet dan tidak akan diperbarui selama proses pelatihan serta yang dilakukan training model hanya lapisan-lapisan baru yang sudah ditambahkan (lapisan akhir model) sehingga bobotnya akan diubah selama pelatihan.

Model ini dioptimalkan menggunakan optimizer Adam dengan learning rate 1×10^(-5). Proses pelatihan dilakukan selama 60 epoch dengan menggunakan teknik callback yaitu EarlyStopping untuk meminimalisir overfitting dan ModelCheckpoint untuk menyimpan model terbaik selama pelatihan. Ringkasan model dapat dilihat pada Tabel 2

![Gambar 2](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture7.png)

*Tabel 2: Hasil Arsitektur Model Residual Network 50*

ResNet50 memiliki 180 lapisan yang terdiri dari 1 lapisan Input, 3 lapisan ZeroPadding2D, 52 lapisan Conv2D, 16 lapisan BatchNormalization, 52 lapisan Activation, 1 lapisan MaxPooling2D, 8 lapisan Add, 1 lapisan GlobalAveragePooling2D, 1 lapisan Dropout, dan 2 lapisan Dense.

Model ini memiliki total 59121562 parameter dengan 16711688 parameter yang dapat dilatih selama pelatihan untuk menyesuaikan bobot dan bias model dengan ukuran 63.75 Megabyte dan 8986496 parameter yang tidak dapat dilatih (dari bobot pra-latih) dengan ukuran 34.28 Megabyte dan 33423378 parameter optimizer untuk mengatur pembaruan bobot dan meminimalkan fungsi kerugian dengan ukuran 127.50 Megabyte. Secara keseluruhan, model ini memiliki kapasitas besar dalam memproses data yang kompleks.

## Evaluasi Model Learning Curve
Setelah membuat model CNN dan menentukan nilai masing-masing hyperparameter, model akan dilakukan proses training dan akan diperoleh nilai loss dan accuracy pada output yang berupa Learning Curve dapat dilihat pada gambar berikut

![Gambar 3](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture4.png)

*Gambar 3: Grafik Train dan Test Accuracy*

Gambar 3 merupakan grafik performa akurasi pada data train dan data test saat pelatihan berlangsung selama 60 epoch. Sumbu horizontal (x) menunjukkan jumlah epoch yaitu iterasi pelatihan model, sedangkan sumbu vertikal (y) menunjukkan nilai akurasi yang berkisar antara 0.0 hingga 1.0. Kurva biru (train accuracy) menunjukkan akurasi model pada data pelatihan, sedangkan kurva orange (test accuracy) menunjukkan akurasi model pada data uji. 

Berdasarkan Gambar 3 terlihat bahwa nilai akurasi pada data pelatihan (garis biru) terus meningkat seiring bertambahnya jumlah epoch. Hal ini menunjukkan bahwa model dapat mempelajari pola-pola pada data latih dengan baik tetapi nilai akurasi pada data uji (garis oranye) terlihat tidak stabil, naik turun dengan pola yang cukup tajam, cenderung lebih rendah dibandingkan dengan nilai akurasi pada data latih. Pola ini mengindikasikan bahwa model mengalami overfitting yaitu kondisi dimana model terlalu menghafal data latih sehingga kurang mampu bekerja dengan baik pada data data uji.

![Gambar 4](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture5.png)

*Gambar 4: Grafik Train dan Test Loss*

Gambar 4 merupakan grafik performa loss data train dan test saat pelatihan berlangsung selama 60 epoch. Sumbu horizontal (x) menunjukkan jumlah epoch yaitu iterasi pelatihan model, sedangkan sumbu vertikal (y) menunjukkan nilai loss yang berkisar antara 20.0 hingga 2.5 menggambarkan tingkat kesalahan model dalam memprediksi data. Semakin rendah nilai loss, semakin baik performa model. Kurva biru (train loss) menunjukkan loss model pada data pelatihan, sedangkan kurva orange (test loss) menunjukkan loss model pada data uji yaitu data yang tidak digunakan untuk melatih model tetapi digunakan untuk mengevaluasi kinerja model. 

Berdasarkan Gambar 4 terlihat bahwa nilai loss pada data pelatihan (kurva biru) terus menurun secara konsisten seiring bertambahnya jumlah epoch yang artinya model semakin baik dalam mempelajari pola-pola dari data latih sehingga kesalahan prediksi pada data latih semakin kecil tetapi nilai loss pada data uji (kurva orange) terlihat naik turun selama pelatihan model. Pada beberapa epoch terutama di awal pelatihan, nilai loss pada data uji lebih tinggi daripada nilai loss pada data pelatihan yang artinya model belum sepenuhnya optimal. Meski begitu, seiring bertambahnya epoch, nilai loss pada data uji cenderung menurun tetapi fluktuasinya (naik turun) masih sering terjadi. Hal ini menunjukkan bahwa model kesulitan mengenali pola pada data uji sehingga hasilnya tidak konsisten. Fluktuasi nilai loss tersebut juga mengindikasikan bahwa model terlalu fokus pada data pelatihan (overfitting), sehingga performa pada data uji belum maksimal. 


## Evaluasi Model Confusion Matrix
![Gambar5](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture3.png)

*Gambar 5: Confusion Matrix.*

Model CNN ResNet50 menunjukkan performa terbaik pada kelas NV dengan jumlah prediksi benar paling tinggi (2411 data), diikuti oleh kelas MEL (488) dan BCC (467). Hal ini menunjukkan bahwa model sangat efektif dalam mengenali kelas-kelas tersebut. Sebaliknya, model masih sering mengalami kesalahan klasifikasi pada kelas AK, BKL, dan SCC. Kelas AK banyak salah diprediksi sebagai BKL dan BCC, sementara BKL sering tertukar dengan NV dan MEL. Kelas SCC memiliki jumlah prediksi benar yang relatif rendah dan sering keliru diprediksi sebagai BCC dan BKL. 

Secara keseluruhan, Confusion Matrix menunjukkan bahwa model CNN ResNet50 dapat mengklasifikasikan kelas-kelas dominan seperti NV, MEL, dan BCC dengan baik, tetapi masih memerlukan peningkatan performa pada kelas-kelas yang memiliki karakteristik visual mirip dan jumlah data yang lebih sedikit, khususnya AK, BKL, dan SCC.

## Evaluasi Model Classification Report

| Kelas   | *precission* | *recall* | *f1-score* |
|---------|--------------|----------| ---------- |
| AK      | 53%          | 46%      | 49%        |
| BCC     | 77%          | 70%      | 73%        |
| BKL     | 54%          | 66%      | 60%        |
| DF      | 77%          | 50%      | 61%        |
| MEL     | 79%          | 54%      | 64%        |
| NV      | 82%          | 94%      | 88%        |
| SCC     | 71%          | 35%      | 47%        |
| VASC    | 92%          | 65%      | 76%        |
| Akurasi |              |          | 77%        |

*Tabel 3: Classification Report*


Evaluasi performa model CNN dilakukan menggunakan Classification Report yang mencakup metrik akurasi, precision, recall, dan f1-score untuk setiap kelas. Model mencapai akurasi keseluruhan sebesar 77% yang menunjukkan bahwa sebagian besar prediksi sesuai dengan data aktual. 

Berdasarkan evaluasi per kelas, model menunjukkan performa terbaik pada kelas NV, dengan nilai precision 82%, recall 94%, dan f1-score 88%, menandakan kemampuan deteksi yang sangat baik. Kelas BCC dan VASC juga memiliki performa yang cukup baik dengan f1-score masing-masing sebesar 73% dan 76%. Sebaliknya, model masih mengalami kesulitan dalam mengenali kelas AK dan SCC, yang ditunjukkan oleh nilai f1-score rendah, masing-masing 49% dan 47%. Kelas BKL, DF, dan MEL memiliki performa sedang dengan f1-score di kisaran 60%. 

Secara keseluruhan, model telah mampu mendeteksi beberapa jenis kanker kulit dengan baik, khususnya NV, BCC, dan VASC, namun masih memerlukan peningkatan performa untuk kelas-kelas yang sulit dideteksi seperti AK dan SCC.

## Deployment Model ke Aplikasi Web
Aplikasi web ini merupakan hasil deployment dari model Convolutional Neural Network (CNN) yang telah dilatih. Deployment trained model adalah proses mengintegrasikan model yang sudah ditraining ke dalam aplikasi agar dapat digunakan langsung oleh pengguna melalui antarmuka web. Model CNN yang telah dilatih disimpan dalam format .h5 (HDF5) menggunakan fungsi model.save(), yang mencakup arsitektur dan bobot model. Pada saat aplikasi dijalankan, model dimuat kembali menggunakan load_model() untuk melakukan prediksi tanpa perlu training ulang. Aplikasi web dibuat menggunakan HTML dan CSS sebagai front-end, serta Flask (Python) sebagai back-end. Flask berfungsi sebagai server yang menerima input gambar dari pengguna, memproses data, menjalankan prediksi menggunakan model CNN, dan mengirimkan hasil deteksi ke front-end. 

Pengguna dapat mengunggah citra dermoskopi melalui halaman utama. Gambar yang diunggah akan disimpan di server, kemudian diproses dengan mengubah ukuran menjadi 224×224 piksel dan dinormalisasi ke rentang [0,1] sesuai dengan kebutuhan input model ResNet50. Selanjutnya, gambar diprediksi menggunakan model.predict(). Hasil prediksi berupa jenis kanker kulit dan tingkat probabilitas ditampilkan pada halaman hasil deteksi. Aplikasi berjalan secara lokal melalui server Flask pada alamat http://127.0.0.1:5000.

![Gambar6](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture1.png)

*Gambar 6: Halaman Utama Website*

Halaman utama website dibuat sederhana agar pengguna mudah melakukan deteksi kanker kulit. Di halaman ini ditampilkan nama website “Deteksi Penyakit Kanker Kulit” serta tombol “Unggah Gambar” untuk memilih gambar kulit yang akan diperiksa. Setelah gambar diunggah, pengguna dapat menekan tombol “Kirim” untuk memulai proses deteksi.

![Gambar7](https://github.com/choirunnish/kankerkulit-resnet50/blob/master/assets/Picture2.png)

*Gambar 7: Halaman Hasil Prediksi Gambar*

Hasil deteksi akan ditampilkan pada halaman berikutnya yang berisi informasi jenis kanker kulit yang terdeteksi, tingkat keparahan, dan persentase kepercayaan prediksi. Tersedia juga tombol “Kembali ke Beranda” untuk mengunggah gambar lain atau kembali ke halaman utama. Secara keseluruhan, tampilan website dirancang sederhana dan mudah dipahami oleh pengguna.

## Kesimpulan
Berikut adalah kesimpulan dari hasil penelitian ini:
- Pemodelan yang terbentuk adalah model CNN ResNet50 yang terdiri dari 152 lapisan yaitu 1 lapisan Input, 3 lapisan ZeroPadding2D, 52 lapisan Conv2D, 16 lapisan BatchNormalization, 52 lapisan Activation, 1 lapisan MaxPooling2D, 8 lapisan Add, 1 lapisan GlobalAveragePooling2D, 1 lapisan Dropout, dan 2 lapisan Dense.
- Berdasarkan Learning Curve, model menunjukkan akurasi meningkat dan loss menurun pada data latih tetapi terjadi fluktuasi pada data uji yang artinya model overfitting. Berdasarkan Confusion Matrix, model menunjukkan performa yang baik pada kelas BCC, MEL, dan NV tetapi masih perlu perbaikan pada kelas AK, SCC, dan BKL yang banyak kesalahan klasifikasi. Berdasarkan Classification Report, model ini diperoleh nilai akurasi keseluruhan sebesar 77%.
- Aplikasi website ini dibuat menggunakan framework Flask dengan Python untuk back-end dan HTML serta CSS untuk front-end. Model yang dilatih disimpan dalam format file .h5 untuk memproses gambar yang diunggah pengguna kemudian menampilkan hasil deteksi jenis kanker kulit dan nilai akurasi. 
