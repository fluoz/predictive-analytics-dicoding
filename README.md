# Laporan Proyek Machine Learning - Hasan Abdullah Munshi

## Domain Proyek
#### Latar Belakang
_Cryptocurrency_ adalah mata uang digital atau virtual yang dijamin dengan kriptografi, yang membuatnya hampir tidak mungkin untuk dipalsukan atau digandakan. Banyak _cryptocurrency_ adalah jaringan terdesentralisasi berdasarkan teknologi blockchain (buku besar terdistribusi yang diberlakukan oleh jaringan komputer yang berbeda). Ciri khas dari _cryptocurrency_ adalah bahwa mereka umumnya tidak dikeluarkan oleh otoritas pusat, membuat mereka secara teoritis kebal terhadap campur tangan atau manipulasi pemerintah. _Crypto_ sendiri "mengacu pada berbagai algoritma enkripsi dan teknik kriptografi yang melindungi entri ini, seperti enkripsi kurva elips, public-private key pairs, dan fungsi hashing.

_Cryptocurrency_ berbasis blockchain pertama adalah _Bitcoin_, yang masih menjadi yang paling populer dan paling berharga. Saat ini, terdapat ribuan mata uang kripto alternatif dengan berbagai fungsi dan spesifikasi. Beberapa mata uang kripto yang bersaing yang lahir dari kesuksesan Bitcoin, yang dikenal sebagai _"altcoin"_, termasuk Litecoin, Peercoin, dan Namecoin, serta Ethereum, Cardano, dan EOS. Hari ini, nilai keseluruhan dari semua _cryptocurrency_ yang ada adalah sekitar $ 214 miliar — Bitcoin saat ini mewakili lebih dari 68% dari total nilai.

_Cryptocurrency_ menjanjikan kemudahan dalam mentransfer dana secara langsung antara dua pihak, tanpa perlu pihak ketiga yang terpercaya seperti bank atau perusahaan kartu kredit. Kerugiannya, sifat semi-anonim transaksi _cryptocurrency_ membuatnya sangat cocok untuk sejumlah kegiatan ilegal, seperti pencucian uang dan penggelapan pajak. Namun, pendukung _cryptocurrency_ seringkali sangat menghargai anonimitas mereka, mengutip manfaat privasi seperti perlindungan bagi pelapor atau aktivis yang hidup di bawah pemerintahan yang represif.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, berikut ini batasan masalah yang dapat diselesaikan dengan proyek ini:
- Bagaimana cara membuat model prediksi berdasarkan dari 4 algoritma yaitu KNN, Random Forest, Adaboost, dan SVR?
- Bagaimana cara membangun model yang dapat memprediksi time series dengan baik?
- Bagaimana cara membangun model dengan akurasi yang baik?

### Goals

- Agar bisa memprediksi harga bitcoin seakurat mungkin
- Mengetahui perbandingan algoritma model _machine learning_ dari KNN, Random Forest, AdaBoost, dan SVR
- Mendapatkan algoritma yang dapat akurasi yang baik

### Solution Statement

1. Menangani _missing value_ pada data
2. Menangani _outliners_ menggunakan IQR Method
3. Membandingkan hasil modelling dari 4 algoritma KNN, Random Forest, AdaBoost, dan SVR

## Data Understanding
Dataset yang digunakan pada proyek ini adalah dataset _Bitcoin_ yang didapatkan melalui _API_ python-binance. dengan jarak interval 1 Hari dan dimulai pada tanggal 01 Januari 2017.

### Variabel-variabel pada Bitcoin dataset adalah sebagai berikut:
- Date : merupakan tanggal dibuatnya data.
- Open : merupakan harga pembukaan pada periode/tanggal tertentu.
- High : merupakan harga tertinggi pada periode/tanggal tertentu.
- Low : merupakan harga terendah pada periode/tanggal tertentu.
- Close : merupakan harga penutupan pada periode/tanggal tertentu. 
- Volume : merupakan jumlah trading aktif pada periode/tanggal tertentu. 

### Exploratory Data Analysis
- Menangani missing value, karena data berjumlah 1927 dan non null disemua kolom berjumlah 1927 juga jadi tidak ada missing value
- Menangani Outliners dengan metode IQR Method yaitu membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3.

##### Visualisasi fitur numerik

![numerik](https://drive.google.com/uc?id=1I98FxSPDUZ-eduAu9A15ck3DPgdjwVsq)
Gambar 1. Visualisasi chart pada fitur numerik

Mari amati Gambar 1 histogram di atas, khususnya histogram untuk variabel "Close" yang merupakan fitur target (label) pada data kita. Dari histogram "Close", kita bisa memperoleh beberapa informasi, antara lain:

- Peningkatan harga _Bitcoin_ sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari histogram "Close" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu x).


##### Visualisasi hubungan antar fitur numerik dan fungsi pairplot
![hubungan_pairplot](https://drive.google.com/uc?id=1S9pzzf86S0OQCuQj5Egb1KIKQl3P7c1z)
- Gambar 2. Visualisasi hubungan antar fitur numerik dan fungsi pairplot

Mari amati korelasi fitur "Close" terhadap fitur lain "High", "Open", "Low", "Volume". dapat disimpulkan bahwa "Close" memiliki korelasi dengan "High", "Open", "Low" yang positif kuat. sedangkan korelasi antara "Close" dan "Volume" memiliki korelasi yang sedang.

## Data Preparation
tahap ini melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.
### Menghapus Kolom yang tidak perlu
Menghapus kolom Date dan Volume karena kita tidak membutuhkannya.
### Train Test Split
Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Ketahuilah bahwa setiap transformasi yang kita lakukan pada data juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. kita membaginya 20% untuk test dan 80% untuk train.
### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Kita menggunakan MinMaxScaler karena untuk menyesuaikan data dengan range tertentu.

## Modeling
### K-Nearest Neighbors
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. hyperparameter yang digunakan yaitu:

- _n_neighbors_ : Jumlah tetangga yang akan digunakan secara default untuk kneighbors kueri. Parameter terbaik yaitu _4_.

##### kelebihan
- Mudah diterapkan
- Mudah beradaptasi
- Memiliki sedikit hyperparameter
- Algoritma K-NN kuat dalam mentraining data yang noisy

##### kekurangan
- Tidak berfungsi dengan baik pada dataset berukuran besar
- Kurang cocok untuk dimensi tinggi
- Perlu penskalaan fitur
- Sensitif terhadap noise data, missing values dan outliers

### Random Forest
Random forest adalah suatu algoritma yang digunakan pada klasifikasi data dalam jumlah yang besar. Klasifikasi random forest dilakukan melalui penggabungan pohon dengan melakukan training pada sampel data yang dimiliki. hyperparameter yang digunakan yaitu:

- _bootstrap_ : sampel bootstrap digunakan saat membuat pohon. Parameter terbaik yaitu _True_.
- _max_depth_ : Kedalaman maksimum pohon. Parameter terbaik yaitu _90_.
- _max_features_ : Jumlah fitur yang perlu dipertimbangkan saat mencari pemisahan terbaik. Parameter terbaik yaitu _2_.
- _min_samples_leaf_ : Jumlah minimum sampel yang diperlukan untuk berada di simpul daun. Parameter terbaik yaitu _3_.
- _min_samples_split_ : Jumlah minimum sampel yang diperlukan untuk membagi simpul internal. Parameter terbaik yaitu _8_.
- _n_estimators_ : Jumlah pohon di hutan. Parameter terbaik yaitu _100_.

##### kelebihan
- Mampu menghasilkan prediksi dengan tingkat akurasi tinggi
- Mudah dipahami
- Jika diterapkan pada kumpulan dataset berskala besar, random forest akan bekerja secara efisien


##### kekurangan
- Diperlukan lebih banyak sumber daya dalam proses komputasi
- Makin banyak waktu yang diperlukan untuk bisa memprediksi hasil
- Random Forest cenderung bias saat berhadapan dengan variabel kategorikal

### AdaBoost
AdaBoost adalah meta-estimator yang dimulai dengan memasang regressor pada dataset asli dan kemudian menyesuaikan salinan regressor tambahan pada dataset yang sama tetapi bobot instance disesuaikan menurut kesalahan prediksi saat ini. Dengan demikian, regressor selanjutnya lebih fokus pada kasus-kasus sulit. hyperparameter yang digunakan yaitu:

- _learning_rate_ : Bobot diterapkan pada setiap regressor pada setiap iterasi boosting. Tingkat pembelajaran yang lebih tinggi meningkatkan kontribusi setiap regressor. Parameter terbaik yaitu _0.1_.
- _n_estimators_ : Jumlah maksimum estimator di mana peningkatan dihentikan. Parameter terbaik yaitu _100_.
- _random_state_ : Mengontrol seed acak yang diberikan pada masing _base_estimator_ -masing iterasi boosting. Jadi, ini hanya digunakan saat _base_estimator_ memaparkan file random_state. Parameter terbaik yaitu _77_.

##### kelebihan
- relatif lebih mudah
- Lebih cepat
- Sangat cocok dalam kondisi real time

##### kekurangan
- Membutuhkan hypertuning yang tepat

### Support Vector Regression
Support Vector Regression adalah algoritma supervised learning yang digunakan untuk memprediksi nilai variabel kontinu. SVR menggunakan prinsip yang sama dengan SVM. hyperparameter yang digunakan yaitu:

- _kernel_ : Menentukan jenis kernel yang akan digunakan dalam algoritme. Parameter terbaik yaitu _"rbf"_.
- _C_ : Parameter regularisasi. Kekuatan regularisasi berbanding terbalik dengan C. Harus benar-benar positif. Parameter terbaik yaitu _1000_.
- _gamma_ : Koefisien kernel untuk 'rbf', 'poly' dan 'sigmoid'. Parameter terbaik yaitu _0.3_.

##### kelebihan
- Model keputusan dapat dengan mudah diperbarui
- Dapat menggunakan beberapa classifier yang dilatih pada berbagai jenis data menggunakan aturan probabilitas
- SVR melakukan komputasi yang lebih rendah dibandingkan dengan teknik regresi lainnya

##### kekurangan
- Penggunaannya tidak cocok untuk kumpulan data yang besar
- Jika jumlah fitur untuk setiap titik data melebihi jumlah sampel data pelatihan, SVR berperforma buruk.
- Model keputusan tidak berkinerja sangat baik ketika kumpulan data memiliki lebih banyak noise yaitu kelas target tumpang tindih.

## Evaluation
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

### $$MSE = {1 \over N}\sum_{i = 1}^{N}(y_{1} - ypred_i)^2$$

Keterangan:

N = jumlah dataset

yi = nilai sebenarnya

ypred = nilai prediksi

##### Berikut hasil evaluasi pada proyek ini :


|          |    train_mse   |    test_mse    |
|:--------:|:--------------:|:--------------:|
|    RF    |  127854.807128 |  375559.28038  |
|    KNN   |  160238.994297 |  350222.639866 |
| Boosting | 1569425.731112 | 2114822.604449 |
|    SVR   | 1176678.656931 | 1277824.200873 |
##### Visualisasi Hasil MSE dari 4 Algoritma

![hasil](https://drive.google.com/uc?id=1nN_8O0jD4WL-zJCABwrkNPqaXMaMkqI1)
- Gambar 3. Visualisasi Hasil MSE dari 4 Algoritma

terlihat bahwa model KNN memiliki data test dengan nilai error yang kecil dan model Random Forest memiliki data train yang nilai errornya kecil

Untuk mengujinya, mari kita buat prediksi menggunakan beberapa harga dari data test.

Hasil dari perbandingan 4 algoritma adalah sebagai berikut:

| y_true | prediksi_RF | prediksi_KNN | prediksi_Boosting | prediksi_SVR |
|:------:|:-----------:|:------------:|:-----------------:|:------------:|
| 6529.2 |    6614.7   |    6604.2    |       5664.8      |    6751.7    |

Terlihat bahwa prediksi dengan K-Nearest Neighbors (KNN) memberikan hasil yang paling mendekati.
