# Laporan Proyek Machine Learning - Farah Dina

## Domain Proyek

Latar Belakang
  Industri e-commerce di sektor skincare dan kecantikan mengalami pertumbuhan pesat seiring dengan meningkatnya kesadaran konsumen terhadap perawatan diri. Menurut laporan McKinsey & Company (2023), pasar kecantikan global diproyeksikan tumbuh sebesar 8% per tahun hingga 2030. Di Indonesia, pertumbuhan ini didukung oleh peningkatan jumlah pengguna internet dan perubahan perilaku belanja masyarakat yang semakin beralih ke platform online.

  Salah satu tantangan utama dalam industri ini adalah menentukan strategi diskon dan promosi yang optimal tanpa mengorbankan profitabilitas. Analisis data e-commerce dapat membantu mengidentifikasi pola penjualan dan mengembangkan strategi yang lebih efektif. Penerapan machine learning dalam analisis data pelanggan telah terbukti meningkatkan pengalaman pengguna dan mendukung pengambilan keputusan berbasis data di platform e-commerce. Misalnya, algoritma clustering dan klasifikasi digunakan untuk memahami pola perilaku pelanggan, sementara reinforcement learning diterapkan untuk meningkatkan adaptasi sistem rekomendasi (Naufal,2025).

  Selain itu, penerapan kecerdasan buatan (AI) dalam sistem informasi e-commerce, seperti personalisasi, chatbot, analisis data perilaku, dan sistem rekomendasi, berperan penting dalam menciptakan pengalaman pengguna yang lebih interaktif, efisien, dan memuaskan (Suparman, 2025). Dengan meningkatnya persaingan di pasar e-commerce, perusahaan perlu memahami faktor-faktor yang mempengaruhi penjualan produk mereka untuk tetap kompetitif.

  Penelitian menunjukkan bahwa penerapan machine learning dalam analisis data e-commerce dapat meningkatkan prediksi penjualan dan membantu dalam pengambilan keputusan strategis. Misalnya, model prediksi penjualan menggunakan pendekatan decision tree classification telah diterapkan pada data e-commerce Indonesia untuk memprediksi jumlah item yang terjual berdasarkan jumlah penonton, harga, dan jenis produk (Andry, 2021). Dengan demikian, penting bagi perusahaan untuk mengadopsi pendekatan berbasis data dan teknologi dalam strategi bisnis mereka guna mengoptimalkan penjualan dan profitabilitas.

Mengapa Masalah Ini Harus Diselesaikan:
  - Mengetahui faktor-faktor yang berkontribusi pada penjualan sangat penting bagi perusahaan agar dapat:
  - Meningkatkan efektivitas strategi pemasaran.
  - Menyesuaikan kebijakan harga dan diskon.
  - Mengoptimalkan profitabilitas tanpa kehilangan pelanggan.

Referensi
  McKinsey & Company (2023). "The Future of Beauty Industry Growth." . [https://www.mckinsey.com/industries/consumer-packaged-goods/our-insights/the-future-of-beauty-industry-growth]

  Naufal Hafizh (2025). "Implementasi Machine Learning dalam Analisis Data Pelanggan: Studi Kasus pada Platform E-commerce." [https://www.academia.edu/127168115/Implementasi_Machine_Learning_dalam_Analisis_Data_Pelanggan_Studi_Kasus_pada_Platform_E_commerce]

  Ade Suparman (2025). "Penerapan Kecerdasan Buatan dalam Sistem Informasi untuk Meningkatkan Pengalaman Pengguna pada Aplikasi E-Commerce."[https://ojs.cahayamandalika.com/index.php/jml/article/view/3939]

  Raden Johannes, Andry Alamsyah (2021). "Sales Prediction Model Using Classification Decision Tree Approach For Small Medium Enterprise Based on Indonesian E-Commerce Data."[https://arxiv.org/abs/2103.03117] 

## Business Understanding

### Problem Statements
- Bagaimana pengaruh diskon terhadap volume penjualan dan profitabilitas?
- Faktor apa saja yang paling berpengaruh terhadap penjualan produk?
- Bagaimana model machine learning dapat digunakan untuk memprediksi jumlah penjualan?
- Bagaimana strategi harga dan diskon yang optimal untuk meningkatkan profitabilitas?

### Goals
  - Menganalisis dampak diskon terhadap penjualan dan keuntungan. Mengidentifikasi sejauh mana diskon meningkatkan jumlah penjualan dan bagaimana pengaruhnya terhadap profitabilitas.
  - Menentukan faktor utama yang berkontribusi terhadap penjualan. Melakukan eksplorasi data dan analisis statistik untuk mengidentifikasi faktor-faktor terpenting dalam meningkatkan penjualan.
  - Mengembangkan model prediksi penjualan berbasis machine learning. Menggunakan model regresi untuk memprediksi jumlah penjualan berdasarkan berbagai faktor seperti harga, diskon, dan kategori produk.
  - Menyusun strategi harga dan diskon yang optimal. Menggunakan hasil model prediksi untuk memberikan rekomendasi strategi pricing yang efektif.


    ### Solution statements
      - Untuk mencapai tujuan tersebut, solusi yang diterapkan adalah:
      - Eksplorasi Data: Melakukan analisis eksploratif terhadap distribusi data dan hubungan antar variabel.
      - Feature Engineering: Menambahkan fitur baru seperti "Discount Impact" dan "Profit Margin" untuk meningkatkan performa model prediksi.
      - Modeling: Menggunakan empat algoritma regresi yang berbeda: K-Nearest Neighbors, Random Forest, Gradient Boosting, dan XGBoost.
      - Evaluasi Model: Membandingkan performa model menggunakan metrik MAE, R2 Score, dan RMSE untuk menentukan model terbaik.

## Data Understanding
Dataset yang digunakan dalam analisis ini berasal dari data e-commerce sektor skincare dan kecantikan. Data ini mencakup berbagai variabel yang berhubungan dengan transaksi penjualan, harga, diskon, kategori produk, dan perilaku konsumen. Dataset ini memiliki 51.290 baris dan 19 kolom, dengan tidak ada missing values di setiap kolom. Tipe data terdiri dari 3 kolom numerik (int64), 4 kolom desimal (float64), dan 12 kolom kategorikal (object). Kolom numerik mencakup Quantity, Sales, Discount, dan Profit, sementara kolom kategorikal mencakup Order ID, Customer ID, Segment, City, State, dan lainnya.Tidak ada missing values dalam dataset ini. Statistik deskriptif menunjukkan bahwa rata-rata jumlah barang yang dipesan adalah 5,41 dengan maksimum 20 unit per transaksi, serta harga penjualan berkisar antara 2 hingga 3.940. Diskon bervariasi dari 0% hingga 85%, sementara profit berkisar dari -1.746 (rugi) hingga 1.820 (untung). Datadiperoleh dari kaggle [https://www.kaggle.com/datasets/shandeep777/e-commerce-analysis-global-skincare-e-store] 

### Variabel-variabel pada dataset adalah sebagai berikut:
- Row ID: Identifikasi unik untuk setiap baris dalam dataset.
- Order ID: Nomor unik untuk setiap pesanan.
- Order Date: Tanggal pesanan dibuat.
- Customer ID: Identifikasi unik untuk setiap pelanggan.
- Segment: Kategori pelanggan berdasarkan profil pembelian.
- City: Kota tempat pelanggan melakukan pembelian.
- State: Negara bagian atau provinsi dari lokasi pelanggan.
- Country: Negara tempat pelanggan melakukan pembelian.
- Country Latitude: Koordinat garis lintang negara pelanggan.
- Country Longitude: Koordinat garis bujur negara pelanggan.
- Region: Wilayah geografis pelanggan.
- Market: Pasar atau wilayah pemasaran tempat produk dijual.
- Subcategory: Kategori kecil dari produk yang dijual.
- Category: Kategori utama produk.
- Product: Nama atau jenis produk yang dijual.
- Quantity: Jumlah produk yang dibeli dalam satu pesanan.
- Sales: Jumlah penjualan produk.
- Discount: Persentase diskon yang diberikan pada produk.
- Profit: Keuntungan yang diperoleh dari penjualan produk.

## Data Preparation
Tahapan preprocessing data yang dilakukan:
1. Menghapus Kolom yang Tidak Relevan
Langkah pertama adalah menghapus kolom yang tidak memiliki pengaruh langsung terhadap analisis prediktif. Kolom seperti Row ID, Order ID, Order Date, Customer ID, City, State, dan Product dihapus karena hanya berupa informasi administratif atau data unik yang tidak berkontribusi pada prediksi.
2. Mengonversi Data ke Format Numerik
Model machine learning hanya dapat bekerja dengan data numerik, sehingga semua kolom kategori harus dikonversi. Untuk kolom kategori dengan sedikit nilai unik, seperti Segment, digunakan Label Encoding yang mengubah kategori menjadi angka. Untuk kolom kategori dengan banyak nilai unik, seperti Country, Category, Subcategory, Region, dan Market, digunakan One-Hot Encoding, yang mengubah setiap kategori menjadi kolom biner (0 atau 1).
3. Menghapus Outlier dengan IQR (Interquartile Range)
Outlier adalah nilai yang terlalu jauh dari data mayoritas, yang bisa menyebabkan model membuat prediksi yang tidak akurat. Identifikasi outlier dilakukan dengan Interquartile Range (IQR), yang menghitung rentang antara Q1 (kuartil pertama) dan Q3 (kuartil ketiga). Data yang berada di luar batas Q1 - 1.5 * IQR atau Q3 + 1.5 * IQR dianggap sebagai outlier dan dihapus.
4. Menangani Missing Values
Nilai yang hilang dalam dataset diisi dengan median dari setiap kolom. digunakan median karena median lebih tahan terhadap outlier, sehingga memberikan hasil yang lebih stabil dibandingkan mean.
5. Feature Engineering
Feature Engineering adalah proses menambahkan fitur baru untuk meningkatkan analisis dan prediksi.
Dua fitur baru dibuat:
- Discount Impact – Menghitung dampak diskon terhadap penjualan dengan mengalikan nilai Discount dengan Sales.
- Profit Margin – Mengukur efisiensi keuntungan dengan menghitung rasio antara Profit dan Sales.
Fitur tambahan ini dapat memberikan wawasan lebih dalam tentang hubungan antara diskon, penjualan, dan profitabilitas.
6. Membagi Data Menjadi Training dan Testing Set
Dataset dibagi menjadi 80% untuk training dan 20% untuk testing. Pembagian ini dilakukan agar model dapat belajar dari sebagian besar data dan kemudian dievaluasi pada data yang belum pernah dilihat sebelumnya. Data test digunakan untuk mengukur performa model sebelum diterapkan ke data dunia nyata.
7. Menangani Nilai Tak Terhingga (Inf) dan NaN
Setelah pembagian data, perlu dicek apakah masih ada nilai Inf (tak terhingga) atau NaN (kosong). Nilai Inf dan -Inf diubah menjadi NaN, lalu NaN diisi dengan nol untuk menghindari error dalam model machine learning.
8. Standardisasi Data
Data distandarisasi menggunakan StandardScaler, sehingga semua fitur memiliki skala yang sama. standardisasi diperlukan karena beberapa algoritma machine learning (seperti regresi linier, SVM, dan k-NN) sensitif terhadap skala data. Standardisasi memastikan bahwa fitur dengan skala besar tidak mendominasi fitur dengan skala keci.
9. Reduksi Dimensi dengan PCA (Jika Jumlah Fitur Lebih dari 1)
Jika dataset memiliki banyak fitur, dilakukan Principal Component Analysis (PCA) untuk mengurangi jumlah fitur tanpa kehilangan terlalu banyak informasi. PCA mempertahankan 95% varians data, sehingga hanya fitur yang paling penting yang digunakan. Reduksi dimensi ini dapat membantu mempercepat proses training model dan menghindari overfitting.
  0. Seleksi Fitur dengan SelectKBest
Untuk memilih fitur yang paling berpengaruh terhadap prediksi Sales, digunakan metode SelectKBest dengan f_regression. Hanya 50 fitur terbaik yang dipilih berdasarkan tingkat korelasi terhadap target (Sales). Ini membantu mengurangi kompleksitas model dan meningkatkan akurasi prediksi.

## Modeling
  Model yang Digunakan
  Beberapa algoritma regresi digunakan untuk memprediksi jumlah penjualan:
  - K-Nearest Neighbors (KNN)
    cara kerja :Model non-parametrik yang mencari tetangga terdekat untuk prediksi. Model ini bekerja dengan mencari beberapa data terdekat yang sudah diketahui hasilnya, 
    lalu mengambil rata-ratanya sebagai prediksi.
    Kelebihan: Mudah dipahami dan efektif untuk dataset kecil.
    Kekurangan: Kurang efisien pada dataset besar dan sensitif terhadap outlier.
    parameter yang digunakan : n_neighbors = 2 : Jumlah tetangga terdekat yang digunakan untuk prediksi.

 - Random Forest
   cara kerja : Model berbasis pohon keputusan yang kuat terhadap overfitting. Model ini bekerja dengan membuat banyak pohon keputusan, lalu menggabungkan hasilnya untuk mendapatkan prediksi yang lebih akurat.
   Kelebihan: Dapat menangani dataset besar dan tidak terlalu terpengaruh oleh outlier.
   Kekurangan: Cenderung lebih lambat dibanding model lain dan dapat menjadi kompleks.
   parameter yang digunakan : n_estimators=100 : Jumlah pohon dalam ensemble (semakin banyak pohon, semakin stabil hasilnya).
                              max_depth=10 : Kedalaman maksimum setiap pohon (membantu mengontrol kompleksitas model).
                              random_state=42 : Menetapkan seed untuk hasil yang konsisten setiap kali model dijalankan.

  - Gradient Boosting
    cara kerja : Model boosting yang memperbaiki kesalahan prediksi secara iteratif.Model ini juga menggunakan banyak pohon keputusan, tapi dengan cara yang berbeda dari Random Forest. Setiap pohon belajar dari kesalahan pohon sebelumnya, jadi makin lama semakin bagus.
    Kelebihan: Memberikan akurasi tinggi dengan meminimalkan error secara bertahap.
    Kekurangan: Membutuhkan tuning parameter yang cermat agar tidak overfitting.
    parameter yang digunakan :  n_estimators=200 : Jumlah pohon dalam boosting (semakin banyak, semakin kompleks modelnya).
                                learning_rate=0.1 : Kecepatan pembelajaran yang mengontrol seberapa besar kontribusi setiap pohon baru.
                                max_depth=5 : Kedalaman maksimum setiap pohon (mengontrol kapasitas model).
                                random_state=42 : Seed untuk memastikan hasil yang sama setiap kali model dijalankan.
    
  - XGBoost
    cara kerja : Algoritma boosting yang lebih cepat dan optimal dibanding Gradient Boosting. Model ini bekerja dengan membuat banyak pohon keputusan, lalu menggabungkan hasilnya untuk mendapatkan prediksi yang lebih akurat.
    Kelebihan: Performa tinggi dan lebih efisien dari segi waktu.
    Kekurangan: Memerlukan lebih banyak sumber daya komputasi.
    parameter yang digunakan :  n_estimators=200 : Jumlah pohon dalam ensemble.
                                learning_rate=0.1 : Kecepatan pembelajaran untuk mengontrol penyesuaian model pada setiap iterasi.
                                max_depth=5 : Kedalaman maksimum setiap pohon untuk mengontrol kompleksitas model.
                                random_state=42 : Seed untuk hasil yang konsisten.


  Pemilihan Model Terbaik
  Model terbaik dipilih berdasarkan metrik evaluasi MAE, R2 Score, dan RMSE. Model dengan R2 Score tertinggi dianggap sebagai model terbaik karena mampu menjelaskan variasi data dengan lebih baik.

## Evaluation
Model dievaluasi menggunakan tiga metrik utama:
- Mean Absolute Error (MAE): Rata-rata perbedaan absolut antara prediksi dan nilai aktual. Semakin kecil nilainya, semakin baik model dalam memprediksi.
- R2 Score: Mengukur seberapa baik model menjelaskan variabilitas data. Nilai mendekati 1 menunjukkan model yang lebih baik.
- Root Mean Squared Error (RMSE): Menghitung akar dari rata-rata kesalahan kuadrat. RMSE lebih sensitif terhadap outlier dibanding MAE.

Hasil evaluasi:
- K-Nearest Neighbors - MAE: 18.9791
  K-Nearest Neighbors - R2 Score: 0.6237
  K-Nearest Neighbors - RMSE: 31.0963

- Random Forest - MAE: 18.9110
  Random Forest - R2 Score: 0.7004
  Random Forest - RMSE: 27.7484

- Gradient Boosting - MAE: 13.5467
  Gradient Boosting - R2 Score: 0.8268
  Gradient Boosting - RMSE: 21.0972

- XGBoost - MAE: 13.1463
  XGBoost - R2 Score: 0.8313
  XGBoost - RMSE: 20.8217


Dari hasil evaluasi, dapat disimpulkan bahwa:
Random Forest memiliki R2 Score 0.7004, MAE 18.9110, dan RMSE 27.7484, yang masih lebih baik dibandingkan K-Nearest Neighbors, tetapi kurang presisi dibandingkan Gradient Boosting dan XGBoost.
K-Nearest Neighbors adalah model dengan performa terburuk, dengan R2 Score 0.6237, MAE 18.9791, dan RMSE tertinggi (31.0963). Selain itu, Train MSE (267.5) jauh lebih rendah dibanding Test MSE (966.9), yang menunjukkan kemungkinan model ini mengalami overfitting, sehingga tidak mampu menangkap pola dalam data secara efektif.
Berdasarkan hasil ini, XGBoost dipilih sebagai model terbaik untuk prediksi karena memiliki R2 Score tertinggi dan kesalahan prediksi paling rendah dibandingkan model lainnya.

Berdasarkan hasil evaluasi model, berikut adalah analisis keberhasilan model dalam mencapai setiap goals yang telah ditetapkan:

1. Menganalisis dampak diskon terhadap penjualan dan keuntungan. Discount Impact dan Profit Margin membantu memahami dampak diskon. Discount vs Sales Analysis menunjukkan adanya peningkatan penjualan ketika diskon diberikan, tetapi juga ada kemungkinan margin keuntungan berkurang. Model memprediksi perubahan jumlah penjualan berdasarkan variasi diskon.	Model dapat menunjukkan efek diskon terhadap penjualan dan keuntungan, membantu dalam strategi promosi yang lebih efektif.
   
2. Menentukan faktor utama yang berkontribusi terhadap penjualan. Feature selection dan heatmap menunjukkan faktor penting seperti diskon, kategori produk, dan harga. Feature Importance dari XGBoost & Gradient Boosting menunjukkan bahwa diskon, harga, dan kategori produk adalah faktor paling berpengaruh terhadap penjualan. Heatmap korelasi menunjukkan hubungan kuat antara harga, diskon, dan total penjualan.	Model membantu memahami faktor-faktor utama yang memengaruhi penjualan, memungkinkan bisnis untuk fokus pada variabel yang paling berpengaruh.
   
3. Mengembangkan model prediksi penjualan berbasis machine learning. Model XGBoost memiliki akurasi tertinggi dengan R2 Score 0.8313 dan RMSE terendah (20.8217). XGBoost outperform model lain dengan Train MSE 255.7987 dan Test MSE 433.5414. R2 Score tertinggi (0.8313) menunjukkan model mampu menjelaskan variabilitas data dengan baik.	Model dapat digunakan untuk memprediksi jumlah penjualan dengan baik, mendukung perencanaan stok dan strategi pemasaran yang lebih akurat.
   
4. Menyusun strategi harga dan diskon yang optimal. Model memberikan wawasan, tetapi keputusan bisnis tetap diperlukan. Analisis profitabilitas menunjukkan bahwa diskon dapat meningkatkan penjualan tetapi belum tentu meningkatkan profit. Tetapi model hanya memberikan prediksi, tetapi keputusan akhir mengenai strategi harga dan diskon masih bergantung pada kebijakan perusahaan.	Model memberikan gambaran dan data pendukung, tetapi strategi akhir tetap memerlukan analisis tambahan dari sisi bisnis dan market.
