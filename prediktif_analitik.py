# -*- coding: utf-8 -*-
"""prediktif_analitik.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TdKeCBz7SKCKm2letAW-vLcWH4sFp7QK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

data_path = '/content/Global skincare and Beauty e-store_E-commerce Analysis_English.csv'
df = pd.read_csv(data_path)

print(df.info())

missing_values = df.isnull().sum()
print(missing_values)

duplicate_rows = df[df.duplicated()]
print(duplicate_rows)

print("Column Names:", df.columns)

print(df.describe())

"""Dataset ini memiliki 51.290 baris dan 19 kolom, dengan tidak ada missing values di setiap kolom. Tipe data terdiri dari 3 kolom numerik (int64), 4 kolom desimal (float64), dan 12 kolom kategorikal (object). Kolom numerik mencakup Quantity, Sales, Discount, dan Profit, sementara kolom kategorikal mencakup Order ID, Customer ID, Segment, City, State, dan lainnya.Tidak ada missing values dalam dataset ini. Statistik deskriptif menunjukkan bahwa rata-rata jumlah barang yang dipesan adalah 5,41 dengan maksimum 20 unit per transaksi, serta harga penjualan berkisar antara 2 hingga 3.940. Diskon bervariasi dari 0% hingga 85%, sementara profit berkisar dari -1.746 (rugi) hingga 1.820 (untung)."""

df.drop(columns=['Row ID', 'Order ID', 'Order Date', 'Customer ID', 'City', 'State', 'Product'], errors='ignore', inplace=True)

label_cols = ['Segment']
label_encoder = LabelEncoder()

for col in label_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

nominal_cols = [ 'Country', 'Category', 'Subcategory', 'Region', 'Market']
existing_nominal_cols = [col for col in nominal_cols if col in df.columns]

if existing_nominal_cols:
    df = pd.get_dummies(df, columns=existing_nominal_cols)

df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

print(df.head())

"""Langkah-langkah ini dilakukan untuk membersihkan dan mengonversi data menjadi format yang lebih siap untuk analisis dan pemodelan machine learning. Pertama, beberapa kolom yang tidak relevan dihapus agar dataset lebih ringkas. Selanjutnya, kolom kategorikal seperti Segment dikonversi ke bentuk numerik menggunakan Label Encoding, sementara kolom nominal lainnya seperti Country, Category, Subcategory, Region, dan Market dikonversi menjadi variabel dummy menggunakan One-Hot Encoding. Setelah itu, semua data dikonversi ke tipe numerik, dan nilai yang hilang (NaN) diisi dengan median dari setiap kolom. Output yang dihasilkan adalah dataset dengan hanya nilai numerik, yang terdiri dari fitur asli seperti Quantity, Sales, Discount, Profit, serta banyak kolom baru yang merepresentasikan kategori dalam bentuk biner (True/False). Hasil akhirnya adalah dataset dengan 8.550 kolom yang siap digunakan untuk analisis atau model machine learning."""

numeric_df = df.select_dtypes(include=['number'])

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

df_clean = df.fillna(df.median(numeric_only=True))
numeric_cols = df_clean.select_dtypes(include=['number'])

plt.figure(figsize=(18, 8))
sns.boxplot(data=numeric_cols, boxprops=dict(facecolor="lightblue"), width=0.6, showfliers=True)

plt.xticks(rotation=45, ha="right")
plt.title("Boxplot Sesudah Penanganan Outlier & Missing Values", fontsize=24)
plt.xlabel("Fitur", fontsize=16)
plt.ylabel("Nilai", fontsize=16)
plt.tight_layout()
plt.show()

"""boxplot membantu mengidentifikasi persebaran data dan keberadaan outlier dalam setiap fitur numerik. Missing values telah diatasi dengan menggantinya menggunakan median, yang membantu mempertahankan kestabilan distribusi data tanpa terpengaruh oleh nilai ekstrem. Sementara itu, outlier tetap terlihat pada beberapa fitur seperti Sales dan Profit, yang menunjukkan adanya nilai ekstrem dalam dataset. Proses ini penting untuk memastikan bahwa analisis atau model machine learning yang dibuat nantinya tidak terganggu oleh data yang tidak wajar atau tidak lengkap, sehingga meningkatkan kualitas hasil analisis dan prediksi."""

plt.figure(figsize=(12, 6))
df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Data Distribution', fontsize=16)
plt.show()

""" Country latitude dan Country longitude memiliki variasi yang cukup luas, menunjukkan cakupan geografis yang beragam. Quantity dan Sales menunjukkan distribusi yang condong ke kanan, mengindikasikan mayoritas transaksi bernilai kecil dengan beberapa transaksi bernilai besar. Discount memiliki distribusi yang tidak merata, dengan sebagian besar nilai di sekitar nol. Sementara itu, Profit menunjukkan distribusi mendekati normal dengan sedikit skew ke kanan, yang berarti sebagian besar keuntungan berkisar di sekitar nol, dengan beberapa transaksi yang menghasilkan keuntungan tinggi maupun kerugian. Analisis distribusi ini membantu dalam mengidentifikasi pola data dan kebutuhan transformasi data sebelum digunakan untuk analisis atau pemodelan lebih lanjut."""

df_numeric = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""Heatmap korelasi ini menunjukkan hubungan antar variabel dalam dataset. Dari visualisasi ini, terlihat bahwa Quantity memiliki korelasi positif yang cukup kuat dengan Sales (0.53), hal ini menandakan semakin banyak jumlah barang yang terjual, semakin tinggi total penjualan. Profit juga memiliki korelasi positif dengan Sales (0.38) dan Quantity (0.24), menandakan bahwa peningkatan penjualan cenderung meningkatkan keuntungan. Namun, Discount memiliki korelasi negatif dengan Profit (-0.44), yang menunjukkan bahwa semakin besar diskon yang diberikan, semakin kecil keuntungan yang diperoleh. Selain itu, korelasi antara variabel geografis seperti Country latitude dan Country longitude dengan variabel lainnya relatif kecil, menunjukkan bahwa lokasi tidak memiliki dampak signifikan terhadap variabel bisnis utama dalam dataset ini. Heatmap ini membantu dalam memahami bagaimana variabel saling berhubungan, yang berguna dalam pengambilan keputusan bisnis dan analisis lebih lanjut."""

plt.figure(figsize=(10, 5))
sns.scatterplot(x=df['Discount'], y=df['Sales'])
plt.title('Discount vs Sales')
plt.xlabel('Discount')
plt.ylabel('Sales')
plt.show()

"""Scatter plot ini menunjukkan hubungan antara Discount dan Sales. Dari visualisasi ini, terlihat bahwa penjualan terjadi pada berbagai tingkat diskon, namun distribusi titik data tidak menunjukkan pola yang jelas. Artinya, peningkatan atau penurunan diskon tidak secara langsung menentukan jumlah penjualan yang lebih tinggi atau lebih rendah. Hal ini selaras dengan heatmap korelasi sebelumnya, yang menunjukkan korelasi rendah antara diskon dan penjualan. Meskipun diskon sering digunakan untuk meningkatkan penjualan, faktor lain seperti permintaan produk, strategi harga, dan loyalitas pelanggan mungkin lebih berpengaruh dalam menentukan total penjualan."""

df.loc[:, 'Discount Impact'] = df['Discount'] * df['Sales']
df.loc[:, 'Profit Margin'] = df['Profit'] / (df['Sales'] + 1e-6)

print("Discount Impact:")
print(df[['Discount', 'Sales', 'Discount Impact']].head(), "\n")

print("Profit Margin:")
print(df[['Profit', 'Sales', 'Profit Margin']].head(), "\n")

"""Dua tabel di atas menunjukkan pengaruh diskon terhadap penjualan (Discount Impact) serta margin keuntungan (Profit Margin). Discount Impact dihitung dari perkalian Discount dan Sales, menunjukkan bahwa diskon tinggi dapat meningkatkan penjualan tetapi berpotensi menurunkan profitabilitas. Analisis ini penting bagi strategi bisnis untuk menyeimbangkan peningkatan penjualan dengan profitabilitas, membantu perusahaan menentukan titik optimal di mana diskon dapat mendorong penjualan tanpa mengorbankan terlalu banyak keuntungan."""

X = df.drop(columns=['Sales'])
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-Test Split:")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\n")

"""code ini untuk pembagian data menjadi dua bagian: X_train (data latih) dan X_test (data uji).

- X_train shape: (28924, 222) : Berarti terdapat 28.924 sampel dengan 222 fitur yang digunakan untuk melatih model.
- X_test shape: (7232, 222) : Berarti terdapat 7.232 sampel dengan 222 fitur yang digunakan untuk menguji performa model.
"""

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
print("Missing values replaced:")
print(X_train.head(), "\n")

"""memastikan bahwa dataset lengkap dan bebas dari nilai NaN, sehingga dapat digunakan secara optimal dalam model machine learning tanpa gangguan akibat nilai yang hilang atau tidak terdefinisi."""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Standardized Features:")
print(X_train[:5], "\n")

"""standarisasi menggunakan StandardScaler, yang menyetarakan skala fitur dengan rata-rata 0 dan standar deviasi 1. Ini penting agar algoritma machine learning bekerja lebih efektif tanpa bias terhadap fitur dengan skala besar, mempercepat konvergensi, dan meningkatkan akurasi prediksi."""

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

if X_train.shape[1] > 1:
    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print("PCA Applied:")
    print(X_train[:1], "\n")

"""code ini menerapkan Principal Component Analysis (PCA) untuk reduksi dimensi jika jumlah fitur pada X_train lebih dari 1. PCA mengurangi jumlah fitur sambil mempertahankan 95% variansi dalam data, sehingga model menjadi lebih efisien dan menghindari overfitting. Dengan menerapkan pca.fit_transform pada X_train dan pca.transform pada X_test, data dikonversi ke bentuk dengan dimensi lebih rendah, yang mempercepat proses pelatihan tanpa kehilangan informasi penting"""

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

"""Code ini berfungsi mengatasi nilai NaN setelah PCA"""

from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=50)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

models = {
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
}

results = {}
mse = pd.DataFrame(columns=['Train MSE', 'Test MSE'], index=models.keys(), dtype=float)

"""code ini terdiri empat model yang akan digunakan untuk memprediksi Sales:

- K-Nearest Neighbors (KNN), memprediksi nilai berdasarkan 5 data terdekat.
- Random Forest, menggunakan 200 pohon keputusan untuk membuat prediksi.
- Gradient Boosting, model yang meningkatkan akurasi dengan menggabungkan banyak pohon keputusan.
- XGBoost, model serupa dengan Gradient Boosting, tetapi lebih cepat dan efisien.

Kode juga membuat dua tempat penyimpanan hasil:
- results untuk menyimpan prediksi model.
- mse untuk menyimpan nilai Mean Squared Error (MSE), yaitu angka yang menunjukkan seberapa besar kesalahan prediksi model.
"""

knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
results['K-Nearest Neighbors'] = {'MAE': mae_knn, 'R2 Score': r2_knn, 'RMSE': rmse_knn}
mse.iloc[0] = [mean_squared_error(y_train, knn.predict(X_train)), mean_squared_error(y_test, y_pred_knn)]
print(f"K-Nearest Neighbors - MAE: {mae_knn:.4f}")
print(f"K-Nearest Neighbors - R2 Score: {r2_knn:.4f}")
print(f"K-Nearest Neighbors - RMSE: {rmse_knn:.4f}\n")

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
results['Random Forest'] = {'MAE': mae_rf, 'R2 Score': r2_rf, 'RMSE': rmse_rf}
mse.iloc[1] = [mean_squared_error(y_train, rf.predict(X_train)), mean_squared_error(y_test, y_pred_rf)]
print(f"Random Forest - MAE: {mae_rf:.4f}")
print(f"Random Forest - R2 Score: {r2_rf:.4f}")
print(f"Random Forest - RMSE: {rmse_rf:.4f}\n")

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
results['Gradient Boosting'] = {'MAE': mae_gb, 'R2 Score': r2_gb, 'RMSE': rmse_gb}
mse.iloc[2] = [mean_squared_error(y_train, gb.predict(X_train)), mean_squared_error(y_test, y_pred_gb)]
print(f"Gradient Boosting - MAE: {mae_gb:.4f}")
print(f"Gradient Boosting - R2 Score: {r2_gb:.4f}")
print(f"Gradient Boosting - RMSE: {rmse_gb:.4f}\n")

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
results['XGBoost'] = {'MAE': mae_xgb, 'R2 Score': r2_xgb, 'RMSE': rmse_xgb}
mse.iloc[3] = [mean_squared_error(y_train, xgb_model.predict(X_train)), mean_squared_error(y_test, y_pred_xgb)]
print(f"XGBoost - MAE: {mae_xgb:.4f}")
print(f"XGBoost - R2 Score: {r2_xgb:.4f}")
print(f"XGBoost - RMSE: {rmse_xgb:.4f}\n")

best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
best_model = models[best_model_name]
print(f"Best Model: {best_model_name}")

print(mse)

"""Hasil evaluasi menunjukkan bahwa XGBoost adalah model terbaik dengan skor R2 = 0.8313, MAE = 13.1463, dan RMSE = 20.8217, mengungguli model lainnya. Gradient Boosting berada di posisi kedua dengan R2 = 0.8268, MAE = 13.5467, dan RMSE = 21.0972. Random Forest memiliki performa lebih rendah dengan R2 = 0.7004, sedangkan K-Nearest Neighbors (KNN) menunjukkan hasil terburuk dengan R2 = 0.6237, menunjukkan kesulitan dalam menangkap pola data. Dari tabel MSE, XGBoost dan Gradient Boosting juga memiliki Test MSE yang paling rendah (433.54 dan 445.09), menandakan generalisasi yang lebih baik dibanding model lain. Dengan demikian, XGBoost dipilih sebagai model terbaik berdasarkan kinerjanya dalam memprediksi data uji."""

mse.sort_values(by='Test MSE', ascending=True).plot(kind='barh')
plt.title('Mean Squared Error Comparison')
plt.xlabel('MSE')
plt.ylabel('Model')
plt.grid(True)
plt.show()

"""Grafik ini membandingkan Mean Squared Error (MSE) dari empat model regresi: K-Nearest Neighbors (KNN), Random Forest, Gradient Boosting, dan XGBoost. Sumbu horizontal menunjukkan nilai MSE, sedangkan sumbu vertikal mewakili model yang diuji. Warna biru menunjukkan MSE pada data pelatihan (Train MSE), sedangkan warna orange menunjukkan MSE pada data uji (Test MSE). KNN memiliki Test MSE tertinggi, menunjukkan bahwa model ini kurang efektif dalam generalisasi.
Random Forest memiliki Test MSE yang lebih rendah dari KNN, tetapi masih lebih tinggi dibandingkan Gradient Boosting dan XGBoost.
Gradient Boosting dan XGBoost memiliki nilai Test MSE yang paling rendah, menunjukkan bahwa kedua model ini lebih baik dalam menangkap pola dalam data.
XGBoost memiliki Train MSE dan Test MSE yang paling kecil, mengindikasikan bahwa model ini memiliki keseimbangan terbaik antara akurasi dan generalisasi.
Secara keseluruhan, XGBoost adalah model terbaik berdasarkan MSE, karena memiliki kesalahan terkecil pada data uji.
"""

pred_sample = X_test[:5]
pred_dict = {'y_true': y_test.iloc[:5].values}
for name, model in zip(models.keys(), [knn, rf, gb, xgb_model]):
    pred_dict[f'pred_{name}'] = model.predict(pred_sample).round(2)

pred_df = pd.DataFrame(pred_dict)
print("Sample Predictions:")
print(pred_df)

"""- XGBoost dan Gradient Boosting cenderung memberikan
prediksi yang lebih akurat dan lebih dekat ke nilai sebenarnya dibandingkan model lainnya.
- KNN dan Random Forest sering kali memberikan prediksi yang lebih jauh dari nilai sebenarnya, terutama pada nilai yang lebih tinggi.
- Secara keseluruhan, XGBoost tampaknya memiliki performa terbaik, karena menghasilkan prediksi yang lebih akurat dan stabil di berbagai sampel.
"""