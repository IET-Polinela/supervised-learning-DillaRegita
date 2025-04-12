import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# 1. Baca dataset
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')

# 2. Visualisasi Boxplot untuk semua fitur numerik
plt.figure(figsize=(12, 8))
sns.boxplot(data=df)
plt.title("Boxplot untuk Semua Fitur Sebelum Penghapusan Outlier")
plt.xticks(rotation=90)
# Simpan gambar ke file
plt.savefig('/content/supervised-learning-DillaRegita/boxplot_before.png')
plt.show()

# 3. Identifikasi Outlier dengan Z-Score
# Hitung Z-Score untuk semua fitur numerik
z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))

# Tentukan threshold Z-Score (nilai abs > 3 adalah outlier)
outliers = (z_scores > 3)

# Tampilkan jumlah outlier pada setiap fitur
outlier_counts = np.sum(outliers, axis=0)
print("Jumlah Outlier per Fitur:")
print(pd.DataFrame(outlier_counts, index=df.select_dtypes(include=[np.number]).columns, columns=["Outlier Count"]))

# 4. Menghapus Outlier
# Baris yang mengandung outlier akan dihapus
df_no_outliers = df[~np.any(outliers, axis=1)]

# 5. Visualisasi Boxplot Setelah Penghapusan Outlier
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_no_outliers)
plt.title("Boxplot untuk Semua Fitur Setelah Penghapusan Outlier")
plt.xticks(rotation=90)
# Simpan gambar ke file
plt.savefig('/content/supervised-learning-DillaRegita/boxplot_after.png')
plt.show()

# Tampilkan ukuran dataset setelah menghapus outlier
print(f"Ukuran dataset tanpa outlier: {df_no_outliers.shape}")
print(f"Ukuran dataset dengan outlier: {df.shape}")
