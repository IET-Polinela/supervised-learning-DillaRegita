import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

# Baca dataset tanpa outlier
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')

# Pilih fitur numerik
numeric_columns = df.select_dtypes(include='number').columns

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Terapkan StandardScaler ke fitur numerik
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Tentukan jumlah subplot (baris dan kolom)
num_columns = 4  # Set kolom menjadi 4 untuk distribusi
num_rows = math.ceil(len(numeric_columns) / num_columns)  # Hitung jumlah baris

# Visualisasikan distribusi data setelah scaling
plt.figure(figsize=(16, num_rows * 4))  # Ukuran figure dinamis sesuai banyaknya baris
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(num_rows, num_columns, i)
    plt.hist(df_scaled[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribusi {column} Setelah StandardScaler')
    plt.xlabel(column)
    plt.ylabel('Frekuensi')

plt.tight_layout()
plt.savefig("distribution_after_standard_scaler.png")
plt.show()
