import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Baca dataset tanpa outlier
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')

# Pilih fitur numerik
numeric_columns = df.select_dtypes(include='number').columns

# Terapkan MinMaxScaler
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Menentukan jumlah baris dan kolom untuk subplot
n_features = len(numeric_columns)
n_cols = 4  # Jumlah kolom dalam subplot
n_rows = int(np.ceil(n_features / n_cols))  # Menyesuaikan baris berdasarkan jumlah fitur

# Visualisasikan distribusi data setelah MinMaxScaler
plt.figure(figsize=(n_cols * 4, n_rows * 4))  # Menyesuaikan ukuran gambar berdasarkan jumlah subplot
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df_scaled[column], bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Distribusi {column} Setelah MinMaxScaler')
    plt.xlabel(column)
    plt.ylabel('Frekuensi')

plt.tight_layout(pad=3.0)  # Menambah jarak antar subplot agar tidak saling tumpang tindih
plt.savefig("/content/supervised-learning-DillaRegita/distribution_after_minmax_scaling.png")
plt.show()
