import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Baca dataset tanpa outlier
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')

# Pilih fitur numerik
numeric_columns = df.select_dtypes(include='number').columns

# Tentukan jumlah kolom dan baris untuk subplot (maksimal 3 kolom)
n_columns = 3  # Jumlah kolom
n_rows = math.ceil(len(numeric_columns) / n_columns)  # Menghitung jumlah baris yang diperlukan

# Visualisasikan distribusi data sebelum scaling
plt.figure(figsize=(15, 5 * n_rows))  # Ukuran gambar menyesuaikan dengan banyaknya baris
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_columns, i)
    plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribusi {column} Sebelum Scaling')
    plt.xlabel(column)
    plt.ylabel('Frekuensi')

plt.tight_layout()  # Untuk memastikan tidak ada tumpang tindih
plt.savefig("distribution_before_scaling.png")
plt.show()

# Menampilkan gambar yang telah disimpan
img = mpimg.imread('distribution_before_scaling.png')
plt.figure(figsize=(10, 6))
plt.imshow(img)
plt.axis('off')  # Menonaktifkan axis agar hanya gambar yang terlihat
plt.show()
