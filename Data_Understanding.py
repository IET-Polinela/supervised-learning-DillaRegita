import pandas as pd

# Baca dataset
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')

# Statistik deskriptif awal (semua tipe data)
deskriptif = df.describe(include='all').transpose()

# Hitung statistik numerik tambahan
deskriptif["mean"] = df.mean(numeric_only=True)
deskriptif["median"] = df.median(numeric_only=True)
deskriptif["std"] = df.std(numeric_only=True)
deskriptif["min"] = df.min(numeric_only=True)
deskriptif["max"] = df.max(numeric_only=True)

# Jumlah data non-null dan missing
deskriptif["count_non_null"] = df.count()
deskriptif["missing"] = df.isnull().sum()
deskriptif["missing_percent"] = (df.isnull().sum() / len(df)) * 100

# Tambahkan Q1 dan Q3
Q1 = df.quantile(0.25, numeric_only=True)
Q2 = df.quantile(0.5, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)

deskriptif["Q1 (25%)"] = Q1
deskriptif["Q2 (50%)"] = Q2
deskriptif["Q3 (75%)"] = Q3

# Tampilkan deskripsi lengkap
kolom_output = ['mean', 'median', 'std', 'min', 'Q1 (25%)', 'Q2 (50%)', 'Q3 (75%)', 'max', 'count_non_null', 'missing', 'missing_percent']
print(deskriptif[kolom_output])
