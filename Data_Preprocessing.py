# 1. Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 2. Baca file dataset (ganti path jika perlu)
df = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')  # pakai delimiter=';' jika file memakai titik koma

# 3. Tampilkan nama-nama kolom (opsional untuk verifikasi target)
print("Kolom-kolom dataset:", df.columns.tolist())

# 4. Tangani nilai kosong (jika ada)
# Untuk kolom kategorikal: isi dengan 'None'
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna('None')

# Untuk kolom numerik: isi dengan median
df.fillna(df.median(numeric_only=True), inplace=True)

# 5. Encoding fitur kategorikal
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. Pisahkan fitur (X) dan target (y)
# Ganti 'SalePrice' jika nama kolom target berbeda
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# 7. Bagi dataset menjadi training dan testing set (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Cek hasil
print(f"Ukuran dataset: {df.shape}")
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
