import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset clean
df = pd.read_csv("/content/supervised-learning-DillaRegita/train_clean.csv", delimiter=';')

# 2. Preprocessing
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

# Imputasi nilai hilang
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (pakai MinMaxScaler seperti pada Linear/Poly Regression sebelumnya)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Visualisasi dan Evaluasi
def evaluate_and_plot_knn(y_true, y_pred, k):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nKNN Regression (K = {k})")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

# Tambahan: Simpan semua skor evaluasi ke dalam list
results = []

for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred_knn)
    r2 = r2_score(y_test, y_pred_knn)
    results.append({'K': k, 'MSE': mse, 'R2': r2})

    evaluate_and_plot_knn(y_test, y_pred_knn, k)

# Tampilkan semua skor dalam bentuk DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison for KNN:")
print(results_df)

# Visualisasi comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.lineplot(data=results_df, x='K', y='MSE', marker='o')
plt.title('MSE vs K')
plt.xlabel('K')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
sns.lineplot(data=results_df, x='K', y='R2', marker='o', color='green')
plt.title('R² Score vs K')
plt.xlabel('K')
plt.ylabel('R² Score')

plt.tight_layout()
plt.savefig('/content/supervised-learning-DillaRegita/KNN_Regression_Comparison.png')
plt.show()
