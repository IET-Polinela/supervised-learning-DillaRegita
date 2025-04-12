import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data clean
df = pd.read_csv('/content/supervised-learning-DillaRegita/train_clean.csv', delimiter=';')

# Preprocessing
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

# Imputasi
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fungsi untuk simpan MSE dan R2
results = []
model_preds = []
model_names = []

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append({'Model': name, 'MSE': mse, 'R2': r2})
    model_preds.append((y_true, y_pred))
    model_names.append(name)

# ========== LINEAR REGRESSION ==========
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
y_pred_linreg = linreg.predict(X_test_scaled)
evaluate_model("Linear Regression", y_test, y_pred_linreg)

# ========== POLYNOMIAL DEGREE 2 ==========
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train_scaled)
X_test_poly2 = poly2.transform(X_test_scaled)
model_poly2 = Ridge()
model_poly2.fit(X_train_poly2, y_train)
y_pred_poly2 = model_poly2.predict(X_test_poly2)
evaluate_model("Polynomial (Deg 2)", y_test, y_pred_poly2)

# ========== POLYNOMIAL DEGREE 3 ==========
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train_scaled)
X_test_poly3 = poly3.transform(X_test_scaled)
model_poly3 = Ridge()
model_poly3.fit(X_train_poly3, y_train)
y_pred_poly3 = model_poly3.predict(X_test_poly3)
evaluate_model("Polynomial (Deg 3)", y_test, y_pred_poly3)

# ========== KNN K = 3 ==========
knn3 = KNeighborsRegressor(n_neighbors=3)
knn3.fit(X_train_scaled, y_train)
y_pred_knn3 = knn3.predict(X_test_scaled)
evaluate_model("KNN (K=3)", y_test, y_pred_knn3)

# ========== KNN K = 5 ==========
knn5 = KNeighborsRegressor(n_neighbors=5)
knn5.fit(X_train_scaled, y_train)
y_pred_knn5 = knn5.predict(X_test_scaled)
evaluate_model("KNN (K=5)", y_test, y_pred_knn5)

# ========== KNN K = 7 ==========
knn7 = KNeighborsRegressor(n_neighbors=7)
knn7.fit(X_train_scaled, y_train)
y_pred_knn7 = knn7.predict(X_test_scaled)
evaluate_model("KNN (K=7)", y_test, y_pred_knn7)

# ======================
# TABEL PERBANDINGAN
# ======================
result_df = pd.DataFrame(results)
print("\n\U0001F4CA Tabel Perbandingan Model:")
print(result_df.sort_values(by='R2', ascending=False).to_string(index=False))

# ======================
# VISUALISASI SEMUA MODEL
# ======================
def compare_all_models_visual(y_tests_preds, titles, filename):
    n_models = len(y_tests_preds)
    rows, cols = 2, 3
    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (y_true, y_pred) in enumerate(y_tests_preds):
        plt.subplot(rows, cols, i + 1)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(titles[i])

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

compare_all_models_visual(model_preds, model_names, '/content/supervised-learning-DillaRegita/linear_regression_comparison.png')
