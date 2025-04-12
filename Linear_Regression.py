import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load data
df_raw = pd.read_csv("/content/supervised-learning-DillaRegita/train.csv", delimiter=';')
df_clean = pd.read_csv("/content/supervised-learning-DillaRegita/train_clean.csv", delimiter=';')

# -------------------- 
# MODEL DENGAN OUTLIER 
# --------------------
X_raw = pd.get_dummies(df_raw.drop(columns=['SalePrice']), drop_first=True)
y_raw = df_raw['SalePrice']

# Impute missing values in the features using mean strategy
imputer = SimpleImputer(strategy='mean')
X_raw = imputer.fit_transform(X_raw)

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

model_raw = LinearRegression()
model_raw.fit(X_train_raw, y_train_raw)
y_pred_raw = model_raw.predict(X_test_raw)

# ---------------------------- 
# MODEL TANPA OUTLIER & SCALING 
# ----------------------------
X_clean = pd.get_dummies(df_clean.drop(columns=['SalePrice']), drop_first=True)
y_clean = df_clean['SalePrice']

# Impute missing values in the features using mean strategy
X_clean = imputer.fit_transform(X_clean)

# Agar kolomnya tetap konsisten antara train & test
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
X_train, X_test = pd.DataFrame(X_train), pd.DataFrame(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)
model_minmax = LinearRegression()
model_minmax.fit(X_train_minmax, y_train)
y_pred_minmax = model_minmax.predict(X_test_minmax)

# StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
model_std = LinearRegression()
model_std.fit(X_train_std, y_train)
y_pred_std = model_std.predict(X_test_std)

# -------------------- 
# VISUALISASI HASIL 
# --------------------

def plot_results(y_true, y_pred, title):
    residuals = y_true - y_pred

    plt.figure(figsize=(16, 4))

    # Scatter plot: Actual vs Predicted
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title} - Prediksi vs Aktual')

    # Residual plot
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title(f'{title} - Plot Residual')

    # Histogram residuals
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f'{title} - Distribusi Residual')

    plt.tight_layout()

    # Save plot to file
    plt.savefig(f'/content/supervised-learning-DillaRegita/Linear_Regression.png')
    plt.close()

# Evaluasi & visualisasi
print("Model dengan Outlier")
print("MSE:", mean_squared_error(y_test_raw, y_pred_raw))
print("R2:", r2_score(y_test_raw, y_pred_raw))
plot_results(y_test_raw, y_pred_raw, "Model dengan Outlier")

print("Model MinMaxScaler")
print("MSE:", mean_squared_error(y_test, y_pred_minmax))
print("R2:", r2_score(y_test, y_pred_minmax))
plot_results(y_test, y_pred_minmax, "Model MinMaxScaler")

print("Model StandardScaler")
print("MSE:", mean_squared_error(y_test, y_pred_std))
print("R2:", r2_score(y_test, y_pred_std))
plot_results(y_test, y_pred_std, "Model StandardScaler")
