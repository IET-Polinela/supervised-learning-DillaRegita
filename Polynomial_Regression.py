import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

# ----------- MinMax Scaling -----------
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------- MODEL 1: Linear Regression -----------
model_linear = LinearRegression()
model_linear.fit(X_train_scaled, y_train)
y_pred_linear = model_linear.predict(X_test_scaled)

# ----------- MODEL 2: Polynomial Degree 2 -----------
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train_scaled)
X_test_poly2 = poly2.transform(X_test_scaled)

model_poly2 = Ridge(alpha=1.0)
model_poly2.fit(X_train_poly2, y_train)
y_pred_poly2 = model_poly2.predict(X_test_poly2)

# ----------- MODEL 3: Polynomial Degree 3 -----------
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train_scaled)
X_test_poly3 = poly3.transform(X_test_scaled)

model_poly3 = Ridge(alpha=1.0)
model_poly3.fit(X_train_poly3, y_train)
y_pred_poly3 = model_poly3.predict(X_test_poly3)

# ----------- VISUALISASI: MinMax Scaled -----------
plt.figure(figsize=(15, 5))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_linear, color='red', label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Id')
plt.ylabel('SalePrice')
plt.legend()

# Polynomial Degree 2
plt.subplot(1, 3, 2)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_poly2, color='green', label='Predicted')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('Id')
plt.legend()

# Polynomial Degree 3
plt.subplot(1, 3, 3)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_poly3, color='purple', label='Predicted')
plt.title('Polynomial Regression (Degree 3)')
plt.xlabel('Id')
plt.legend()

plt.suptitle('Model Comparison for MinMaxScaled', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('/content/supervised-learning-DillaRegita/polynomial_minmax_plot.png')
plt.show()

# ======================== TAMBAHAN: StandardScaler ========================
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)

# MODEL 4: Linear Regression (Standard Scaled)
model_linear_std = LinearRegression()
model_linear_std.fit(X_train_std, y_train)
y_pred_linear_std = model_linear_std.predict(X_test_std)

# MODEL 5: Polynomial Degree 2 (Standard Scaled)
X_train_poly2_std = poly2.fit_transform(X_train_std)
X_test_poly2_std = poly2.transform(X_test_std)

model_poly2_std = Ridge(alpha=1.0)
model_poly2_std.fit(X_train_poly2_std, y_train)
y_pred_poly2_std = model_poly2_std.predict(X_test_poly2_std)

# MODEL 6: Polynomial Degree 3 (Standard Scaled)
X_train_poly3_std = poly3.fit_transform(X_train_std)
X_test_poly3_std = poly3.transform(X_test_std)

model_poly3_std = Ridge(alpha=1.0)
model_poly3_std.fit(X_train_poly3_std, y_train)
y_pred_poly3_std = model_poly3_std.predict(X_test_poly3_std)

# ----------- VISUALISASI: Standard Scaled -----------
plt.figure(figsize=(15, 5))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_linear_std, color='red', label='Predicted')
plt.title('Linear Regression (Standard Scaled)')
plt.xlabel('Id')
plt.ylabel('SalePrice')
plt.legend()

# Polynomial Degree 2
plt.subplot(1, 3, 2)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_poly2_std, color='green', label='Predicted')
plt.title('Polynomial Regression (Degree 2)')
plt.xlabel('Id')
plt.legend()

# Polynomial Degree 3
plt.subplot(1, 3, 3)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual', alpha=0.6)
plt.plot(range(len(y_test)), y_pred_poly3_std, color='purple', label='Predicted')
plt.title('Polynomial Regression (Degree 3)')
plt.xlabel('Id')
plt.legend()

plt.suptitle('Model Comparison for StandardScaled', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('/content/supervised-learning-DillaRegita/polynomial_std_plot.png')
plt.show()
