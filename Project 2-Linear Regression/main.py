
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


np.random.seed(42)
n = 120

TV = np.random.uniform(0, 300, n)
Radio = np.random.uniform(0, 50, n)
Newspaper = np.random.uniform(0, 50, n)


Sales = 3 + 0.045*TV + 0.18*Radio + 0.02*Newspaper + np.random.normal(0, 2, n)

df = pd.DataFrame({
    'TV': TV,
    'Radio': Radio,
    'Newspaper': Newspaper,
    'Sales': Sales
})

print("✅ Dataset created successfully!")
print(df.head())


print("\n✅ Basic Statistics:")
print(df.describe())

print("\n✅ Checking Missing Values:")
print(df.isnull().sum())

print("\n✅ Correlation Matrix:")
print(df.corr())


sns.pairplot(df)
plt.show()


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n✅ Data split done.")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model trained successfully!")
print("Intercept: ", model.intercept_)
print("Coefficients: ", model.coef_)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nFeature Coefficients:\n", coef_df)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print("MSE  =", mse)
print("RMSE =", rmse)
print("MAE  =", mae)
print("R²   =", r2)


residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("\n✅ Cross Validation R² Scores:", cv_scores)
print("Average R² =", cv_scores.mean())


joblib.dump(model, "model/linear_regression_model.joblib")
print("\n✅ Model saved to model/linear_regression_model.joblib")
