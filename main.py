import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("data/student-mat.csv", sep=";")

print(df.shape)
print(df.info())
print(df.isnull().sum())  # Checking the missing data.

df_encoded = pd.get_dummies(df, drop_first=True)

print(df_encoded)

plt.figure(figsize=(10,8))
sns.heatmap(df_encoded.corr(), cmap="coolwarm")
plt.title("correlation matrix")
plt.show()


selected_features = ['G1', 'G2', 'failures', 'studytime', 'Medu', 'Fedu', 'Dalc', 'Walc']

X = df_encoded[selected_features]
y = df_encoded["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model 1 linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)

print("Linear Regression - MSE:", lr_mse)
print("Linear Regression - R2 Score:", lr_r2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, lr_pred, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([0, 20], [0, 20], color='red', linestyle='--')  # Ideal line
plt.grid(True)
plt.show()

# model 2 random forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred)
rf_r2 = r2_score(y_test, y_pred)

print("Random Forest - MSE:", rf_mse)
print("Random Forest -R2 Score:", rf_r2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Random Forest: Actual vs Predicted")
plt.plot([0, 20], [0, 20], color='red', linestyle='--')
plt.grid(True)
plt.show()

print("Compare the two models:")
print(f"Random Forest - R2: {rf_r2:.4f} | MSE: {rf_mse:.4f}")
print(f"Linear Regression - R2: {lr_r2:.4f} | MSE: {lr_mse:.4f}")
