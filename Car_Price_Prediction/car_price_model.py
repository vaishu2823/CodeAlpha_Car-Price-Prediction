# ==========================================
# Car Price Prediction Project
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("car data.csv")

# Display First 5 Rows
print("\nFirst 5 Rows:\n")
print(df.head())

# Dataset Information
print("\nDataset Information:\n")
print(df.info())

# Missing Values
print("\nMissing Values:\n")
print(df.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:\n")
print(df.describe())

# Visualization
plt.figure(figsize=(8,6))
sns.heatmap(df.select_dtypes(include=['number']).corr(),
            annot=True,
            cmap='coolwarm')

plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.show()

# Convert Categorical Data
df = pd.get_dummies(df, drop_first=True)

# Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Actual vs Predicted Graph
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")

plt.savefig("images/prediction_graph.png")
plt.show()

print("\nProject Completed Successfully!")