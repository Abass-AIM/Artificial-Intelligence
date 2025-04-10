# Load library
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
# Load data
housing = fetch_california_housing()
features = housing.data[:,:]
target = housing.target
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create lasso regression with alpha value
regression = Lasso(alpha=0.001)
# Fit the linear regression
model = regression.fit(features_standardized, target)
# View coefficients
print(model.coef_)
# Create lasso regression with a high alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
print(model_a10.coef_)
from sklearn.metrics import r2_score
# Calculate R² score (accuracy percentage)
accuracy = model.score(features_standardized, target) * 100
print(f"Accuracy (R²): {accuracy:.2f}%")
accuracy_a10 = model_a10.score(features_standardized, target) * 100
print(f"Accuracy (α=10): {accuracy_a10:.2f}%")