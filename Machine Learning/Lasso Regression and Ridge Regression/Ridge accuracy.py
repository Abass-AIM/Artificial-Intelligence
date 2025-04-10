from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
housing = fetch_california_housing()
features = housing.data[:,0:2]
target = housing.target
# Load libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create ridge regression with an alpha value
regression = Ridge(alpha=0.9)
# Fit the linear regression
model = regression.fit(features_standardized, target)
# Load library
from sklearn.linear_model import RidgeCV
# Create ridge regression with three alpha values
#regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
#model_cv = regr_cv.fit(features_standardized, target)
# View coefficients
#print(model_cv.coef_)
# View alpha
#print(model_cv.alpha_)

accuracy = model.score(features_standardized, target) * 100
print(f"Accuracy: {accuracy:.2f}%")
