from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
housing = fetch_california_housing()
features = housing.data[:,0:2]
target = housing.target
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features, target)

print("housing.keys(): \n{}".format(housing.keys()))
print(model.intercept_)
print(target[0]*1000)
print(model.predict(features)[0]*1000)
print(model.coef_)
print(model.coef_[0]*1000)


# Create interaction term
interaction = PolynomialFeatures(
 degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)
#Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_interaction, target)
# Import library
import numpy as np
# For each observation, multiply the values of the first and second feature
interaction_term = np.multiply(features[:, 0], features[:, 1])
# View interaction term for first observation
print(interaction_term[0])

# Create polynomial features x^2 and x^3
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)
# Create linear regression
regression = LinearRegression()
# Fit the linear regression
model = regression.fit(features_polynomial, target)
print(features[0])
print(features[0]**2)
print(features[0]**3)
print(features_polynomial[0])
accuracy = model.score(features_polynomial, target) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Load libraries
'''from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)
# Create ridge regression with an alpha value
regression = Ridge(alpha=0.5)
# Fit the linear regression
model = regression.fit(features_standardized, target)
# Load library
from sklearn.linear_model import RidgeCV
# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)
# View coefficients
print(model_cv.coef_)
# View alpha
print(model_cv.alpha_)'''