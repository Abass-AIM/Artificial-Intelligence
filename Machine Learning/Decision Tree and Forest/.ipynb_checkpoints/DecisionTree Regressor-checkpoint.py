# Load libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets

import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep= r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
features = data[:,0:2]
target = raw_df.values[1::2, 2]

# Load data with only two features
#boston = datasets.load_boston()
#features = boston.data[:,0:2]
#target = boston.target
# Create decision tree classifier object
decisiontree = DecisionTreeRegressor(random_state=0)
# Train model
model = decisiontree.fit(features, target)

# Make new observation
observation = [[0.02, 16]]
# Predict observation's value
print(model.predict(observation))

# Create decision tree classifier object using entropy
decisiontree_mae = DecisionTreeRegressor(criterion="absolute_error", random_state=0)
# Train model
model_mae = decisiontree_mae.fit(features, target)

# Make new observation
observation = [[0.02, 16]]
# Predict observation's value
print(model_mae.predict(observation))
