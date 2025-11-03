# Load libraries
#import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create decision tree classifier object
decisiontree = DecisionTreeClassifier(random_state=0)

# Train model
model = decisiontree.fit(features, target)
# Make new observation
observation = [[5,4,3,2]]
# Predict observation's value
print(model.predict(observation))


# Create random forest classifier object using entropy
randomforest_entropy = RandomForestClassifier(
 criterion="entropy", random_state=0)
# Train model
model_entropy = randomforest_entropy.fit(features, target)
print(model_entropy.predict(observation))
