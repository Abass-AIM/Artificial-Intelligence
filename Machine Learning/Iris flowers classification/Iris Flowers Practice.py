import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import sys
from scipy import sparse
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][ : 193] + "\n...")
print("Firts Five columns of data:\n{}".format(iris_dataset['data'][ :]))
print("shape of data:\n{}".format(iris_dataset['data'].shape))
print("Target:\n{}".format(iris_dataset['target']))
print("Shape of target:\n{}".format(iris_dataset['target']))

#SPLITTING THE DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

#PRINT TRAIN DATASET
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
#PRINT TEST DATASET
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


#VISUALIZATION
# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),marker='o', hist_kwds={'bins': 20}, s=60,alpha=.8)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
 #PREDICTION
 
X_new = np.array([[5,2.9,1,0.3]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction:", prediction)
print("Predicted target name:",iris_dataset['target_names'][prediction])
#EVALUATING THE MODEL
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))