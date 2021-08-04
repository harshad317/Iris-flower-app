import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

df = pd.read_csv('Iris.csv')

df = df.drop('Id', axis= 1)

df.Species = df.Species.map({'Iris-setosa': 0,
                             'Iris-versicolor': 1,
                             'Iris-virginica': 2})
    
X = df.drop(['Species'], axis= 1).values
y = df.Species.values

regressor = LogisticRegression(max_iter=1000)
regressor.fit(X, y)

pickle.dump(regressor, open('model.pkl', 'wb'))