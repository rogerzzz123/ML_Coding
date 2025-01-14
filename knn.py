
# Algorithm:
# 1. choose a value of K
# 2. for each data point in the test set, calculate distance to all data points in the dataset
# 3. select the k nearest neighbors fro the distances
# 4. assign label based on the k nearest neighbors

from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k, distance='euclidan'):
        self.k=k
        self.distance=distance

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        y_pred=[]

        for x in X_test:
            distances=np.linalg.norm(self.X_train-X_test)
        
            nearest=np.argsort(distances)[:self.k]
            nearest_labels=self.y_train[nearest]

            label=Counter(nearest_labels).most_common(1)[0][0]
            y_pred.append(label)
        return y_pred

        