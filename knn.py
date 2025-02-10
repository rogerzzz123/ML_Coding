
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
            distances=np.linalg.norm(self.X_train-x, axis=1)
        
            nearest=np.argsort(distances)[:self.k]
            nearest_labels=self.y_train[nearest]

            label=Counter(nearest_labels).most_common(1)[0][0]
            y_pred.append(label)
        return np.array(y_pred)
    
    # Training data
X_train = np.array([[1, 2], [3, 4], [5, 6], [8, 8]])
y_train = np.array([0, 1, 1, 0])

# Test data
X_test = np.array([[2, 3], [6, 7]])

# Train KNN
knn = KNN(k=2, distance='euclidean')
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
print("Predicted labels:", predictions)