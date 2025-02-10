#algorihm:
# step 1: define parameters, and gradient decent 
# step 2: fit to update params
# - initialize w,b
# calculate gradient
# update w,b
# step 3: predict using params

import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr=lr
        self.max_iter=max_iter
        self.w=None
        self.b=None

    def fit(self, X, y):
        n_samples=len(X)
        n_features=X.shape[1]
        self.w=np.zeros(n_features)
        self.b=0

        for _ in range(self.max_iter):
            y_pred=np.dot(X, self.w)+self.b
            dw=(1/n_samples)*(np.dot(X.T, (y_pred-y)))
            db=(1/n_samples)*(np.sum(y_pred-y))
            self.w-=self.lr*dw
            self.b-=self.lr*db
    def predict(self,X):
        return np.dot(X, self.w)+self.b

# Generate synthetic dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])  # y = 2x (simple linear relationship)

# Train the model
model = LinearRegression(lr=0.1, max_iter=1000)
model.fit(X, y)

# Predict on new data
X_test = np.array([[6], [7], [8]])
predictions = model.predict(X_test)

print("Predictions:", predictions)