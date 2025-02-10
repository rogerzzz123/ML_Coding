import numpy as np

class LogisticRegression:
    def __init__(self, lr, max_iter):
        self.lr=lr
        self.max_iter=max_iter
        self.w=None
        self.b=None
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def fit(self, X,y):

        n_samples, n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0
        for i in range(self.max_iter):
            y_pred=self.sigmoid(np.dot(X, self.w)+self.b)

            dw=(1/n_samples)*(np.dot(X.T,(y_pred-y)))
            db=(1/n_samples)*np.sum(y_pred-y)
            self.w-=self.lr*dw
            self.b-=self.lr*db

    def predict(self, X):
        pred=np.dot(X, self.w)+self.b
        pred_prob=self.sigmoid(pred)
        return pred_prob


# Generate synthetic dataset
X_train = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]
])
y_train = np.array([0, 0, 1, 1, 1])  # Binary labels

# Train the model
model = LogisticRegression(lr=0.1, max_iter=1000)
model.fit(X_train, y_train)

# Predict on new data
X_test = np.array([
    [1.5, 2.5],  # Should be close to class 0
    [3.5, 4.5],  # Should be close to class 1
    [5, 7]       # Should be class 1
])
predictions = model.predict(X_test)

print("Predictions:", predictions)