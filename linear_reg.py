#algorihm:
# step 1: define parameters, and gradient decent 
# step 2: fit to update params
# - initialize w,b
# calculate gradient
# update w,b
# step 3: predict using params

class LinearRegression:
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr=lr
        self.max_iter=max_iter
        self.w=None
        self.b=None

    def fit(self, X, y):
        n_samples=len(X)
        n_features=len(X[0])
        self.w=np.zeros()
        self.b=0

        for _ in range(self.max_iter):
            y_pred=np.dot(X, self.w)+self.b
            dw=(1/n_samples)*(np.dot(X.T, (y_pred-y)))
            db=(1/n_samples)*(np.sum(y_pred-y))
            self.w-=self.lr*dw
            self.b-=self.lr*db
    def predict(self,X):
        return np.dot(X, self.w)+self.b