class LinerRegression:
    def __init__(self, lr, max_iter):
        self.lr=lr
        self.max_iter=max_iter
        self.w=None
        self.b=None

    def fit(self,X,y):
        n_samples, n_features=X.shape
        self.w=np.zeros(n_features)
        self.b=0

        for _ in range(max_iter):
            y_pred=np.dot(X, self.w)+self.b
            dw=(1/n_samples) * np.dot(X.T, (y_pred-y))
            db=(1/n_samples) * np.sum(y_pred-y)
            self.w=w-self.lr*dw
            self.b=b-self.b+db
    
    def predict(self,x):
        return np.dot(X, self.w)+self.b
