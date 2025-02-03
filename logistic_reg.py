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