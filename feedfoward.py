# assume we have a two-layer feedforeward nerual network, write the code to build it

import numpy as np

class FeedForward:
    def __init__(self, input_size, hidden_size, output_size):
        self.params={}
        self.params["w1"] = np.random.randn(input_size, hidden_size)
        self.params["w2"] = np.random.randn(hidden_size, output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"]=np.zeros(output_size)
    
    def forward(self, X):
        w1, b1=self.params["w1"], self.params["b1"]
        w2, b2=self.params["w2"], self.params["b2"]
        z1 = np.dot(X, w1)+b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, w2)+b2
        z2 -= np.max(z2, axis=1, keepdims=True)  # Stability trick
        probs=np.exp(z2)/np.sum(np.exp(z2), axis=-1, keepdims=True)   
        return probs, z1, a1, z2


    def backward(self, X, y, probs, z1, a1, z2):
        w2=self.params["w2"]
        num_samples=X.shape[0]

        loss=-np.mean(np.log(probs[np.arange(num_samples), y]))

        loss=-np.mean(np.log(probs[np.arange(num_samples), y]))

        delta3=probs 
        delta3[np.arange(num_samples), y]-=1 # derivative of loss w.r.t z2 (softmax layer)
        delta3/=num_samples

        dw2=np.dot(a1.T, delta3)
        db2=np.sum(delta3, axis=0)
        delta2=np.dot(delta3, w2.T) * (z1>0)

        dw1=np.dot(X.T, delta2)
        db1=np.sum(delta2, axis=0)
        # grads={'dw1':dw1, 'dw2':dw2, 'db1':db1, 'db2':db2}
        return loss, dw1, dw2, db1, db2
    
    def train(self, X, y, num_epochs, lr=1e-5):
        for epoch in range(num_epochs):
            probs, z1, a1, z2 = self.forward(X)
            loss, dw1, dw2, db1, db2 = self.backward(X,y, probs, z1, a1, z2)
            self.params["w1"]-=lr*dw1
            self.params["w2"]-=lr*dw2
            self.params['b1']-=lr*db1
            self.params['b2']-=lr*db2

            if epoch%100==0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    





