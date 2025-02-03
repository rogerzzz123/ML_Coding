import numpy as np

class KMeans:
    def __init__(self, k, max_iter, tol=1e-4):
        self.k=k
        self.max_iter=max_iter
        self.tol=tol

    # def assign_cluster(self,x):
    #     n_samples=x.shape[0]
    #     distance=np.zeros((n_samples, self.k))
    #     for i, centroid in enumerate(self.centroids):
    #         distance[:i]=np.sqrt(np.sum((X-centroid)**2), axis=1)
    #     return np.argmin(distance, axis=1)

    def fit(self, X): 

        random_idx=np.random.choice(len(X), size=self.k, replace=False)
        self.centroids=X[random_idx]
        for _ in range(self.max_iter):
            distance=np.linalg.norm(X[:,np.newaxis]-self.centroids, axis=2)
            self.labels=np.argmin(distance, axis=1)
            new_centroids=np.array([X[self.label==i].mean(axis=0) for i in range(self.k)])
            if np.linalg.norm(new_centroids-self.centroids)<self.tol:
                break
            self.centroids=new_centroids
    
    def predict(self, X):
        distances=np.linalg.norm(X[:np.newaxis]-self.centroids, axis=2)
        labels=np.argmin(distances, axis=1)
        return labels
    

class KMeans:
    def __init__(self, k, max_iter):
        self.k=k
        self.max_iter=max_iter
        self.centroids=None
        self.clusters=None
    
    def fit(self, clients):
        clients=np.array(clients)
        n_samples=len(clients)
        self.centroids=clients[np.random.choice(n_samples, self.k, replace=False)]

        for _ in range(self.max_iter):
            clusters=[[] for _ in range(self.k)]
            for x in clients:
                distance=np.linalg.norm(x-self.centroids,axis=1)
                nearest=np.argmin(distance)
                clusters[nearest].append(x)
            new_centroids=[]
            for cluster in clusters:
                new_centroids.append(np.mean(cluster), axis=0)
            
            if np.linalg.norm(new_centroids-self.centroids)<1e-6:
                break
            self.centroids=new_centroids
    
    def predict(self, X):
        cluster=[]
        for x in X:
            distance=np.linalg.norm(x-self.centroids, axis=1)
            nearest=np.argmin(distance)
            cluster.append(nearest)
        return cluster


                    