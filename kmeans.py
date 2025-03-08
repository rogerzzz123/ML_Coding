import numpy as np
# 无论什么样的input 我们都尝试给他转换成numpy array， 然后统一一种numpy array的写法即可
# K-Means 的输入应该是 (n_samples, n_features)，所以一维数据应该视作 n_samples，而不是 n_features。

import numpy as np
import pandas as pd


################################################
######### 这里我们先处理好input的shape############
##############################################

def process_input(X):
    """Ensure input X is a NumPy array with shape (n_samples, n_features)"""
    
    # Case 1: Pandas DataFrame 或 Pandas Series
    if isinstance(X, pd.DataFrame):
        X = X.values  # 转换为 NumPy array
    elif isinstance(X, pd.Series):
        X = X.values.reshape(-1, 1)  # Series 需要变成 2D
    
    # Case 2: Python List 转 NumPy
    X = np.array(X, dtype=np.float64)
    
    # Case 3: 如果是一维数组 `(n_samples,)`，转换为 `(n_samples, 1)`
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    return X

# # ✅ 2D NumPy Array (已满足要求)
# X1 = np.array([[1, 2], [3, 4], [5, 6]])
# print(process_input(X1).shape)  # (3, 2)

# # ❌ 1D NumPy Array → 需要 reshape
# X2 = np.array([1, 2, 3, 4])
# print(process_input(X2).shape)  # (4, 1)

# # ✅ Python List of Lists (已满足要求)
# X3 = [[1, 2], [3, 4], [5, 6]]
# print(process_input(X3).shape)  # (3, 2)

# # ❌ Python List of Scalars → 需要 reshape
# X4 = [1, 2, 3, 4]
# print(process_input(X4).shape)  # (4, 1)

# # ✅ Pandas DataFrame
# df = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
# print(process_input(df).shape)  # (3, 2)

# # ✅ Pandas Series
# series = pd.Series([1, 2, 3, 4])
# print(process_input(series).shape)  # (4, 1)


class KMeans:
    def __init__(self, k, max_iter):
        self.k=k
        self.max_iter=max_iter
        self.centroids=None
    
    def fit(self, X):
        n_samples, n_features=X.shape
        random_idx=np.random.choice(n_samples, size=self.k, replace=False)
        self.centroids=X[random_idx]
        
        for _ in range(self.max_iter):
            # distances=np.linalg.norm(X[:, np.newaxis]-self.centroids, axis=2)
            distances=np.zeros((n_samples, self.k))
            for i in range(self.k):
                distances[:,i]=np.sqrt(np.sum((X-self.centroids[i])**2, axis=1)) # 按照行
            clusters=np.argmin(distances, axis=1)
            new_centroids=np.zeros_like(self.centroids)
            for i in range(self.k):
                cluster_points=X[clusters==i]
                if len(cluster_points)>0:
                    new_centroids[i]=cluster_points.mean(axis=0) # 按照列
                else:
                    new_centroids[i]=self.centroids[i]
            if np.linalg.norm(new_centroids-self.centroids)<1e-6:
                break
            self.centroids=new_centroids
        return
    
    def predict(self, X):
        distances=np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            distances[:,i]=np.sqrt(np.sum((X-self.centroids[i])**2, axis=1)) # 按照行
        return np.argmin(distances, axis=1)

if __name__ == "__main__":
    X = np.array([
        [1, 2], [1, 4], [1, 0],
        [4, 2], [4, 4], [4, 0]
    ])
    k = 2
    model = KMeans(k, 100)
    model.fit(X)
    print("Centroids:", model.centroids)
    labels = model.predict(X)
    print("Cluster assignments:", labels)

# class KMeans:
#     def __init__(self, k, max_iter, tol=1e-4):
#         self.k=k
#         self.max_iter=max_iter
#         self.tol=tol

#     # def assign_cluster(self,x):
#     #     n_samples=x.shape[0]
#     #     distance=np.zeros((n_samples, self.k))
#     #     for i, centroid in enumerate(self.centroids):
#     #         distance[:i]=np.sqrt(np.sum((X-centroid)**2), axis=1)
#     #     return np.argmin(distance, axis=1)

#     def fit(self, X): 

#         random_idx=np.random.choice(len(X), size=self.k, replace=False)
#         self.centroids=X[random_idx]
#         for _ in range(self.max_iter):
#             distance=np.linalg.norm(X[:,np.newaxis]-self.centroids, axis=2)
#             self.labels=np.argmin(distance, axis=1)
#             new_centroids=np.array([X[self.label==i].mean(axis=0) for i in range(self.k)])
#             if np.linalg.norm(new_centroids-self.centroids)<self.tol:
#                 break
#             self.centroids=new_centroids
    
#     def predict(self, X):
#         distances=np.linalg.norm(X[:np.newaxis]-self.centroids, axis=2)
#         labels=np.argmin(distances, axis=1)
#         return labels
    

# class KMeans:
#     def __init__(self, k, max_iter):
#         self.k=k
#         self.max_iter=max_iter
#         self.centroids=None
#         self.clusters=None
    
#     def fit(self, clients):
#         clients=np.array(clients)
#         n_samples=len(clients)
#         self.centroids=clients[np.random.choice(n_samples, self.k, replace=False)]

#         for _ in range(self.max_iter):
#             clusters=[[] for _ in range(self.k)]
#             for x in clients:
#                 distance=np.linalg.norm(x-self.centroids,axis=1)
#                 nearest=np.argmin(distance)
#                 clusters[nearest].append(x)
#             new_centroids=[]
#             for cluster in clusters:
#                 new_centroids.append(np.mean(cluster), axis=0)
            
#             if np.linalg.norm(new_centroids-self.centroids)<1e-6:
#                 break
#             self.centroids=new_centroids
    
#     def predict(self, X):
#         cluster=[]
#         for x in X:
#             distance=np.linalg.norm(x-self.centroids, axis=1)
#             nearest=np.argmin(distance)
#             cluster.append(nearest)
#         return cluster


# class KMeans:
#     def __init__(self, k, max_iter):
#         self.k=k
#         self.max_iter=max_iter
#         self.centroids=None
#         self.clusters=None
    
#     def fit(self, X):
#         random_idx=np.random.choice(len(X), size=self.k, replace=False)
#         self.centroids=X[random_idx]
        
#         for _ in range(self.max_iter):
#             distance=np.linalg.norm(X[:, np.newaxis]-self.centroids, axis=2)
#             nearest=np.argmin(distance)
#             new_centroids=np.array([X[nearest==i].mean(axis=0) if np.any(nearest==i) else self.centroids[i] for i in range(self.k)])
#             if np.linalg.norm(new_centroids-self.centroids)<1e-6:
#                 break
#             self.centroids=new_centroids
    
#     def predict(self, X):
#         distances=np.linalg.norm(X[:,np.newaxis]-self.centroids, axis=2)
#         labels=np.argmin(distances, axis=1)
#         return labels
    
# # Create sample data
# X = np.array([
#     [1, 2], [1, 4], [1, 0],
#     [10, 2], [10, 4], [10, 0]
# ])
# print(X.shape)

# # # Train K-Means
# # kmeans = KMeans(k=2, max_iter=100)
# # kmeans.fit(X)
# # # Predict cluster labels
# # labels = kmeans.predict(X)
# # print("Cluster assignments:", labels)
# # print("Centroids:", kmeans.centroids)
            

# import numpy as np

# def kmeans(X, k, centroids):
#     # centroids = X[np.random.choice(len(X), size=k, replace=False)]
#     clusters = []
    
#     for i in range(10):
#         distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
#         clusters = np.argmin(distances, axis=1)
#         new_centroids = np.array([X[np.where(clusters==i)].mean(axis=0) if len(X[np.where(clusters == i)]) > 0 else centroids[i] for i in range(k)])
#         print(new_centroids)
#         if np.linalg.norm(new_centroids-centroids)<1e-6:
#             break
#         centroids=new_centroids.copy()
#     return list(clusters)


# def kmeans(X, k, centroids):
#     clusters=np.zeros(len(X), dtype=int)
#     for _ in range(100):
#         distances=np.sum((X[:,np.newaxis]-centroids)**2, axis=2)
#         clusters=np.argmin(distances, axis=1)
#         new_centroids=np.array([X[clusters==i].mean(axis=0) if np.any(clusters==i) else centroids[i] for i in range(k)])
#         if np.sum((new_centroids-centroids)**2)<1e-10:
#             break
#         centroids=new_centroids
#     return clusters

# if __name__ == '__main__':
#     X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
#     k = 2
#     centroids = np.array([[1, 0], [4, 0]])
#     print("Res:", kmeans(X, k, centroids))