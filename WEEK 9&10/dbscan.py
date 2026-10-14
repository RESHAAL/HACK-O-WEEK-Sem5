from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

print("Cluster Labels:")
print(labels)