from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

print("Cluster Labels:")
print(model.labels_)