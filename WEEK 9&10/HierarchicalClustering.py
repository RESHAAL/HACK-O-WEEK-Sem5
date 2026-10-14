from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

print("Cluster Labels:")
print(labels)