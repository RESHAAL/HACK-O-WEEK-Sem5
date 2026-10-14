from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = LogisticRegression(penalty="l2", max_iter=500)

model.fit(X, y)

print("Accuracy:", model.score(X, y))