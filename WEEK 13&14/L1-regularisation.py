from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

X, y = load_iris(return_X_y=True)

model = OneVsRestClassifier(
    LogisticRegression(penalty="l1", solver="liblinear")
)

model.fit(X, y)

print("Accuracy:", model.score(X, y))