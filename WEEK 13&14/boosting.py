from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

print("Accuracy:", model.score(X, y))