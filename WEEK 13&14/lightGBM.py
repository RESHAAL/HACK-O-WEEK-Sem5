from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = LGBMClassifier(n_estimators=50, verbosity=-1)
model.fit(X, y)

print("Accuracy:", model.score(X, y))