from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

model = LogisticRegression(max_iter=5000)

scores = cross_val_score(model, X, y, cv=5)

print("CV Scores:", scores)
print("Average Score:", scores.mean())