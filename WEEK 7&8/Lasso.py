import numpy as np
from sklearn.linear_model import Lasso

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y = np.array([10, 15, 20, 25, 30])

model = Lasso(alpha=0.1)

model.fit(X, y)

prediction = model.predict([[6, 7]])

print("Prediction:", prediction[0])
print("Coefficients:", model.coef_)