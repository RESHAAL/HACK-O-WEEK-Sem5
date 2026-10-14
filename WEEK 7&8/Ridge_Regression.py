import numpy as np
from sklearn.linear_model import Ridge

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

y = np.array([10, 15, 20, 25, 30])

model = Ridge(alpha=1.0)

model.fit(X, y)

prediction = model.predict([[6, 7]])

print("Prediction:", prediction[0])
print("Coefficients:", model.coef_)