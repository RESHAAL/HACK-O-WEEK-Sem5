import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [2],
    [4],
    [5],
    [7],
    [8],
    [9]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

model = LogisticRegression()

model.fit(X, y)

prediction = model.predict([[6]])

print("Prediction:", prediction[0])