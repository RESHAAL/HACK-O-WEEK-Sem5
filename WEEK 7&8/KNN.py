import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 7],
    [7, 8],
    [8, 8]
])

y = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X, y)

prediction = model.predict([[4, 5]])

print("Predicted Class:", prediction[0])