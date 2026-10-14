import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 5, 10, 17, 26])

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Value:", prediction[0])