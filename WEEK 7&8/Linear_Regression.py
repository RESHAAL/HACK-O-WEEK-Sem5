import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 40000, 45000, 50000])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Salary:", prediction[0])
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)