import numpy as np

# Actual and predicted values
y_true = np.array([1.0, 0.0])
y_pred = np.array([0.8, 0.2])

# Mean Squared Error
mse = np.mean((y_true - y_pred) ** 2)
print("MSE Loss:", mse)

# Binary Cross-Entropy
eps = 1e-7
p = np.clip(y_pred, eps, 1 - eps)

bce = -np.mean(
    y_true * np.log(p) + (1 - y_true) * np.log(1 - p)
)
print("Binary Cross-Entropy:", bce)

# SGD update
weight = 1.0
gradient = 0.5
learning_rate = 0.1

weight = weight - learning_rate * gradient
print("Weight after SGD:", weight)

# Adam optimizer: one simplified update step
weight = 1.0
gradient = 0.5
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
learning_rate = 0.1

m = (1 - beta1) * gradient
v = (1 - beta2) * gradient ** 2

m_hat = m / (1 - beta1)
v_hat = v / (1 - beta2)

weight = weight - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
print("Weight after Adam:", weight)