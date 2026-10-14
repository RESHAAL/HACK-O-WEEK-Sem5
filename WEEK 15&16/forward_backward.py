import numpy as np

# Input and initial parameters
x = 2.0
w = 0.5
b = 0.1
target = 1.0
learning_rate = 0.1

# Forward propagation
z = w * x + b
prediction = 1 / (1 + np.exp(-z))

# Loss: Mean Squared Error for one sample
loss = (prediction - target) ** 2

# Backpropagation
dz = 2 * (prediction - target) * prediction * (1 - prediction)
dw = dz * x
db = dz

# Update weights and bias
w = w - learning_rate * dw
b = b - learning_rate * db

print("Prediction:", prediction)
print("Loss:", loss)
print("Updated Weight:", w)
print("Updated Bias:", b)