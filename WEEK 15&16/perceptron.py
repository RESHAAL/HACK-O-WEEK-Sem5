import numpy as np

inputs = np.array([1, 0])
weights = np.array([0.5, 0.4])
bias = -0.2

# Weighted sum
z = np.dot(inputs, weights) + bias

# Step activation function
output = 1 if z >= 0 else 0

print("Weighted Sum:", z)
print("Perceptron Output:", output)