import numpy as np

x = np.array([-2, -1, 0, 1, 2])

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# ReLU
relu = np.maximum(0, x)

# Tanh
tanh = np.tanh(x)

print("Sigmoid:", sigmoid)
print("ReLU:", relu)
print("Tanh:", tanh)