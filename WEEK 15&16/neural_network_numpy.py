import numpy as np

np.random.seed(42)

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)

# Initialize weights and biases
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

for epoch in range(epochs):

    # Forward propagation
    z1 = X @ W1 + b1
    a1 = np.tanh(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Binary cross-entropy loss
    eps = 1e-8
    loss = -np.mean(
        y * np.log(a2 + eps) +
        (1 - y) * np.log(1 - a2 + eps)
    )

    # Backpropagation
    dz2 = (a2 - y) / len(X)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update parameters using gradient descent
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 2000 == 0:
        print("Epoch:", epoch, "Loss:", round(loss, 4))

# Final predictions
z1 = X @ W1 + b1
a1 = np.tanh(z1)
predictions = sigmoid(a1 @ W2 + b2)

print("\nPredictions:")
print(np.round(predictions, 3))
print("\nPredicted Classes:")
print((predictions >= 0.5).astype(int))