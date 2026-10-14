import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Prediction:", loaded_model.predict([X[0]]))
print("Model loaded successfully!")