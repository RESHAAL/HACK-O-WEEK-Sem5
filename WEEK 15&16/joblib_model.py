import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save model
joblib.dump(model, "model.joblib")

# Load model
loaded_model = joblib.load("model.joblib")

print("Prediction:", loaded_model.predict([X[0]]))
print("Model loaded successfully!")