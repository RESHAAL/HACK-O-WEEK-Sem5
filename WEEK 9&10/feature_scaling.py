import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "Age": [20, 25, None, 30],
    "Salary": [25000, None, 40000, 50000]
})

# Fill missing values
imputer = SimpleImputer(strategy="mean")
data[["Age", "Salary"]] = imputer.fit_transform(data[["Age", "Salary"]])

# Feature scaling
scaler = StandardScaler()
data[["Age", "Salary"]] = scaler.fit_transform(data[["Age", "Salary"]])

print(data)