import pandas as pd

data = pd.DataFrame({
    "Height": [170, 165, 180],
    "Weight": [70, 60, 80]
})

# Create new feature: BMI
data["BMI"] = data["Weight"] / (data["Height"] / 100) ** 2

print(data)