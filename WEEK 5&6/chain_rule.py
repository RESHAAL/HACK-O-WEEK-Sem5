import sympy as sp

x = sp.symbols('x')

y = (x**2 + 1)**3

derivative = sp.diff(y, x)

print("Function:", y)
print("Derivative:", derivative)