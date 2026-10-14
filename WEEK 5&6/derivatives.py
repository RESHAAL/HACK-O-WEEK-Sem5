import sympy as sp

x = sp.symbols('x')

f = x**2

derivative = sp.diff(f, x)

print("Function:", f)
print("Derivative:", derivative)