import sympy as sp

x, y = sp.symbols('x y')

f = x**2 + y**2

gradient = [
    sp.diff(f, x),
    sp.diff(f, y)
]

print("Function:", f)
print("Gradient:", gradient)