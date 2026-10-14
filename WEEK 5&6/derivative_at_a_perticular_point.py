import sympy as sp

x = sp.symbols('x')

f = x**2 + 3*x + 2

derivative = sp.diff(f, x)

value = derivative.subs(x, 2)

print("Derivative:", derivative)
print("Derivative at x=2:", value)