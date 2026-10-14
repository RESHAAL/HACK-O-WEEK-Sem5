x = 2

# Forward pass
z = x * 3
y = z ** 2

print("z =", z)
print("y =", y)

# Backward pass
dy_dz = 2 * z
dz_dx = 3

dy_dx = dy_dz * dz_dx

print("Gradient =", dy_dx)