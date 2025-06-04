import cvxpy as cp

# Plane coefficients
a, b, c, d = 1, -1, 1, -2
min_val = 1
max_val = 2

x = cp.Variable(3)

# Constraints
constraints = [
    a*x[0] + b*x[1] + c*x[2] + d == 0,  # plane
    x >= min_val,
    x <= max_val
]

# Problem: feasibility (no objective)
prob = cp.Problem(cp.Minimize(0), constraints)

result = prob.solve()

if prob.status == cp.OPTIMAL or prob.status != cp.INFEASIBLE:
    print("Plane intersects the cube.")
    print("Example intersection point:", x.value)
else:
    print("No intersection between plane and cube.")