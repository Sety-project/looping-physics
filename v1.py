import itertools
import numpy as np
import math
import sys

# ----------------------------
# Defaults (from your request)
# ----------------------------
p = dict(
    epsilon=0.2,
    r=0.2,
    l=10.0,
    APY_l=0.15,
    APY_d=0.25,
    APY_v=0.75,
    r_b=0.12,
    R=8_000_000.0,
    r_star=0.0,
    N=20_000_000.0,
    U=0.9,
    alpha=0.5
)

# small tolerance for near-zero checks
EPS = 1e-12

# ----------------------------
# Helper: robust divide with check
# ----------------------------
def safe_div(numer, denom, name):
    if abs(denom) <= EPS:
        raise ZeroDivisionError(f"Denominator near-zero for '{name}': denom = {denom}")
    return numer / denom

# ----------------------------
# Compute intermediate denominators and check them
# ----------------------------
den_A_l = p["APY_l"] - p["r_b"] * (1 - p["epsilon"])
den_A_d = p["APY_d"] - p["r"] / 2
den_A_v = p["APY_v"] - p["l"] * p["r"] + (p["l"] - 1) * p["r_b"]

# checks
for name, val in [("den_A_l", den_A_l), ("den_A_d", den_A_d), ("den_A_v", den_A_v)]:
    if abs(val) <= EPS:
        raise ZeroDivisionError(f"{name} is zero or too small: {val}")

# compute intermediates
A_l = safe_div(1.0, den_A_l, "A_l")
A_d = safe_div(1.0, den_A_d, "A_d")
A_v = safe_div(1.0, den_A_v, "A_v")

B_l = 1.0 + (p["r_b"] * (1 - p["epsilon"])) / den_A_l
B_d = 1.0 + (p["r"] / 2) / den_A_d

C = (p["l"] - 1.0) * (1.0 / p["U"] + 1.0 / p["alpha"])

# D as given
D = 1.0 - ((p["l"] - 1.0) * p["r_b"] * (1 - p["epsilon"])) / (den_A_v * p["U"]) \
      - ((p["l"] - 1.0) * p["r"]) / (2.0 * p["alpha"] * den_A_v)

if abs(D) <= EPS:
    raise ZeroDivisionError(f"D is zero or too small: D = {D}")

# ----------------------------
# Coefficients (exactly per LaTeX)
# ----------------------------
alpha_l = A_l + A_v * B_l / D
alpha_d = A_d + A_v * B_d / D
alpha_0 = A_v * p["R"] / D

gamma_l = -A_l + A_v * C * B_l / D
gamma_d = -A_d + A_v * C * B_d / D
gamma_0 = A_v * C * p["R"] / D

delta_l = (p["U"] * den_A_v) / ((p["l"] - 1.0) * den_A_l) - B_l / D
delta_d = -B_d / D
delta_0 = p["R"] / D

epsilon_l = -B_l / D
epsilon_d = (p["alpha"] * den_A_v) / ((p["l"] - 1.0) * den_A_d) - B_d / D
epsilon_0 = p["R"] / D


# ----------------------------
# (Optional) quick feasible-vertex search as before
# ----------------------------
constraints = [
    (gamma_l, gamma_d, p["N"] + gamma_0),
    (delta_l, delta_d, delta_0),
    (epsilon_l, epsilon_d, epsilon_0)
]

# x>=0, y>=0 lines
lines = constraints + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

def intersect(c1, c2):
    a1, b1, c1rhs = c1
    a2, b2, c2rhs = c2
    det = a1 * b2 - a2 * b1
    if abs(det) <= EPS:
        return None
    x = (c1rhs * b2 - c2rhs * b1) / det
    y = (a1 * c2rhs - a2 * c1rhs) / det
    return np.array([x, y], dtype=float)

vertices = []
for c1, c2 in itertools.combinations(lines, 2):
    pt = intersect(c1, c2)
    if pt is None:
        continue
    if np.any(pt < -1e-9):
        continue
    R_l, R_d = pt
    if all(a * R_l + b * R_d <= c + 1e-8 for a, b, c in constraints):
        vertices.append(pt)

if not vertices:
    print("Warning: no feasible vertices found with given parameters.")
else:
    vertices = np.unique(np.round(np.array(vertices), 10), axis=0)
    def TVL(x,y): return alpha_l*x + alpha_d*y + alpha_0
    vals = [TVL(x,y) for x,y in vertices]
    idx = int(np.argmax(vals))
    print("Feasible vertices found:", len(vertices))
    for v,val in zip(vertices, vals):
        print(f"  R_l={v[0]:.6g}, R_d={v[1]:.6g} -> TVL={val:.6g}")
    print("Optimal approx at vertex:", vertices[idx], "TVL=", vals[idx])

