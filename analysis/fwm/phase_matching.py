import numpy as np
from scipy.optimize import fsolve, root
from analysis.fiber_analysis.load_fiber_values import load_phase_delay
from matplotlib import pyplot as plt
from numpy import polyval
import cvxpy as cp
from itertools import product
import analysis.utils.cfg as cfg

def get_plane(k, p, m):
    """Compute plane coefficients enforcing p·k conservation for a four-mode FWM tuple."""
    a = p[0]*(k[m[3], 1]-k[m[0], 1])
    b = p[0]*(k[m[3], 1]-k[m[1], 1])
    c = p[0]*(k[m[3], 1]-k[m[2], 1])
    d = p[0]*k[m[0], 2] + p[1]*k[m[1], 2] + p[2]*k[m[2], 2] - k[m[3], 2]
    return a, b, c, d

def main():
    """Plot beta0 polynomials and flag FWM planes intersecting frequency cube."""
    k = load_phase_delay() # TODO check the conversion
    omin = 1e12
    omax = 2e12
    freq = np.linspace(omin, omax, 1000)  # Frequency range in Hz
    modes = range(4)  # Modes to consider
    for m in modes:
        plt.plot(freq, polyval(k[m, :], freq), label=f'Mode {m}')
    plt.savefig('media/test/beta0.png')

    # Fixed permutation to (1, 1, -1)
    # Fixed modes (1, 2, 3 ,2)
    ## calculate the phase velocity

    p = [1, 1, -1]
    m = [1, 2, 3, 2]
    mode_names = ['lp01', 'lp11', 'lp21', 'lp02']
    p_range = [[1, 1, -1], 
               [1, -1, 1], 
               [-1, 1, 1]]
    mode_combinations = list(product(modes, repeat=4))
    flagged = [] # modes, p values where the plane intersects the range

    x = cp.Variable(3)
    for p in p_range:
        for m in mode_combinations:
            # Plane coefficients
            min_val = omin
            max_val = omax

            a, b, c, d = get_plane(k, p, m)        
            norm = np.sqrt(a**2 + b**2 + c**2)
            if norm == 0:
                print(f"{m} {p} degenerate")
                continue
            # Normalize coefficients
            constraints = [
                a*x[0] + b*x[1] + c*x[2] + d == 0,  # plane
                x >= min_val,
                x <= max_val
            ]

            prob = cp.Problem(cp.Minimize(0), constraints)
            result = prob.solve()
            print(f"{m} {p} ", end='')
            if prob.status == cp.OPTIMAL or prob.status != cp.INFEASIBLE:
                print(f"Y : {x.value}")
                flagged.append((m, p))
            else:
                print("N")
                
    print(flagged)
    N = 400
    x = np.linspace(omin, omax, N)
    y = np.linspace(omin, omax, N)
    z = np.linspace(omin, omax, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    print("Number of FWM relevant modalities:", len(flagged))
    for f in flagged:
        print(f"Modes: {f[0]}, p: {f[1]}")
        p = f[1]
        m = f[0]
        a, b, c, d = get_plane(k, p, m)        
        norm = np.sqrt(a**2 + b**2 + c**2)
        print(norm)
        distances = np.abs(a * X + b * Y + c * Z + d) / norm


if __name__ == "__main__":
    main()
    # print(distances)
