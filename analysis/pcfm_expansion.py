import numpy as np
import matplotlib.pyplot as plt

# Fixed detuning
delta_f = 1.0

# Small B/f regime
B = np.linspace(1e-4, 2, 2000)  # since delta_f=1, this is also B/delta_f

# Exact function
exact = np.log(np.abs((delta_f + B/2) / (delta_f - B/2)))

# Asymptotic orders for B/delta_f << 1
order1 = B / delta_f
order3 = B / delta_f + (B**3) / (12 * delta_f**3)
order5 = order3 + (B**5) / (80 * delta_f**5)

# Plot exact and approximations
plt.figure()
plt.plot(B / delta_f, exact, label="exact")
plt.plot(B / delta_f, order1, "--", label=r"$O(B/\Delta f)$")
plt.plot(B / delta_f, order3, "--", label=r"$O((B/\Delta f)^3)$")
plt.plot(B / delta_f, order5, "--", label=r"$O((B/\Delta f)^5)$")

plt.xlabel(r"$B/\Delta f$")
plt.ylabel(r"$\log\left|\frac{\Delta f+B/2}{\Delta f-B/2}\right|$")
plt.title("Exact function and low-$B/\\Delta f$ asymptotic orders")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig("low_B_over_f_expansion.png", dpi=200)
plt.close()

# Optional: relative error plot
rel1 = np.abs(order1 - exact) / np.abs(exact)
rel3 = np.abs(order3 - exact) / np.abs(exact)
rel5 = np.abs(order5 - exact) / np.abs(exact)

plt.figure()
plt.plot(B / delta_f, rel1, label=r"$O(B/\Delta f)$")
plt.plot(B / delta_f, rel3, label=r"$O((B/\Delta f)^3)$")
plt.plot(B / delta_f, rel5, label=r"$O((B/\Delta f)^5)$")

plt.yscale("log")
plt.xlabel(r"$B/\Delta f$")
plt.ylabel("relative error")
plt.title("Relative error of low-$B/\\Delta f$ asymptotic orders")
plt.legend()
plt.ylim(1e-2, 0.2)
plt.axvline(1, color="gray", linestyle="--", label=r"$B/\Delta f=1$")
# plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("low_B_over_f_relative_error.png", dpi=200)
print("Plots saved: low_B_over_f_expansion.png and low_B_over_f_relative_error.png")
plt.close()