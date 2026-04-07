from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("media/pcfm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed detuning
delta_f = 1.0

# Small B/f regime, expressed through Delta f / B in [1, 1e3].
delta_f_over_B = np.logspace(0, 3, 2000)
B = delta_f / delta_f_over_B

# Exact function
exact = np.log(np.abs((delta_f + B/2) / (delta_f - B/2)))

# Asymptotic orders for B/delta_f << 1
order1 = B / delta_f
order3 = B / delta_f + (B**3) / (12 * delta_f**3)
order5 = order3 + (B**5) / (80 * delta_f**5)
b_over_delta_f = B / delta_f

# Original plots against B / Delta f.
plt.figure()
plt.plot(b_over_delta_f, exact, label="exact")
plt.plot(b_over_delta_f, order1, "--", label=r"$O(B/\Delta f)$")
plt.plot(b_over_delta_f, order3, "--", label=r"$O((B/\Delta f)^3)$")
plt.plot(b_over_delta_f, order5, "--", label=r"$O((B/\Delta f)^5)$")

plt.xlabel(r"$B/\Delta f$")
plt.ylabel(r"$\log\left|\frac{\Delta f+B/2}{\Delta f-B/2}\right|$")
plt.title("Exact function and low-$B/\\Delta f$ asymptotic orders")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "low_B_over_f_expansion.png", dpi=200)
plt.close()

# Plot against Delta f / B.
x = delta_f_over_B
exact_plot = exact
order1_plot = order1
order3_plot = order3
order5_plot = order5

# Plot exact and approximations on a linear scale.
plt.figure()
plt.plot(x, exact_plot, label="exact")
plt.plot(x, order1_plot, "--", label=r"$O(B/\Delta f)$")
plt.plot(x, order3_plot, "--", label=r"$O((B/\Delta f)^3)$")
plt.plot(x, order5_plot, "--", label=r"$O((B/\Delta f)^5)$")

plt.xlabel(r"$\Delta f / B$")
plt.ylabel(r"$\log\left|\frac{\Delta f+B/2}{\Delta f-B/2}\right|$")
plt.title(r"Low-$B/|\Delta f|$ expansion vs. $\Delta f / B$")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "low_B_over_f_expansion_vs_deltaf_over_B.png", dpi=200)
plt.close()

# Plot exact and approximations on log-log axes.
plt.figure()
plt.loglog(x, exact_plot, label="exact")
plt.loglog(x, order1_plot, "--", label=r"$O(B/\Delta f)$")
plt.loglog(x, order3_plot, "--", label=r"$O((B/\Delta f)^3)$")
plt.loglog(x, order5_plot, "--", label=r"$O((B/\Delta f)^5)$")

plt.xlabel(r"$\Delta f / B$")
plt.ylabel(r"$\log\left|\frac{\Delta f+B/2}{\Delta f-B/2}\right|$")
plt.title(r"Low-$B/|\Delta f|$ expansion vs. $\Delta f / B$ (log-log)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "low_B_over_f_expansion_vs_deltaf_over_B_loglog.png", dpi=200)
plt.close()

# Optional: relative error plot
rel1 = np.abs(order1 - exact) / np.abs(exact)
rel3 = np.abs(order3 - exact) / np.abs(exact)
rel5 = np.abs(order5 - exact) / np.abs(exact)
rel1_plot = rel1
rel3_plot = rel3
rel5_plot = rel5

# Original relative error plot against B / Delta f.
plt.figure()
plt.plot(b_over_delta_f, rel1, label=r"$O(B/\Delta f)$")
plt.plot(b_over_delta_f, rel3, label=r"$O((B/\Delta f)^3)$")
plt.plot(b_over_delta_f, rel5, label=r"$O((B/\Delta f)^5)$")

plt.yscale("log")
plt.xlabel(r"$B/\Delta f$")
plt.ylabel("relative error")
plt.title("Relative error of low-$B/\\Delta f$ asymptotic orders")
plt.legend()
plt.ylim(1e-2, 0.2)
plt.axvline(1, color="gray", linestyle="--", label=r"$B/\Delta f=1$")
# plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(OUT_DIR / "low_B_over_f_relative_error.png", dpi=200)
plt.close()

plt.figure()
plt.plot(x, rel1_plot, label=r"$O(B/\Delta f)$")
plt.plot(x, rel3_plot, label=r"$O((B/\Delta f)^3)$")
plt.plot(x, rel5_plot, label=r"$O((B/\Delta f)^5)$")

plt.yscale("log")
plt.xlabel(r"$\Delta f / B$")
plt.ylabel("relative error")
plt.title(r"Relative error of low-$B/|\Delta f|$ asymptotic orders")
plt.legend()
plt.ylim(1e-2, 0.2)
plt.axvline(1, color="gray", linestyle="--", label=r"$\Delta f / B=1$")
# plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(OUT_DIR / "low_B_over_f_relative_error_vs_deltaf_over_B.png", dpi=200)
print(
    "Plots saved: "
    f"{OUT_DIR / 'low_B_over_f_expansion.png'}, "
    f"{OUT_DIR / 'low_B_over_f_expansion_vs_deltaf_over_B.png'}, "
    f"{OUT_DIR / 'low_B_over_f_expansion_vs_deltaf_over_B_loglog.png'}, "
    f"{OUT_DIR / 'low_B_over_f_relative_error.png'}, "
    f"and {OUT_DIR / 'low_B_over_f_relative_error_vs_deltaf_over_B.png'}"
)
plt.close()
