#!/usr/bin/env python3
"""
Signal power in dBm under counter-propagating Raman pump (undepleted pump).
Pump power is chosen to (approximately) compensate signal loss => ~flat Ps(z).
"""

import numpy as np
import matplotlib.pyplot as plt

# -------- Parameters --------
L_km = 20.0               # fiber length [km]
P_s0 = 1e-6               # signal input at z=0 [W]
alpha_s_dB_km = 0.20      # signal attenuation [dB/km]
alpha_p_dB_km = 0.25      # pump attenuation [dB/km]

g_R = 6.0e-14             # Raman gain coeff [m/W]
A_eff = 80e-12            # effective area [m^2]
rho_pol = 2/3             # polarization factor (≈2/3 for random)

margin = 1.00             # 1.00 => perfectly flat; >1 slight net gain, <1 slight net loss
num_points = 1500
save_path = "media/undepleted/raman_counterprop_profile.png"

# -------- Helpers & conversions --------
km = 1e3
L = L_km * km

def dB_per_km_to_np_per_m(val_db_km):
    return (val_db_km * np.log(10) / 10.0) / 1e3

alpha_s = dB_per_km_to_np_per_m(alpha_s_dB_km)  # [1/m]
alpha_p = dB_per_km_to_np_per_m(alpha_p_dB_km)  # [1/m]
g = rho_pol * g_R / A_eff                       # [1/W/m]

# Pump needed for Ps(L) = Ps(0) (flat overall gain):
#   -alpha_s L + (g * Pp_in / alpha_p) * (1 - e^{-alpha_p L}) = 0
Pp_in_flat = (alpha_s * L * alpha_p) / (g * (1.0 - np.exp(-alpha_p * L)))
Pp_in = margin * Pp_in_flat

z = np.linspace(0.0, L, num_points)

def Pp_counter(z):
    # Pump launched at z=L, propagates toward -z
    return Pp_in * np.exp(-alpha_p * (L - z))

def Ps_counter(z):
    # Closed-form solution for undepleted counter-prop pump
    integral = (g * Pp_in * np.exp(-alpha_p * L) / alpha_p) * (np.exp(alpha_p * z) - 1.0)
    return P_s0 * np.exp(-alpha_s * z) * np.exp(integral)

def to_dBm(Pw):
    return 10.0 * np.log10(np.maximum(Pw, 1e-30)) + 30.0

# -------- Compute --------
Ps_z = Ps_counter(z)
Ps_dBm = to_dBm(Ps_z)

# -------- Plot (dBm only) --------
plt.figure(figsize=(7, 4.0))
plt.plot(z / km, Ps_dBm, label="Signal $P_s(z)$ [dBm]")
plt.xlabel("Distance z [km]")
plt.ylabel("Signal power [dBm]")
plt.grid(True, which="both", alpha=0.3)

# Annotate parameters
txt = (
    fr"$L={L_km:.1f}\,\mathrm{{km}},\; P_s(0)={P_s0*1e3:.2f}\,\mathrm{{mW}}$" "\n"
    fr"$\alpha_s={alpha_s_dB_km:.2f}\,\mathrm{{dB/km}},\; \alpha_p={alpha_p_dB_km:.2f}\,\mathrm{{dB/km}}$" "\n"
    fr"$g_R={g_R:.2e}\,\mathrm{{m/W}},\; A_\mathrm{{eff}}={A_eff*1e12:.0f}\,\mathrm{{\mu m^2}},\; \rho={rho_pol:.2f}$" "\n"
    fr"$P_{{p,\mathrm{{in}}}}={Pp_in:.3f}\,\mathrm{{W}}\;(\mathrm{{margin}}={margin:.2f})$"
)
plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
plt.show()

print(f"Chosen pump for flat profile: P_p,in ≈ {Pp_in:.4f} W (margin={margin})")
print(f"P_s(0) = {to_dBm(P_s0):.2f} dBm,  P_s(L) = {to_dBm(Ps_z[-1]):.2f} dBm")
