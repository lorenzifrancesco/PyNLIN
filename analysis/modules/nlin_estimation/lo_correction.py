import matplotlib.pyplot as plt
import os
from analysis.modules.collision import build_I_low_interpolator, MAX_LLD
import analysis.modules.cfg as cfg
import numpy as np
from typing import Tuple
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
from loguru import logger as lg
from analysis.modules.log_init import init_logging
init_logging()
from analysis.modules.nlin_estimation.raman_integrals import load_fB, raman_integral, load_raman_integral_extremes
from analysis.modules.nlin_estimation.ideal_fits import ideal_fit_coefficients
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, zoomed_inset_axes

SPATIAL_MODES = np.array([1, 2, 2, 1])
LLW_MIN = 0.01  # target L/LW
LLW_MAX = 100.0
# 64-QAM <|b_0|^4>/<|b_0|^2>^2 this is compatible with the (mu_0 - 1)=0.32*1.19 previously used.
MU0 = 1.3809


def build_lookup_integral_table_with_raman_custom(cf,
                                                  fB: callable,
                                                  m_lo_truncation: int = 2,
                                                  ipulse: int = 1,
                                                  recompute=False,
                                                  index=0) -> Tuple[callable, callable]:
    # sampling the gvda, gvdb space, build the callable function
    # giving the correction integrals for fB_max and fB_min: integral(L/gvda, L/gvdb).
    _, _, _, fB_min, fB_max = load_fB(cf)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld = np.linspace(1e-30, MAX_LLD, n_samples)
    ld = fiber_length / lld
    lg.debug(
        f"Useful range of L/LD from LO time integral data: {lld[0]:.2e} to {lld[-1]:.2e}")
    raman_correction_grid = np.zeros((n_samples, n_samples))
    filename = f"results/raman_correction_grid_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}_customfB{index}.npy"

    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed Raman correction grid from {filename}")
        data = np.load(filename, allow_pickle=True).item()
        raman_correction_grid = data['raman_correction_grid']
    else:
        lg.info(f"Computing Raman correction grid and saving to {filename}")
        for m_lo in range(m_lo_truncation+1):
            add = np.zeros((n_samples, n_samples))
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    # apply symmetry:
                    if idb < ida:
                        add[ida, idb] = add[idb, ida]
                    else:
                        lg.debug(
                            f"Point {ida*n_samples+idb+1}/{n_samples*n_samples}, spanning LLDA={cf.fiber_length/lda:.2e}, LLDB={cf.fiber_length/ldb:.2e}")
                        # this is also in normalized units
                        def I_specific(x): return interp(x/lda, x/ldb)
                        # compute the integral
                        add[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB(x), 0, fiber_length)[0] / fiber_length)**2
                        lg.trace(
                            f"Contribution of the m_lo={m_lo} integral: {add[ida, idb]:.2e}")
            if m_lo != 0:
                add *= 2
            raman_correction_grid += add
        np.save(filename, {
            'raman_correction_grid': raman_correction_grid,
        })

    inter_func = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid,
        bounds_error=False,
        fill_value=None)

    def func_wrapper(func):
        def func_wrapper(x, y):
            assert (x <= 1.01 * lld[-1] and y <= 1.01 * lld[-1]
                    ), f"Input {x} exceeds the 110% of the interpolation range [{lld[0]}, {lld[-1]}]"
            assert (x >= 0 and y >=
                    0), f"Input has negative values check that your LD is positive"
            return func((x, y))
        return func_wrapper
    return func_wrapper(inter_func)


def build_lookup_integral_table_with_raman(cf,
                                           m_lo_truncation: int = 2,
                                           ipulse: int = 1,
                                           recompute=False) -> Tuple[callable, callable]:
    # sampling the gvda, gvdb space, build the callable function
    # giving the correction integrals for fB_max and fB_min: integral(L/gvda, L/gvdb).
    _, _, _, fB_min, fB_max = load_fB(cf)
    n_samples = 20
    fiber_length = cf.fiber_length
    lld = np.linspace(1e-30, MAX_LLD, n_samples)
    ld = fiber_length / lld
    lg.debug(
        f"Useful range of L/LD from LO time integral data: {lld[0]:.2e} to {lld[-1]:.2e}")
    raman_correction_grid_max = np.zeros((n_samples, n_samples))
    raman_correction_grid_min = np.zeros((n_samples, n_samples))

    # plot max and min fB
    # exit()
    # save to file with exhaustive namefile information
    filename = f"results/raman_correction_grid_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.npy"

    if os.path.exists(filename) and not recompute:
        lg.info(f"Loading precomputed Raman correction grid from {filename}")
        data = np.load(filename, allow_pickle=True).item()
        raman_correction_grid_max = data['raman_correction_grid_max']
        raman_correction_grid_min = data['raman_correction_grid_min']
    else:
        lg.info(f"Computing Raman correction grid and saving to {filename}")
        for m_lo in range(m_lo_truncation+1):
            add_min = np.zeros((n_samples, n_samples))
            add_max = np.zeros((n_samples, n_samples))
            lg.info(f"Calculating m_lo={m_lo}")
            I_low_dataset = np.load(
                f"results/I_low_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo}.npz")
            interp = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
            for ida, lda in enumerate(ld):
                for idb, ldb in enumerate(ld):
                    # apply symmetry:
                    if idb < ida:
                        add_max[ida, idb] = add_max[idb, ida]
                        add_min[ida, idb] = add_min[idb, ida]
                    else:
                        lg.debug(
                            f"Point {ida*n_samples+idb+1}/{n_samples*n_samples}, spanning LLDA={cf.fiber_length/lda:.2e}, LLDB={cf.fiber_length/ldb:.2e}")
                        # this is also in normalized units
                        def I_specific(x): return interp(x/lda, x/ldb)
                        # compute the integral
                        add_max[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_max(x), 0, fiber_length)[0] / fiber_length)**2
                        add_min[ida, idb] += (
                            quad(lambda x: I_specific(x) * fB_min(x), 0, fiber_length)[0] / fiber_length)**2
                        lg.trace(
                            f"Contribution of the m_lo={m_lo} integral: max       {add_max[ida, idb]:.2e}, min {add_min[ida, idb]:.2e}")
            if m_lo != 0:
                add_max *= 2
                add_min *= 2
            raman_correction_grid_max += add_max
            raman_correction_grid_min += add_min

            plt.figure(figsize=(4, 4))
            plt.imshow(raman_correction_grid_max, extent=(
                lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
            plt.colorbar()
            plt.title(
                f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
            plt.xlabel("L/LDa")
            plt.ylabel("L/LD b")
            plt.savefig(
                f"results/raman_correction_grid_max_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
            plt.close()
            plt.figure(figsize=(4, 4))
            plt.imshow(raman_correction_grid_min, extent=(
                lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
            plt.colorbar()
            plt.title(
                f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
            plt.xlabel("L/LDa")
            plt.ylabel("L/LD b")
            plt.savefig(
                f"media/debug/raman_correction_grid_min_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
            plt.close()
            # raman_correction_grid_max = np.zeros((n_samples, n_samples))
            # raman_correction_grid_min = np.zeros((n_samples, n_samples))
        np.save(filename, {
            'raman_correction_grid_max': raman_correction_grid_max,
            'raman_correction_grid_min': raman_correction_grid_min,
        })

    # build the interpolator and return it
    # checking
    plt.figure(figsize=(4, 4))
    plt.imshow(raman_correction_grid_max, extent=(
        lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
    plt.colorbar()
    plt.title(
        f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
    plt.xlabel("L/LDa")
    plt.ylabel("L/LD b")
    plt.savefig(
        f"results/raman_correction_grid_max_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
    plt.close()
    plt.figure(figsize=(4, 4))
    plt.imshow(raman_correction_grid_min, extent=(
        lld[0], lld[-1], lld[0], lld[-1]), origin='lower')
    plt.colorbar()
    plt.title(
        f"Raman correction grid max, m_lo={m_lo_truncation}, {cf.fiber_length/1e3:.1f} km")
    plt.xlabel("L/LDa")
    plt.ylabel("L/LD b")
    plt.savefig(
        f"media/debug/raman_correction_grid_min_{'gaussian' if ipulse == 0 else 'nyquist'}_m{m_lo_truncation}_n{n_samples}_L{fiber_length/1e3:.1f}km_lld{lld[-1]:.2f}.png", dpi=300)
    plt.close()

    interp_func_max = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_max,
        bounds_error=False,
        fill_value=None)

    interp_func_min = RegularGridInterpolator(
        (lld, lld),
        raman_correction_grid_min,
        bounds_error=False,
        fill_value=None)

    def func_wrapper(func):
        def func_wrapper(x, y):
            assert (x <= 1.01 * lld[-1] and y <= 1.01 * lld[-1]
                    ), f"Input {x} exceeds the 110% of the interpolation range [{lld[0]}, {lld[-1]}]"
            assert (x >= 0 and y >=
                    0), f"Input has negative values check that your LD is positive"
            return func((x, y))
        return func_wrapper

    return func_wrapper(interp_func_min), func_wrapper(interp_func_max)


def  validate_maxmin_interpolation(cf):
    lldas = np.linspace(0, 2.3, 20)
    ps_ideal = ideal_fit_coefficients(0.0, 0.0)
    raman_gvd_correction_min, raman_gvd_correction_max = build_lookup_integral_table_with_raman(
        cf, recompute=False)
    fB, fB_min, fB_max, _, _ = load_fB(cf)
    
    n_samples = 10
    custom_raman_correction = []
    z_axis = np.linspace(0, cf.fiber_length, len(fB_min))
    for i in range(n_samples):
        #linear interpolation of the fB extremes
        fB_custom = fB_min + (fB_max - fB_min) * i / 9.0 # linear combination
        coeffs = np.polyfit(z_axis, fB_custom, 6)
        def fB_custom_function(z):
            return np.polyval(coeffs, z)
        # build interpolation
        custom_raman_correction.append(build_lookup_integral_table_with_raman_custom(
            cf, recompute=False, fB=fB_custom_function, index=i))
        
    # get samples on the LLDA, LLDB grid
    r_lo_min, r_lo_max, _, _ = load_raman_integral_extremes(cf)
    raman_integral_fB_lo = raman_integral(cf, "LO", fB_custom)
        
    # assume: lldas is 1D of length M
    M = len(lldas)
    lo_value_fB_custom = np.empty((M, M, n_samples), dtype=float)
    lo_value_fB_maxmin = np.empty((M, M, n_samples), dtype=float)
    for i in range(n_samples):
        fB_custom = fB_min + (fB_max - fB_min) * i / 9.0
        # rows correspond to second arg (lldas[:,None]),
        # cols correspond to first arg (lldas[None,:])
        for r in range(M):               # over lldas as the SECOND argument
            y = lldas[r]
            for c in range(M):           # over lldas as the FIRST argument
                x = lldas[c]
                lo_value_fB_custom[r, c, i] = custom_raman_correction[i](x, y)
            
                lo_value_max = raman_gvd_correction_max(lldas[c], lldas[r])
                lo_value_min = raman_gvd_correction_min(lldas[c], lldas[r])
                raman_integral_fB_lo = raman_integral(cf, "LO", fB_custom)
                lo_value_fB_maxmin[r, c, i] = (raman_integral_fB_lo - r_lo_min) / (r_lo_max - r_lo_min) * \
                    (lo_value_max - lo_value_min) + lo_value_min

    lg.debug(f"some values from custom: {lo_value_fB_custom[0,0,0]}, {lo_value_fB_custom[M//2,M//2,0]}, {lo_value_fB_custom[-1,-1,0]}")
    lg.debug(f"some values from maxmin: {lo_value_fB_maxmin[0,0, 0]}, {lo_value_fB_maxmin[M//2,M//2, 0]}, {lo_value_fB_maxmin[-1,-1,0]}") 
    # for i in range(n_samples):
    #     lo_value_fB_custom = my_raman_correction[i](
    #         lldas[None, :], lldas[:, None])
    
    # Choose a representative slice along one LLDA direction
    r_index = M // 2  # middle slice (fix y = lldas[r_index])
    alphas = np.linspace(0, 1, n_samples)
    x_values = alphas  # vary x = lldas[c]

    # plotting together
    # --- Combined side-by-side figure ---
    fig, axes = plt.subplots(2, 1, figsize=(3, 6))

    # Left: Raman correction vs α
    ax1 = axes[0]
    skipping = 4
    for i in range(len(lldas)//skipping):
        color = plt.cm.jet(1 - (skipping * i / (len(lldas) - 1)))
        ax1.plot(
            np.linspace(0, 1, n_samples),
            lo_value_fB_custom[r_index, skipping * i, :],
            lw=0.7,
            color=color,
            label=fr"{lldas[skipping*i]:.1f}"
        )
        ax1.plot(
            np.linspace(0, 1, n_samples),
            lo_value_fB_maxmin[r_index, skipping * i, :],
            'k--',
            lw=0.5
        )
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax1.text(0.89, 0.01, fr"(a)"),
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\overline{\mathcal{P}}_B^{\mathrm{LO}}$")
    ax1.legend(fontsize=8, loc="upper left", title=r"$L/L_{{DB}}$")

    # Right: scatter comparison
    ax2 = axes[1]
    for i in range(n_samples):
        ax2.scatter(
            lo_value_fB_custom[:, :, i].flatten(),
            lo_value_fB_maxmin[:, :, i].flatten(),
            s=17,
            facecolors='none',
            lw=0.3,
            edgecolors=plt.cm.ocean(1 - (i / (n_samples - 1))),
            alpha=0.3
        )

    ax2.plot(
        [0, np.max(lo_value_fB_maxmin)],
        [0, np.max(lo_value_fB_maxmin)],
        'k-', lw=0.5,
        color='red'
    )
    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(rf"$\overline{{\mathcal{{P}}}}_B^{{\mathrm{{LO}}}}$")
    ax2.set_ylabel(rf"approx. $\overline{{\mathcal{{P}}}}_B^{{\mathrm{{LO}}}}$")
    # ax2.text(0.0, np.max(lo_value_fB_custom[:, :])*0.96, fr"(b)"),
    ax2.text(0.089, 0.0, fr"(b)"),

    plt.tight_layout()
    plt.savefig("media/raman-validation.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02)
    lg.info("Saved to media/raman-validation.pdf")
    # ax1
    extent1 = ax1.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("media/raman-validation-left.pdf", bbox_inches=extent1, dpi=300)
    lg.info("Saved to media/raman-validation-left.pdf")
    # ax2
    extent2 = ax2.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("media/raman-validation-right.pdf", bbox_inches=extent2, dpi=300)
    lg.info("Saved to media/raman-validation-right.pdf")
    plt.close(fig)
    
    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))

    # === Left panel content (your original ax1) ===
    skipping = 4
    for i in range(len(lldas)//skipping):
        color = plt.cm.jet(1 - (skipping * i / (len(lldas) - 1)))
        ax1.plot(
            np.linspace(0, 1, n_samples),
            lo_value_fB_custom[r_index, skipping * i, :],
            lw=0.7,
            color=color,
            label=fr"{lldas[skipping*i]:.1f}"
        )
        ax1.plot(
            np.linspace(0, 1, n_samples),
            lo_value_fB_maxmin[r_index, skipping * i, :],
            'k--',
            lw=0.5
        )

    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\overline{\mathcal{P}}_B(\alpha)$")
    ax1.legend(fontsize=8, loc="upper left", title=r"$L/L_{\mathrm{DB}}$")
    ax1.text(0.95, 0.02, "(a)", ha='right', va='bottom')

    # === Inset axis (your original ax2) ===
    # Position is in figure fraction coordinates relative to ax1
    ax2 = inset_axes(ax1, width="45%", height="45%", loc="lower right",
                    borderpad=1.2)  # adjust width/height/loc as needed

    # --- Scatter comparison (original ax2 content) ---
    for i in range(n_samples):
        ax2.scatter(
            lo_value_fB_custom[:, :, i].flatten(),
            lo_value_fB_maxmin[:, :, i].flatten(),
            s=7, facecolors='none', lw=0.3,
            edgecolors=plt.cm.ocean(1 - (i / (n_samples - 1))),
            alpha=0.3
        )

    ax2.plot(
        [0, np.max(lo_value_fB_maxmin)],
        [0, np.max(lo_value_fB_maxmin)],
        color='red', lw=0.5
    )
    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel(rf"$\overline{{\mathcal{{P}}}}_B$", fontsize=7)
    ax2.set_ylabel(rf"${{\mathcal{{P}}}}_B$", fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.text(0.05, 0.05, "(b)", transform=ax2.transAxes, fontsize=7)

    # --- Save ---
    plt.tight_layout()
    plt.savefig("media/raman-validation-inset.pdf",
                dpi=300, bbox_inches='tight', pad_inches=0.01)
    lg.info("Saved to media/raman-validation-inset.pdf")
    plt.close(fig)
    
    
    
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    # --- Compute difference (residual) ---
    diff = lo_value_fB_maxmin - lo_value_fB_custom

    # --- Scatter all results across all parameters ---
    for i in range(n_samples):
        ax.scatter(
            lo_value_fB_custom[:, :, i].flatten(),   # x-axis: actual P_B
            diff[:, :, i].flatten()/lo_value_fB_custom[:, :, i].flatten(),                 # y-axis: difference (approx - actual)
            s=7, facecolors='none', lw=0.3,
            edgecolors=plt.cm.ocean(1 - (i / (n_samples - 1))),
            alpha=0.3
        )

    # --- Reference line y = 0 ---
    ax.axhline(0, color='red', lw=0.5)

    # --- Labels and formatting ---
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    ax.set_xlabel(r"$\overline{\mathcal{P}}_B$")
    ax.set_ylabel(r"$\Delta\overline{\mathcal{P}}_B$")
    # ax.text(0.05, 0.92, "(c)", transform=ax.transAxes, fontsize=9)
    ax.set_aspect('auto', adjustable='box')

    plt.tight_layout()
    plt.savefig("media/raman-validation-residuals.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
    lg.info("Saved to media/raman-validation-residuals.pdf")
    plt.close(fig)

        

if __name__ == "__main__":
    validate_maxmin_interpolation(cfg.load_toml_to_struct("./input/mmf.toml")) 
    exit()