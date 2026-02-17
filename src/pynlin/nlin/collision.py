"""Collision-integral utilities and plotting helpers for NLIN workflows."""

from loguru import logger as lg

from pynlin.log_init import init_logging

init_logging()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import RegularGridInterpolator

import pynlin
from pynlin.fiber_data.beta_utils import (
    beta2avg_complementary,
    beta2rms,
    beta2rms_complementary,
)
from pynlin.fiber_data.load_fiber_values import load_group_delay, load_oi
from pynlin.fiber import MMFiber
from pynlin.nlin import m_th_time_integral_general
from pynlin.pulses import GaussianPulse, NyquistPulse
from pynlin.wdm import WDM

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])

PULSE_NAMES = ["gaussian", "nyquist"]
PULSE_LINE_STYLE = ["-", "--"]
MAX_LLD = 2.3
        
def get_space_integrals(m, z, I):
    """(wrapper) Compute spatial collision integral X0mm for a given time integral profile."""
    X0mm = pynlin.nlin.X0mm_space_integral(z, I, amplification_function=None)
    return X0mm

def plot_illustrative(fiber, wdm, cf, recompute=False):
    """Reproduce illustrative pulse-collision plots (Fig.1-style) for Gaussian/Nyquist pulses."""
    lg.debug("Plotting Fig.1 (pulse collisions)...")
    nyquist_pulse = NyquistPulse(
        baud_rate=cf.baud_rate,
        # beware of "aliasing" (i.e. when it is smaller than the maximum m we have repeated values)
        num_symbols=220,
        samples_per_symbol=2**5,
        rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate=cf.baud_rate,
        # beware of "aliasing" (i.e. when it is smaller than the maximum m we have repeated values)
        num_symbols=220,
        samples_per_symbol=2**5,
    )
    ls = ["-", "-"]
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
        plt.figure(figsize=(4, 2.2))
        m = [-10, -90]
        m1 = -10
        m2 = -90
        dgd_hi = 100e-15
        beta2a = 100e-27
        beta2b = 50e-27 # BEWARE
        if ipulse == 0:
            complementary_function = beta2rms_complementary
            beta2a = 100e-27
            beta2b = 50e-27
            beta2c = 132e-27
            beta2c_check = beta2rms_complementary(beta2a, beta2b)
            # assert beta2c_check == beta2c, f"Values do not match: {beta2c_check} vs {beta2c}"
        else:
            beta2a = 100e-27
            beta2b = 50e-27
            beta2c = 132e-27
            beta2c_check = beta2avg_complementary(beta2a, beta2b)
            # assert beta2c_check == beta2c, f"Values do not match: {beta2c_check} vs {beta2c}"
            complementary_function = beta2avg_complementary # beta2rms_complementary
            # xlabel = rf"$z/L_{{Dm}}$"
        xlabel = rf"$z/L_{{D}}^{{\mathrm{{rms}}}}$"
        # beta2bar = beta2rms(beta2a, beta2b)
        assert pulse.baud_rate == cf.baud_rate
        LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2a))
        LW = 1/(pulse.baud_rate * np.abs(dgd_hi))
        lg.debug("-"*20, "ILLUSTRATIVE COLLISION PLOT", "-"*20)
        lg.debug(f"LDbar = {LDbar:.2e} | L/LDbar = {fiber.length/LDbar:.2e}")
        lg.debug(
            f"low DGD parameters L/LD1: {-beta2a * pulse.baud_rate**2 * fiber.length:.2e}, L/LD2: {-beta2b * pulse.baud_rate**2 * fiber.length:.2e}")
        lg.debug(
            f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {complementary_function(beta2a, beta2b)}|")
        z = np.linspace(0, fiber.length, 400)
        # we need to be with a fixed L/LD
        assert (np.isclose(fiber.length, 2 * LDbar))

        # 1. the two pulses
        # cases are related to single collisions
        # assume beta2a = beta2b
        cases = [(dgd_hi, beta2a, beta2a, -10),
                 (dgd_hi, beta2a, beta2a, -90),]
        I_list = []
        for dgd, beta2a, _, m in cases:
            I = np.real(m_th_time_integral_general(m, z, pulse, dgd, beta2b, beta2c))
            I_list.append(I)

        # 2. the set of pulse peaks:
        # cases are related to set of parameters, not to single collisions
        # we utilize the beta2rms_complementary function to compute the beta2b value
        cases_peaks = [(dgd_hi, beta2a, beta2a, m1),
                       (dgd_hi, beta2b, beta2c, m2),]
        # assert np.abs(beta2rms(beta2b, complementary_function(beta2a, beta2b))) == np.abs(
            # beta2a), f"values: {beta2a}, {beta2rms(beta2b, complementary_function(beta2a, beta2b))}" # this only holds for the RMS case
        zw = 1/(pulse.baud_rate * dgd_hi)
        m_max = fiber.length / zw
        # lg.debug(f"  > m_max: {m_max}")
        m_axis = -np.array(range(int(round(m_max))))
        m_axis = m_axis[::20]
        peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
        z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
        #
        if not os.path.exists("results/fig1_peaks.npy") or recompute:
            # lg.debug("  Computing peaks...")
            for im, m in enumerate(m_axis):
                lg.debug(f"  m: {m:>10}")
                for ic, (dgd, beta2a_example, beta2b_example, _) in enumerate(cases_peaks):
                    # lg.debug(f"    case {ic}: dgd={dgd:.2e}, beta2a={beta2a_example:.2e}, beta2b={beta2b_example:.2e}")
                    # lg.debug(f"L/LDA = {fiber.length * pulse.baud_rate**2 * np.abs(beta2a_example):.2e}, L/LDB = {fiber.length * pulse.baud_rate**2 * np.abs(beta2b_example):.2e}")
                    I = np.real(m_th_time_integral_general(m, z, pulse, dgd, beta2a_example, beta2b_example))
                    peaks[im, ic] = np.max(I)
                    z_peaks[im, ic] = z[np.argmax(I)]
            np.save("results/fig1_peaks.npy", (peaks, z_peaks))
            # lg.debug("  Done computing and saving peaks.")
        else:
            # lg.debug("  Loading peaks...")
            peaks, z_peaks = np.load("results/fig1_peaks.npy")
            # lg.debug("  Done loading peaks.")

        # 3. case of very low DGD (almost zero)
        m_lo = 0
        # gvda_alt = 0.0
        # gvdb_alt = 0.0e-27
        I_low = np.real(m_th_time_integral_general(m_lo, z,
            pulse, 0.0, beta2a, beta2a))
        I_low_2 = np.real(m_th_time_integral_general(m_lo, z,
            pulse, 0.0, beta2b, beta2c))

        # 4. Plotting
        colors = ["grey", "blue"]
        markers = ["x", "o"] # x with equal dispersion, dots with unequal
        marker_sizes = [7, 3]
        lw = 1.0
        # plotting the peaks
        for ip, (peak, z_peak) in enumerate(zip(peaks, z_peaks)):
            for ic in range(len(cases_peaks)):
                if ip == 0:
                    plt.plot(z_peak[ic]/LDbar,
                             peak[ic]/pulse.baud_rate,
                             marker=markers[ic],
                             markersize=marker_sizes[ic],
                             label='case'+str(ic),
                             color=colors[ic],
                             linewidth=lw)
                else:
                    plt.plot(z_peak[ic]/LDbar,
                             peak[ic]/pulse.baud_rate,
                             markersize=marker_sizes[ic],
                             marker=markers[ic],
                             color=colors[ic],
                             linewidth=lw)
        # plotting the low DGD case
        plt.plot(z/LDbar,
                 I_low / pulse.baud_rate,
                 label='low DGD',
                 color="green",
                 linewidth=lw,
                 linestyle=ls[ipulse])
        plt.plot(z/LDbar,
                 I_low_2 / pulse.baud_rate,
                 label='low DGD with GVD',
                 color="orange",
                 linewidth=lw,
                 linestyle=ls[ipulse])
        # plt.title(f"m={m_lo}")
        for xi, yi, wi in zip(z[::100]/LDbar, I_low[::100]/pulse.baud_rate, I_low_2[::100] / pulse.baud_rate):
            lg.debug(f"{xi:.2e}, {yi:.2e}, {wi:.2e}")
        # plotting the pulses
        for ii, I in enumerate(I_list):
            if ii == 0:
                plt.plot(z/LDbar,
                         I / pulse.baud_rate,
                         label=f'try',
                         color="red",
                         linewidth=lw,
                         linestyle=ls[ipulse], )
            else:
                plt.plot(z/LDbar,
                         I / pulse.baud_rate,
                         color="red",
                         linewidth=lw,
                         linestyle=ls[ipulse], )
        # plt.legend()
        # add (a) or (b) label
        if ipulse == 0:
            label = "(a)"
        else:
            label = "(b)"

        plt.text(
            0.98, 0.95, label,
            transform=plt.gca().transAxes,
            ha='right', va='top',
            fontsize=12, fontweight='bold'
        )
        plt.xlabel(xlabel)
        # plt.xlabel(r'$z / L_D^{\mathrm{rms}}$')
        plt.ylabel(r'$I(z) \cdot T $')
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.tight_layout()
        # different pulses: 
        plt.savefig(f"media/1-quovadis_{PULSE_NAMES[ipulse]}.pdf", dpi=300, bbox_inches="tight", pad_inches=0)
        lg.debug("Done plotting Fig.1 media/1-quovadis_"+PULSE_NAMES[ipulse]+".pdf")
        plt.clf()


# range of the evaluation is hardcoded in the function to L/LDA=2
def get_I_low(fiber, m_lo, recompute=False):
    """Precompute I_low tables over dispersion ranges for Gaussian and Nyquist pulses."""
    nyquist_pulse = NyquistPulse(
        baud_rate=cf.baud_rate,
        num_symbols=220,
        samples_per_symbol=32,
        rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate=cf.baud_rate,
        num_symbols=220,
        samples_per_symbol=32,
    )
    z = np.linspace(0, fiber.length, 2)
    lld_range = np.linspace(1e-30, MAX_LLD, 50)  # HARDCODED

    LD_min = fiber.length / lld_range[-1]
    beta2_max = 1/(cf.baud_rate**2 * LD_min)
    beta2_range = np.linspace(1e-30, beta2_max, 50)
    lg.debug("-"*20, "DISPERSION ANALYSIS", "-"*20)
    lg.debug(
        f"LDbar_min = {LD_min:.2e} | L/LDbar_min = {fiber.length/LD_min:.2e}")
    # assert (np.isclose(fiber.length, 2 * LD_min))
    lg.debug(f"Computing with L/LD = {fiber.length*cf.baud_rate**2 * beta2_max:.2e}")
    I_low_values_multipulse = []
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
        assert pulse.baud_rate == cf.baud_rate
        def compute_I_low(pulse, z, beta2a, beta2b):
            I_low = np.real(m_th_time_integral_general(
                m_lo, [z[-1]], pulse, 0.0, beta2a, beta2b
            ))
            return I_low[-1]

        # Compute and/or load I_low
        save_path = f"results/I_low_{PULSE_NAMES[ipulse]}_m{m_lo}.npz"
        if recompute or not os.path.exists(save_path):
            I_low_values = np.array([
                [compute_I_low(pulse,  z, beta2a, beta2b)
                 for beta2a in beta2_range]
                for beta2b in beta2_range
            ], dtype=float) / cf.baud_rate
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # tip: use np.savez_compressed(...) if files are large
            np.savez(
                save_path,
                I_low_values=I_low_values,
                lld_range=np.asarray(lld_range),
                z=np.asarray(z),
            )

        I_low_dataset = np.load(save_path, allow_pickle=False)
        I_low_values = I_low_dataset["I_low_values"]
        I_low_values_multipulse.append(I_low_values)
    # I_low_values 1: Gaussian, 2: Nyquist
    return I_low_values_multipulse, lld_range


def ensure_i_low_dataset(m_lo: int,
                         ipulse: int,
                         baud_rate: float,
                         fiber_length: float,
                         max_lld: float,
                         recompute: bool = False) -> str:
    """Ensure an I_low dataset exists for the requested L/LD range."""
    if max_lld <= 0 or not np.isfinite(max_lld):
        raise ValueError("max_lld must be a positive finite value.")
    if ipulse not in (0, 1):
        raise ValueError(f"Unsupported ipulse={ipulse}. Expected 0 (gaussian) or 1 (nyquist).")

    save_path = f"results/I_low_{PULSE_NAMES[ipulse]}_m{m_lo}.npz"
    if os.path.exists(save_path) and not recompute:
        dataset = np.load(save_path, allow_pickle=False)
        lld_range = dataset["lld_range"]
        if float(lld_range[-1]) >= max_lld:
            return save_path

    lg.info(
        f"Computing I_low table for m_lo={m_lo}, pulse={PULSE_NAMES[ipulse]}, "
        f"L/LD_max={max_lld:.2f}"
    )
    if ipulse == 0:
        pulse = GaussianPulse(
            baud_rate=baud_rate,
            num_symbols=220,
            samples_per_symbol=32,
        )
    else:
        pulse = NyquistPulse(
            baud_rate=baud_rate,
            num_symbols=220,
            samples_per_symbol=32,
            rolloff=0.0,
        )

    z = np.linspace(0, fiber_length, 2)
    lld_range = np.linspace(1e-30, max_lld, 50)
    LD_min = fiber_length / lld_range[-1]
    beta2_max = 1.0 / (baud_rate**2 * LD_min)
    beta2_range = np.linspace(1e-30, beta2_max, 50)

    def compute_I_low(beta2a, beta2b):
        I_low = np.real(m_th_time_integral_general(
            m_lo, [z[-1]], pulse, 0.0, beta2a, beta2b
        ))
        return I_low[-1]

    I_low_values = np.array([
        [compute_I_low(beta2a, beta2b) for beta2a in beta2_range]
        for beta2b in beta2_range
    ], dtype=float) / baud_rate

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        I_low_values=I_low_values,
        lld_range=np.asarray(lld_range),
        z=np.asarray(z),
    )
    return save_path


# TODO implement the argument exchange symmetry, somehow
def build_I_low_interpolator(I_low_dataset, ipulse: int):
    """Wrap RegularGridInterpolator for I_low values with basic input validation."""
    lld_range = I_low_dataset['lld_range']
    I_low_values = I_low_dataset['I_low_values']
    interp_func = RegularGridInterpolator(
        (lld_range, lld_range),
        I_low_values,
        bounds_error=False,
        fill_value=None
    )
    def interp_func_wrapped(x, y):
        assert(x <= 1.1 * lld_range[-1] and y <= 1.1 * lld_range[-1]), f"Input {x} exceeds the 110% of the interpolation range [{lld_range[0]}, {lld_range[-1]}]"
        assert(x>=0 and y>=0), f"Input has negative values check that your LD is positive"
        return interp_func([x, y])
    return interp_func_wrapped


def get_systems_dispersions(cf):
    """Return normalized dispersion (L/L_D) combinations for all mode/channel pairs."""
    fiber = MMFiber(
        effective_area=cf.effective_area,
        overlap_integrals=np.load('results/oi_fit.npy'),
        group_delay=load_group_delay(),
        length=cf.fiber_length,
        n_modes=cf.n_modes
    )
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    # for every choice of channel in each mode, compute the gvda and gvdb.
    gvds = np.zeros((cf.n_modes, cf.n_channels))
    for imode in range(cf.n_modes):
        for ichan in range(cf.n_channels):
            freq = wdm.frequency_grid()[ichan]
            gvds[imode, ichan] = fiber.group_delay.evaluate_beta2(
                imode, freq)
    lld = cf.baud_rate**2 * fiber.length * np.abs(gvds)
    X, Y = np.meshgrid(lld, lld, indexing='xy')
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    return X_flat, Y_flat
    

def plot_dispersion_analysis(fiber, 
                             m_lo=3, 
                             recompute=True, 
                             with_system_data=False,
                             system_config=None):
    """Plot contour maps of I_low over dispersion ranges and optionally overlay system points."""
    lg.debug("Plotting Fig.1 (pulse collisions)...")
    I_low_values, lld_range = get_I_low(fiber, m_lo, recompute=recompute)
    if with_system_data:
        if system_config is None:
            raise ValueError("system_config is required when with_system_data=True.")
        X, Y = get_systems_dispersions(system_config)
        
    for ipulse, I_low_values in enumerate(I_low_values):
        plt.figure(figsize=(3, 2.3))
        if ipulse == 0:
            vmax = 0.36
        else:
            vmax = 0.6
        contour = plt.contourf(lld_range, lld_range, np.clip(
            I_low_values, a_min=-10, a_max=0.91), levels=20, cmap='viridis', 
                               vmin=0, vmax = vmax)
        contour_lines = plt.contour(lld_range, lld_range, np.clip(
            I_low_values, a_min=-10, a_max=0.6), levels=5, colors="w")
        plt.clabel(contour_lines, inline=True, fontsize=8)
        
        if with_system_data:
            plt.scatter(X, Y, marker=".", s = 0.011, color='white', alpha=0.15)
           
        # plt.xlabel(r'$|\beta_{2A}|LT^{-2}$')
        # plt.ylabel(r'$|\beta_{2B}|LT^{-2}$')
        plt.xlabel(r'$z/L_{DA}$')
        plt.ylabel(r'$z/L_{DB}$')
        # plt.colorbar(label=r'$I_{0;AB}(\bar{L}_{D0}) \cdot T$')
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax = plt.gca()
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks(np.linspace(0, 2, 5))
        ax.set_yticks(np.linspace(0, 2, 5))
        # plt.title(f"m = {m_lo}")
        plt.tight_layout()
        plt.savefig("media/differential_dispersion_" +
                    PULSE_NAMES[ipulse]+"_m"+str(m_lo)+".pdf", dpi=300, bbox_inches="tight", pad_inches=0.01)
        lg.debug(
            f"Saved dispersion in media/differential_dispersion_{PULSE_NAMES[ipulse]}.pdf")
        plt.clf()
        
        
if __name__ == "__main__":
    lg.info("Running collision module as main...")
    lg.add(sys.stdout, level="DEBUG")
    
    import numpy as np
    class Fiber:
        length = 200e3
    class Config:
        baud_rate = 10e9  # this is bound by the assertion 
    class WDM:
        pass  # placeholder
    
    fiber = Fiber()
    cf = Config()
    wdm = WDM()

    plot_illustrative(fiber, wdm, cf, recompute=True)
    exit()
    for m_lo in [0, 1, 2, 3, 4, 5]:
        plot_dispersion_analysis(fiber, 
                                 recompute=False,
                                 m_lo=m_lo, 
                                 with_system_data=False)
    exit()
    
    for ipulse in [0, 1]:
        I_low_dataset = np.load(f"results/I_low_{PULSE_NAMES[ipulse]}.npz", allow_pickle=False)
        interp_func = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
        lg.debug(interp_func((0.0, 0.0)))
        lg.debug("type of interp_func:", type(interp_func))
        lg.trace(
            f"Testing the interpolating function for pulse:")
        test_points = [(0, 0), (0, 2), (2, 0), (1, 1), (2, 2)]
        for point in test_points:
            lg.trace(f"Point {point}: Interpolated = {interp_func(point)}")
            
