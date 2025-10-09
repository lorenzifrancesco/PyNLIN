from loguru import logger as lg
from log_init import init_logging
init_logging()

import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from scripts.modules.beta_utils import beta2rms, beta2rms_complementary, beta2avg, beta2avg_complementary
from pynlin.pulses import NyquistPulse, GaussianPulse
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.integrate import trapezoid

from type_utils import PulseShape

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])

PULSE_NAMES = ["gaussian", "nyquist"]
PULSE_LINE_STYLE = ["-", "--"]

def plot_illustrative(fiber, wdm, cf, recompute=False):
    print("Plotting Fig.1 (pulse collisions)...")
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
    plt.figure(figsize=(4, 2.2))
    ls = ["-", "--"]
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
        m = [-10, -90]
        m1 = -10
        m2 = -90
        dgd_hi = 100e-15
        beta2a = -100e-27  # here is the problem !?
        beta2b = -1.0e-50
        # beta2bar = beta2rms(beta2a, beta2b)
        assert pulse.baud_rate == cf.baud_rate
        LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2a))
        LW = 1/(pulse.baud_rate * np.abs(dgd_hi))
        print("-"*20, "ILLUSTRATIVE COLLISION PLOT", "-"*20)
        print(f"LDbar = {LDbar:.2e} | L/LDbar = {fiber.length/LDbar:.2e}")
        print(
            f"low DGD parameters L/LD1: {-beta2a * pulse.baud_rate**2 * fiber.length:.2e}, L/LD2: {-beta2b * pulse.baud_rate**2 * fiber.length:.2e}")
        print(
            f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {beta2rms_complementary(beta2a, beta2b)}|")
        z = np.linspace(0, fiber.length, 200)
        # we need to be with a fixed L/LD
        assert (np.isclose(fiber.length, 2 * LDbar))

        # 1. the two pulses
        # cases are related to single collisions
        # assume beta2a = beta2b
        cases = [(dgd_hi, beta2a, beta2a, -10),
                 (dgd_hi, beta2a, beta2a, -90),]
        I_list = []
        for dgd, beta2a, _, m in cases:
            I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0),
                        0.0, m, z, dgd, None, beta2b, beta2rms_complementary(beta2a, beta2b)))
            I_list.append(I)

        # 2. the set of pulse peaks:
        # cases are related to set of parameters, not to single collisions
        # we utilize the beta2rms_complementary function to compute the beta2b value
        cases_peaks = [(dgd_hi, beta2a, beta2a, m1),
                       (dgd_hi, beta2rms_complementary(beta2a, beta2b), beta2b, m2),]
        assert np.abs(beta2rms(beta2b, beta2rms_complementary(beta2a, beta2b))) == np.abs(
            beta2a), f"values: {beta2a}, {beta2rms(beta2b, beta2rms_complementary(beta2a, beta2b))}"
        zw = 1/(pulse.baud_rate * dgd_hi)
        m_max = fiber.length / zw
        # print(f"  > m_max: {m_max}")
        m_axis = -np.array(range(int(round(m_max))))
        m_axis = m_axis[::10]
        peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
        z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
        #
        if not os.path.exists("results/fig1_peaks.npy") or recompute:
            # print("  Computing peaks...")
            for im, m in enumerate(m_axis):
                print(f"  m: {m:>10}")
                for ic, (dgd, beta2a_example, beta2b_example, _) in enumerate(cases_peaks):
                    # print(f"    case {ic}: dgd={dgd:.2e}, beta2a={beta2a_example:.2e}, beta2b={beta2b_example:.2e}")
                    # print(f"L/LDA = {fiber.length * pulse.baud_rate**2 * np.abs(beta2a_example):.2e}, L/LDB = {fiber.length * pulse.baud_rate**2 * np.abs(beta2b_example):.2e}")
                    I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0),
                                (0, 0), 0.0, m, z, dgd, None, beta2a_example, beta2b_example))
                    peaks[im, ic] = np.max(I)
                    z_peaks[im, ic] = z[np.argmax(I)]
            np.save("results/fig1_peaks.npy", (peaks, z_peaks))
            # print("  Done computing and saving peaks.")
        else:
            # print("  Loading peaks...")
            peaks, z_peaks = np.load("results/fig1_peaks.npy")
            # print("  Done loading peaks.")

        # 3. case of very low DGD (almost zero)
        I_low = np.real(m_th_time_integral_general(
            pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, z, 1e-40, None, beta2a, beta2a))
        I_low_2 = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0),
                          0.0, 0, z, 1e-40, None, beta2b, beta2rms_complementary(beta2a, beta2b)))

        # 4. Plotting
        colors = ["grey", "blue"]
        markers = ["x", "o"]
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
                 label='low DGD',
                 color="orange",
                 linewidth=lw,
                 linestyle=ls[ipulse])
        for xi, yi, wi in zip(z[::100]/LDbar, I_low[::100]/pulse.baud_rate, I_low_2[::100] / pulse.baud_rate):
            print(f"{xi:.2e}, {yi:.2e}, {wi:.2e}")
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
    plt.xlabel(r'$z / L_D^{\mathrm{rms}}$')
    plt.ylabel(r'$I(z) \cdot T $')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/1-quovadis.pdf")
    print("Done plotting Fig.1 media/1-quovadis.pdf .")


# range of the evaluation is hardcoded in the function to L/LDA=2
def get_I_low(fiber, wdm, recompute=False):
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
    lld_range = np.linspace(1e-30, 2, 50)  # HARDCODED

    LD_min = fiber.length / lld_range[-1]
    beta2_max = 1/(cf.baud_rate**2 * LD_min)
    beta2_range = np.linspace(1e-30, beta2_max, 50)
    lg.debug("-"*20, "DISPERSION ANALYSIS", "-"*20)
    lg.debug(
        f"LDbar_min = {LD_min:.2e} | L/LDbar_min = {fiber.length/LD_min:.2e}")
    assert (np.isclose(fiber.length, 2 * LD_min))
    I_low_values_multipulse = []
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
        assert pulse.baud_rate == cf.baud_rate

        def compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b):
            I_low = np.real(m_th_time_integral_general(
                pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, [
                    z[-1]], 1e-40, None, beta2a, beta2b
            ))
            return I_low[-1]

        # Compute and/or load I_low
        save_path = f"results/I_low_{PULSE_NAMES[ipulse]}.npz"
        if recompute or not os.path.exists(save_path):
            I_low_values = np.array([
                [compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b)
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


# TODO implement the argument exchange symmetry, somehow
def build_I_low_interpolator(I_low_dataset, ipulse: int):
    lld_range = I_low_dataset['lld_range']
    I_low_values = I_low_dataset['I_low_values']
    z_array = I_low_dataset['z']
    # save also a grid fitting function if it is not present
    save_path_interp = f"results/low_dgd_integral/lowdgd_integral_interp_{PULSE_NAMES[ipulse]}__LoverLD_[{lld_range[0]:.1e}-{lld_range[-1]:.1e}]_N{len(lld_range)}_Nz{len(z_array)}.npz"
    # generate the grid interpolation
    interp_func = RegularGridInterpolator(
        (lld_range, lld_range),
        I_low_values,
        bounds_error=False,
        fill_value=None
    )
    # implement a wrapper to make sure the input is not too much over the original bounds
    def interp_func_wrapped(x):
        assert(x[0] <= 1.1 * lld_range[-1] and x[1] <= 1.1 * lld_range[-1]), f"Input {x} exceeds the 110% of the interpolation range [{lld_range[0]}, {lld_range[-1]}]"
        assert(x[0]>=0 and x[1]>=0), f"Input {x} has negative values check that your LD is positive"
        return interp_func(x)
    return interp_func_wrapped


def plot_dispersion_analysis(fiber, wdm, recompute=True):
    print("Plotting Fig.1 (pulse collisions)...")
    I_low_values, lld_range = get_I_low(fiber, wdm, recompute=recompute)

    for ipulse, I_low_values in enumerate(I_low_values):
        plt.figure(figsize=(4, 4.3))
        contour = plt.contourf(lld_range, lld_range, np.clip(
            I_low_values, a_min=-10, a_max=0.91), levels=20, cmap='viridis')
        contour_lines = plt.contour(lld_range, lld_range, np.clip(
            I_low_values, a_min=-10, a_max=0.6), levels=5, colors="w")

        plt.clabel(contour_lines, inline=True, fontsize=8)
        plt.xlabel(r'$|\beta_{2A}|LT^{-2}$')
        plt.ylabel(r'$|\beta_{2B}|LT^{-2}$')
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
        plt.tight_layout()
        plt.savefig("media/differential_dispersion_" +
                    PULSE_NAMES[ipulse]+".pdf")
        print(
            f"Saved dispersion in media/differential_dispersion_{PULSE_NAMES[ipulse]}.pdf")
        plt.clf()
        
        
if __name__ == "__main__":
    lg.info("Running collision module as main...")
    import numpy as np
    class Fiber:
        length = 200e3
    class Config:
        baud_rate = 10e9  # 32 Gbaud
    class WDM:
        pass  # placeholder
    
    fiber = Fiber()
    cf = Config()
    wdm = WDM()

    # Run both plots
    plot_dispersion_analysis(fiber, wdm, recompute=True)
    plot_illustrative(fiber, wdm, cf, recompute=False)
    
    for ipulse in [0, 1]:
        I_low_dataset = np.load(f"results/I_low_{PULSE_NAMES[ipulse]}.npz", allow_pickle=False)
        interp_func = build_I_low_interpolator(I_low_dataset, ipulse=ipulse)
        print(interp_func((0.0, 0.0)))
        lg.debug("type of interp_func:", type(interp_func))
        lg.trace(
            f"Testing the interpolating function for pulse:")
        test_points = [(0, 0), (0, 2), (2, 0), (1, 1), (2, 2)]
        for point in test_points:
            lg.trace(f"Point {point}: Interpolated = {interp_func(point)}")