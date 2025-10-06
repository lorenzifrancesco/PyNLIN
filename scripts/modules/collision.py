import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from scripts.modules.beta_utils import beta2rms, beta2rms_complementary, beta2avg, beta2avg_complementary
from pynlin.pulses import NyquistPulse, GaussianPulse

formatter = ScalarFormatter()
formatter.set_scientific(True)
formatter.set_powerlimits([0, 0])
 

def plot_illustrative(fiber, wdm, cf, recompute=False):
    """
    Plot of Marco
    """
    print("Plotting Fig.1 (pulse collisions)...")
    
    nyquist_pulse = NyquistPulse(
    baud_rate=cf.baud_rate,
    num_symbols=200, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
    samples_per_symbol=30,
    rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate = cf.baud_rate,
        num_symbols = 100, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
        samples_per_symbol = 2**5,
    )
    plt.figure(figsize=(4, 2.2))
    ls = ["-", "--"]
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
      m = [-10, -90]
      m1 = -10
      m2 = -90
      dgd_hi = 100e-15
      beta2a = -100e-27
      beta2b = -50e-27
      # beta2bar = beta2rms(beta2a, beta2b)
      LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2a))
      LW = 1/(pulse.baud_rate * np.abs(dgd_hi))
      print(f"LDbar = {LDbar:.2e} | LW/LDbar = {LW/LDbar:.2e}") 
      z = np.linspace(0, fiber.length, 1000)
      
      # 1. the two pulses
      # cases are related to single collisions
      # assume beta2a = beta2b
      cases = [(dgd_hi, beta2a, beta2a, -10),
               (dgd_hi, beta2a, beta2a, -90),]
      I_list = []
      for dgd, beta2a, _, m in cases:
          I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2b, beta2rms_complementary(beta2a, beta2b)))
          I_list.append(I)

      # 2. the set of pulse peaks:
      # cases are related to set of parameters, not to single collisions
      # we utilize the beta2rms_complementary function to compute the beta2b value
      print(f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {beta2rms_complementary(beta2a, beta2b)}|")
      cases_peaks = [(dgd_hi, beta2a, beta2a, m1), 
                    (dgd_hi, beta2rms_complementary(beta2a, beta2b), beta2b, m2),]
      zw = 1/(pulse.baud_rate * dgd_hi)
      m_max = fiber.length / zw
      print(f"  > m_max: {m_max}")
      m_axis = -np.array(range(int(round(m_max))))
      m_axis = m_axis[::20]
      peaks   = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      #
      if not os.path.exists("results/fig1_peaks.npy") or recompute:
        print("  Computing peaks...")
        for im, m in enumerate(m_axis):
          print(f"  m: {m:>10}")
          for ic, (dgd, beta2a_example, beta2b_example, _) in enumerate(cases_peaks):
            I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a_example, beta2b_example))
            peaks[im, ic]   = np.max(I)
            z_peaks[im, ic] = z[np.argmax(I)]
        np.save("results/fig1_peaks.npy", (peaks, z_peaks))
        print("  Done computing and saving peaks.")
      else:
        print("  Loading peaks...")
        peaks, z_peaks = np.load("results/fig1_peaks.npy")
        print("  Done loading peaks.")
      print(peaks)
      
      # 3. case of very low DGD (almost zero)
      print(f"low DGD parameters L/LD1: {-beta2a * pulse.baud_rate**2 * fiber.length:.2e}, L/LD2: {-beta2b * pulse.baud_rate**2 * fiber.length:.2e}")
      I_low = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, z, 1e-40, None, beta2a, beta2a))
      I_low_2 = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, z, 1e-40, None, beta2b, beta2rms_complementary(beta2a, beta2b)))
      
      # 4. Plotting
      colors  = ["grey", "blue"]
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
          print(len(z))
          plt.plot(z/LDbar, 
                  I / pulse.baud_rate, 
                  color="red",
                  linewidth=lw,
                  linestyle=ls[ipulse], )
    # plt.legend()
    plt.xlabel(r'$z / \bar{L_D}$')
    plt.ylabel(r'$I(z) \cdot T $')
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig("media/1-quovadis.pdf")
    print("Done plotting Fig.1 media/1-quovadis.pdf .")


def plot_dispersion_analysis(fiber, wdm, cf, recompute=True):
    print("Plotting Fig.1 (pulse collisions)...")
    
    nyquist_pulse = NyquistPulse(
    baud_rate=cf.baud_rate,
    num_symbols=200, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
    samples_per_symbol=10,
    rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate=cf.baud_rate,
        num_symbols=5e2, # CHANGING THIS SOLVES THE ALIASING PROBLEM TODO
        samples_per_symbol=2**5,
    )
    plt.figure(figsize=(2.2, 2.2))
    ls = ["-", "--"]
    names = ["gaussian", "nyquist"]
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
      z = np.linspace(0, fiber.length, 2)
      # 3. case of very low DGD (almost zero)
      def compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b):
        I_low = np.real(m_th_time_integral_general(
            pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, [z[-1]], 1e-40, None, beta2a, beta2b
        ))
        return I_low[-1]
      beta2_range = np.linspace(0.1, 100, 50) * 1e-27
      beta20 = 1 / (cf.baud_rate**2 * z[-1])
      print(beta2_range)
      print(beta20)
      # assert(np.isclose(beta2_range[-1]/beta20, 1.0))
      
      print(f"Showing values of z/L_D from {z[-1] * cf.baud_rate**2 * beta2_range[0]:.2e} to {z[-1] * cf.baud_rate **2 * beta2_range[-1]:.2e}")
      beta2a_values, beta2b_values = np.meshgrid(beta2_range, beta2_range)

      if recompute:
        I_low_values = np.array([
            [compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b) for beta2a in beta2_range]
            for beta2b in beta2_range
        ]) / cf.baud_rate
        print("  Done computing I_low values.")
        np.save(f"results/I_low_{names[ipulse]}.npy", I_low_values)
      I_low_values = np.load(f"results/I_low_{names[ipulse]}.npy")
      # max 
      # plt.figure(figsize=(10, 10.3))
      print(I_low_values)
      contour      = plt.contourf(beta2a_values/beta20, beta2b_values/beta20, np.clip(I_low_values, a_min=-10, a_max=0.91), levels=100, cmap='viridis')
      contour_lines = plt.contour(beta2a_values/beta20, beta2b_values/beta20, np.clip(I_low_values, a_min=-10, a_max=0.91), levels=20, colors="w")
      plt.clabel(contour_lines, inline=True, fontsize=8)

      plt.xlabel(r'$|\beta_{2A}|LT^{-2}$')
      plt.ylabel(r'$|\beta_{2B}|LT^{-2}$')
      # plt.colorbar(label=r'$I_{0;AB}(\bar{L}_{D0}) \cdot T$')
      plt.gca().set_aspect('equal')
      plt.tight_layout()
      plt.savefig("media/differential_dispersion_"+names[ipulse]+".pdf")
      print(f"Saved dispersion in media/differential_dispersion_{names[ipulse]}.pdf")
      plt.clf()