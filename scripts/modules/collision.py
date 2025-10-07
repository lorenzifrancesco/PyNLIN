import numpy as np
from pynlin.nlin import m_th_time_integral_general
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
from scripts.modules.beta_utils import beta2rms, beta2rms_complementary, beta2avg, beta2avg_complementary
from pynlin.pulses import NyquistPulse, GaussianPulse
from scipy.interpolate import RegularGridInterpolator

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
    num_symbols=220, # beware of "aliasing" (i.e. when it is smaller than the maximum m we have repeated values)
    samples_per_symbol=2**5,
    rolloff=0.0,
    )
    gaussian_pulse = GaussianPulse(
        baud_rate = cf.baud_rate,
        num_symbols = 220, # beware of "aliasing" (i.e. when it is smaller than the maximum m we have repeated values)
        samples_per_symbol = 2**5,
    )
    plt.figure(figsize=(4, 2.2))
    ls = ["-", "--"]
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
      m = [-10, -90]
      m1 = -10
      m2 = -90
      dgd_hi = 100e-15
      beta2a = -100e-27 # here is the problem !?
      beta2b = -1.0e-50
      # beta2bar = beta2rms(beta2a, beta2b)
      assert pulse.baud_rate == cf.baud_rate
      LDbar = 1/(pulse.baud_rate**2 * np.abs(beta2a))
      LW = 1/(pulse.baud_rate * np.abs(dgd_hi))
      print("-"*20, "ILLUSTRATIVE COLLISION PLOT" ,"-"*20)
      print(f"LDbar = {LDbar:.2e} | L/LDbar = {fiber.length/LDbar:.2e}") 
      print(f"low DGD parameters L/LD1: {-beta2a * pulse.baud_rate**2 * fiber.length:.2e}, L/LD2: {-beta2b * pulse.baud_rate**2 * fiber.length:.2e}")
      print(f"B2A = {beta2a:.2e} | B2B = {beta2b:.2e}  B2complement = {beta2rms_complementary(beta2a, beta2b)}|")
      z = np.linspace(0, fiber.length, 200)
      # we need to be with a fixed L/LD
      assert(np.isclose(fiber.length, 2 * LDbar))
      
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
      cases_peaks = [(dgd_hi, beta2a, beta2a, m1), 
                    (dgd_hi, beta2rms_complementary(beta2a, beta2b), beta2b, m2),]
      assert np.abs(beta2rms(beta2b, beta2rms_complementary(beta2a, beta2b))) == np.abs(beta2a), f"values: {beta2a}, {beta2rms(beta2b, beta2rms_complementary(beta2a, beta2b))}"
      zw = 1/(pulse.baud_rate * dgd_hi)
      m_max = fiber.length / zw
      # print(f"  > m_max: {m_max}")
      m_axis = -np.array(range(int(round(m_max))))
      m_axis = m_axis[::10]
      peaks   = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      z_peaks = np.zeros(2)[np.newaxis, :].repeat(len(m_axis), axis=0)
      #
      if not os.path.exists("results/fig1_peaks.npy") or recompute:
        # print("  Computing peaks...")
        for im, m in enumerate(m_axis):
          print(f"  m: {m:>10}")
          for ic, (dgd, beta2a_example, beta2b_example, _) in enumerate(cases_peaks):
            # print(f"    case {ic}: dgd={dgd:.2e}, beta2a={beta2a_example:.2e}, beta2b={beta2b_example:.2e}")
            # print(f"L/LDA = {fiber.length * pulse.baud_rate**2 * np.abs(beta2a_example):.2e}, L/LDB = {fiber.length * pulse.baud_rate**2 * np.abs(beta2b_example):.2e}")
            I = np.real(m_th_time_integral_general(pulse, fiber, wdm, (0, 0), (0, 0), 0.0, m, z, dgd, None, beta2a_example, beta2b_example))
            peaks[im, ic]   = np.max(I)
            z_peaks[im, ic] = z[np.argmax(I)]
        np.save("results/fig1_peaks.npy", (peaks, z_peaks))
        # print("  Done computing and saving peaks.")
      else:
        # print("  Loading peaks...")
        peaks, z_peaks = np.load("results/fig1_peaks.npy")
        # print("  Done loading peaks.")
      
      # 3. case of very low DGD (almost zero)
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
    
    z = np.linspace(0, fiber.length, 2)
    lld_range = np.linspace(1e-30, 2, 50)
    LD_min = fiber.length/ lld_range[-1]
    beta2_max = 1/(cf.baud_rate**2 * LD_min)
    beta2_range = np.linspace(1e-30, beta2_max, 50)
    # beta2_range = np.linspace(0.1, 100, 50) * 1e-27
    # beta20 = 1 / (cf.baud_rate**2 * z[-1])
    # LDbar_min = 1/(cf.baud_rate**2 * np.max(np.abs(beta2_range)))
    print("-"*20, "DISPERSION ANALYSIS" ,"-"*20)
    print(f"LDbar_min = {LD_min:.2e} | L/LDbar_min = {fiber.length/LD_min:.2e}") 
    # print(f"low DGD parameters L/LD1: {-beta2a * pulse.baud_rate**2 * fiber.length:.2e}, L/LD2: {-beta2b * pulse.baud_rate**2 * fiber.length:.2e}")
    plt.figure(figsize=(20.2, 20.2))
    ls = ["-", "--"]
    names = ["gaussian", "nyquist"]
    
    assert(np.isclose(fiber.length, 2 * LD_min))
    for ipulse, pulse in enumerate([gaussian_pulse, nyquist_pulse]):
      # 3. case of very low DGD (almost zero)
      assert pulse.baud_rate == cf.baud_rate
      def compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b):
        I_low = np.real(m_th_time_integral_general(
            pulse, fiber, wdm, (0, 0), (0, 0), 0.0, 0, [z[-1]], 1e-40, None, beta2a, beta2b
        ))
        return I_low[-1]
      # assert(np.isclose(beta2_range[-1]/beta20, 1.0))

      # Meshgrid for plotting (rows: beta2b, cols: beta2a)
      beta2a_values, beta2b_values = np.meshgrid(beta2_range, beta2_range, indexing="xy")

      # Compute and/or load I_low
      save_path = f"results/I_low_{names[ipulse]}.npy"
      if recompute or not os.path.exists(save_path):
          I_low_values = np.array([
              [compute_I_low(pulse, fiber, wdm, z, beta2a, beta2b) for beta2a in beta2_range]
              for beta2b in beta2_range
          ], dtype=float) / cf.baud_rate
          os.makedirs(os.path.dirname(save_path), exist_ok=True)
          np.save(save_path, I_low_values)

      I_low_values = np.load(save_path)

      # Downsample for interpolation (stride = 2)
      llda = beta2_range[::2]  # along columns (a-axis)
      lldb = beta2_range[::2]  # along rows (b-axis)
      I_low_vals = I_low_values[::2, ::2]

      # Build interpolator
      # IMPORTANT: axis 0 corresponds to rows -> lldb (beta2b), axis 1 to cols -> llda (beta2a)
      interp_func = RegularGridInterpolator((lldb, llda), I_low_vals, bounds_error=False, fill_value=None)

      # Evaluate interpolator on the full plotting grid
      # Points must be in the same axis order as given to the interpolator: (beta2b, beta2a)
      points = np.stack([beta2b_values.ravel(), beta2a_values.ravel()], axis=-1)
      I_interp = interp_func(points).reshape(beta2a_values.shape)
      print(interp_func((1, 1)))
      # exit()
      # Clip once, reuse for both contour and contour lines
      Zf = np.clip(I_interp, a_min=-10, a_max=0.91)
      Zf_lines = np.clip(I_interp, a_min=-10, a_max=0.16)

      Xa = lld_range
      Yb = lld_range

      # Plot
      contour = plt.contourf(Xa, Yb, Zf, levels=10, cmap="viridis")
      contour_lines = plt.contour(Xa, Yb, Zf_lines, levels=50, colors="w")

      # max 
      # plt.figure(figsize=(10, 10.3))
      # contour      = plt.contourf(beta2a_values/beta20, beta2b_values/beta20, np.clip(I_low_values, a_min=-10, a_max=0.91), levels=100, cmap='viridis')
      # contour_lines = plt.contour(beta2a_values/beta20, beta2b_values/beta20, np.clip(I_low_values, a_min=-10, a_max=0.16), levels=50, colors="w")
      
      plt.clabel(contour_lines, inline=True, fontsize=8)

      plt.xlabel(r'$|\beta_{2A}|LT^{-2}$')
      plt.ylabel(r'$|\beta_{2B}|LT^{-2}$')
      # plt.colorbar(label=r'$I_{0;AB}(\bar{L}_{D0}) \cdot T$')
      ax = plt.gca()
      ax.set_aspect('equal')
      ax.xaxis.set_major_formatter(formatter)
      ax.yaxis.set_major_formatter(formatter)
      plt.tight_layout()
      plt.savefig("media/differential_dispersion_"+names[ipulse]+".pdf")
      print(f"Saved dispersion in media/differential_dispersion_{names[ipulse]}.pdf")
      plt.clf()
      
      # ----- 1D sweeps along the axes (vary L with one beta2 set to 0) -----
      # Pick the beta2 value for the nonzero axis (you can tweak these)
      beta_extreme = beta2_range[-1]   # e.g. mid of your range

      # Sweep L from small positive to full fiber length (avoid exactly 0)
      L_vals = np.linspace(1e-6 * fiber.length, fiber.length, 200)

      def I_over_L(beta2a, beta2b):
          vals = []
          for L in L_vals:
              z_local = np.array([0.0, L])
              vals.append(compute_I_low(pulse, fiber, wdm, z_local, beta2a, beta2b))
          return np.array(vals, dtype=float) / cf.baud_rate

      # Along x-axis of the 2D figure: beta2b = 0, vary L (equivalent to varying |beta2a| L T^{-2})
      I_L_a = I_over_L(beta_extreme, 0.0)
      x_a = np.abs(beta_extreme) * (cf.baud_rate**2) * L_vals  # |beta2a| L T^{-2}

      # Along y-axis of the 2D figure: beta2a = 0, vary L (equivalent to varying |beta2b| L T^{-2})
      new_beta2 = beta_extreme/np.sqrt(2)
      I_L_b = I_over_L(new_beta2, new_beta2)  # to have the same LD as beta2b_fixed
      x_b = np.abs(beta_extreme) * (cf.baud_rate**2) * L_vals  # |beta2b| L T^{-2}

      # Plot and save
      plt.figure(figsize=(4.2, 3.0))
      plt.plot(x_a, np.clip(I_L_a, a_min=-10, a_max=0.91),
               label=rf'$\,\beta_{{2A}}={beta_extreme:.2e}\,$, $\,\beta_{{2B}}=0$')
      plt.plot(x_b, np.clip(I_L_b, a_min=-10, a_max=0.91), linestyle='--',
               label=rf'$\,\beta_{{2A}}=0\,$, $\,\beta_{{2B}}={beta_extreme:.2e}$')
      plt.xlabel(r'$|\beta_{2}|\,L\,T^{-2}$')
      plt.ylabel(r'$I_{0;AB}(\bar{L}_{D0}) \cdot T$')
      plt.legend()
      plt.tight_layout()
      plt.savefig(f"media/axis_sweeps_{names[ipulse]}.pdf")
      print(f"Saved axis sweeps in media/axis_sweeps_{names[ipulse]}.pdf")
      plt.clf()

if __name__ == "__main__":
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
    plot_dispersion_analysis(fiber, wdm, cf, recompute=False)
    plot_illustrative(fiber, wdm, cf, recompute=True)