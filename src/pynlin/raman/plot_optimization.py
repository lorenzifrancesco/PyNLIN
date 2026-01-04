import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import viridis

from pynlin.system import System
from pynlin.utils import get_next_filename
from pynlin.utils import dBm2watt, watt2dBm


def adjust_luminosity(color, factor):
    """Scale an RGB color toward lighter or darker variants."""
    rgb = np.array(mcolors.to_rgb(color))  # Convert to RGB
    return np.clip(rgb * factor, 0, 1)  # Scale and clip values

def plot_profiles(signal_wavelengths,
                  signal_solution,
                  ase_solution,
                  pump_wavelengths,
                  pump_solution,
                  pump_powers,
                  cf: System,
                  wallpaper_mode=False, 
                  single_out_mode = None):
    """Plot signal, ASE, and pump power profiles and save flatness snapshots."""
    # Guard against zeros/NaNs to avoid -inf in dBm plots
    eps = 1e-18
    signal_solution = np.where(np.isfinite(signal_solution), signal_solution, np.nan)
    signal_solution = np.maximum(signal_solution, eps)
    if ase_solution is not None:
        ase_solution = np.where(np.isfinite(ase_solution), ase_solution, np.nan)
        ase_solution = np.maximum(ase_solution, eps)
    if pump_solution is not None:
        pump_solution = np.where(np.isfinite(pump_solution), pump_solution, np.nan)
        pump_solution = np.maximum(pump_solution, eps)
    plt.clf()
    # plt.figure(figsize=(2.5, 2))
    plt.figure(figsize=(3.6, 2))

    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3
    # lss = ["-", "--", "-.", ":", "-"]
    mode_labels = ["LP01", "LP11", "LP21", "LP02"]
    
    if single_out_mode is not None:
        submodes = [single_out_mode]
    else:
        submodes = range(cf.n_modes)
        
    for i in submodes:
        if wallpaper_mode:
          plt.plot(z_plot,
                   watt2dBm(signal_solution[:, :, i]), color=cmap(i / cf.n_modes), alpha=0.9, lw=0.01)
        else:
           plt.plot(z_plot,
                   watt2dBm(signal_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.3), alpha=0.1, lw=0.3)
        try:
          plt.plot(z_plot,
                 watt2dBm(ase_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.7, ls="-")
        except:
          print(f"got data without ASE.")
    
    if wallpaper_mode:
       lww = 1
    else:
       lww = 3
    plt.plot(z_plot, watt2dBm(np.max(signal_solution, axis=(1, 2))), color=adjust_luminosity("cyan", 0.8),    lw = lww, ls ="-.")
    plt.plot(z_plot, watt2dBm(np.min(signal_solution, axis=(1, 2))), color=adjust_luminosity("magenta", 0.8), lw = lww, ls ="-.")
    pass
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    # plt.legend()
    plt.tight_layout()
    plt.grid(False)
    name = get_next_filename("media/optimization/signal_ase_profile", "pdf", use_active_naming=True)
    plt.savefig(name, bbox_inches='tight', pad_inches=0.01)
    print(f"Plot saved as {name}")
    plt.clf()
    #
    plt.figure(figsize=(4, 3))
    cmap = viridis
    z_plot = np.linspace(0, cf.fiber_length, len(pump_solution[:, 0, 0])) * 1e-3  
    #
    for i in range(cf.n_modes):
        plt.plot(z_plot,
                 watt2dBm(pump_solution[:, :, i]), color=cmap(i / cf.n_modes + 0.2), alpha=0.2)
    plt.grid(False)
    plt.ylabel(r"$P$ [dBm]")
    plt.xlabel(r"$z$ [km]")
    # plt.legend()
    plt.tight_layout()
    name = get_next_filename("media/optimization/pump_profile", "pdf", use_active_naming=True)
    plt.savefig(name)
    print(f"Plot saved as {name}")
    #
    fiber_len = getattr(cf, "fiber_length", None)
    loss = -0.2e-3 * fiber_len if fiber_len is not None else 0.0
    launch_dbm = cf.launch_power if getattr(cf, "launch_power", None) is not None else 0.0
    raman_gain = cf.raman_gain if getattr(cf, "raman_gain", None) is not None else 0.0
    on_off_gain = -loss + raman_gain
    plt.clf()
    plt.figure(figsize=(4, 3))
    for i in range(cf.n_modes):
        plt.plot(signal_wavelengths * 1e6,
                 watt2dBm(signal_solution[-1, :, i]) - launch_dbm - loss, 
                 label=mode_labels[i], 
                 color=cmap(i / cf.n_modes + 0.2))
    plt.legend()
    plt.axhline(on_off_gain, ls="--", color="black")
    plt.xlabel(r"Channel Wavelength [$\mu$ m]")
    plt.ylabel("On Off Gain [dB]")
    plt.tight_layout()
    name = get_next_filename("media/optimization/flatness", "pdf", use_active_naming=True)
    plt.savefig(name)
    print(f"Plot saved as {name}")
    return


def analyze_optimization(
  signal_wavelengths, 
  signal_solution, # in Watt
  ase_solution, # in Watt
  pump_wavelengths,
  pump_solution, # in Watt
  pump_powers, # in Watt
  cf):
  """Print quick metrics to judge optimization quality and noise impact."""
  signal_solution_dBm = watt2dBm(signal_solution)
  pump_solution_dBm = watt2dBm(pump_solution)
  flatness = np.max(signal_solution_dBm[-1, :, :]) - np.min(signal_solution_dBm[-1, :, :])
  approx_loss = -0.2e-3 * cf.fiber_length
  avg_pump_power_0 = watt2dBm(np.mean(dBm2watt(pump_solution_dBm[0, :, :])))
  avg_pump_power_L = watt2dBm(np.mean(dBm2watt(pump_solution_dBm[-1, :, :])))
  print(f"\n{'Optimization metric':<30} | {'Value':>10}")
  print("-" * 43)
  print(f"{'Flatness':<30} | {flatness:7.3f} dB")
  print(f"{'Attenuation':<30} | {approx_loss:7.3f} dB")
  try:
    ase_solution_dBm = watt2dBm(ase_solution)
    avg_ase = watt2dBm(np.mean(dBm2watt(ase_solution_dBm[-1, :, :])))
    print(f"{'Average ASE':<30} | {avg_ase:7.3f} dBm")
  except:
    print(f"got data without ASE.")
    pass
  print(f"{'Average pump power at z=0':<30} | {avg_pump_power_0:7.3f} dBm")
  print(f"{'Average pump power at z=L':<30} | {avg_pump_power_L:7.3f} dBm")
  print("pump configuration for copy-paste not shown")
  # print(f" ° Wavel [m] : {repr(pump_wavelengths)}")
  # print(f" ° Pow. [dBm] : {repr(pump_powers)}")
  return
