import logging
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
import seaborn as sns
import pynlin.wdm
from pynlin.utils import nu2lambda
from scripts.modules.load_fiber_values import load_group_delay, load_dummy_group_delay
from numpy import polyval
from pynlin.fiber import *
from pynlin.pulses import *
from pynlin.nlin import compute_all_collisions_time_integrals, get_dgd, X0mm_space_integral, get_gvd
from matplotlib.gridspec import GridSpec
import json
from pynlin.collisions import get_m_values, get_collision_location

rc('text', usetex=True)
logging.basicConfig(filename='MMF_optimizer.log', encoding='utf-8', level=logging.INFO)
log = logging.getLogger(__name__)
log.debug("starting to load sim_config.json")
f = open("./input/sim_config.json")

data = json.load(f)
dispersion = data["dispersion"]
effective_area = data["effective_area"]
baud_rate = data["baud_rate"]
fiber_lengths = data["fiber_length"]
num_channels = data["num_channels"]
interfering_grid_index = data["interfering_grid_index"]
channel_spacing = data["channel_spacing"]
center_frequency = data["center_frequency"]
store_true = data["store_true"]
pulse_shape = data["pulse_shape"]
partial_collision_margin = data["partial_collision_margin"]
num_co = data["num_co"]
num_ct = data["num_ct"]
wavelength = data["wavelength"]
special = data["special"]
pump_direction = data["pump_direction"]
num_only_co_pumps = data['num_only_co_pumps']
num_only_ct_pumps = data['num_only_ct_pumps']
gain_dB_setup = data['gain_dB_list']
gain_dB_list = np.linspace(gain_dB_setup[0], gain_dB_setup[1], gain_dB_setup[2])
power_dBm_setup = data['power_dBm_list']
power_dBm_list = np.linspace(power_dBm_setup[0], power_dBm_setup[1], power_dBm_setup[2])
num_modes = data['num_modes']
oi_fit = np.load('oi_fit.npy')
oi_avg = np.load('oi_avg.npy')
use_avg_oi = False

grid = False
wdm = pynlin.wdm.WDM(
    spacing=channel_spacing,
    num_channels=num_channels,
    center_frequency=center_frequency
)
beta1_params = load_group_delay()
dummy_fiber = MMFiber(
    effective_area=80e-12,
    overlap_integrals=oi_fit,
    group_delay=beta1_params,
    length=100e3,
    n_modes=4
)
freqs = wdm.frequency_grid()
modes = [0, 1, 2, 3]
mode_names = ['LP01', 'LP11', 'LP21', 'LP02']
beta1 = np.zeros((len(modes), len(freqs)))
for i in modes:
    beta1[i, :] = dummy_fiber.group_delay.evaluate_beta1(i, freqs)
beta2 = np.zeros((len(modes), len(freqs)))
for i in modes:
    beta2[i, :] = dummy_fiber.group_delay.evaluate_beta2(i, freqs)
beta1 = np.array(beta1)
beta2 = np.array(beta2)

signal_powers = np.load("results/signal_power.npy")
print(np.shape(signal_powers))
fB_max = np.max(signal_powers, axis=(1, 2))
fB_min = np.min(signal_powers, axis=(1, 2))
fB_max /= fB_max[0]
fB_min /= fB_min[0]

z_axis = np.linspace(0, dummy_fiber.length, len(fB_max))
dz = z_axis[1] - z_axis[0]
coeffs_max = np.polyfit(z_axis, fB_max, 6)
coeffs_min = np.polyfit(z_axis, fB_min, 6)

fB_integral_max = np.sum(fB_max) * dz / dummy_fiber.length
fB_integral_min = np.sum(fB_min) * dz / dummy_fiber.length

def fB_max_function(z):
  return np.polyval(coeffs_max, z)

def fB_min_function(z): 
  return np.polyval(coeffs_min, z)

def get_space_integrals_max(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = X0mm_space_integral(z, I, amplification_function=fB_max_function)
    return X0mm

def get_space_integrals_min(m, z, I):
    '''
      Read the time integral file and compute the space integrals 
    '''
    X0mm = X0mm_space_integral(z, I, amplification_function=fB_min_function)
    return X0mm


dpi = 300

dgd1 = 1e-16
dgd2g = 1e-11
dgd2n = 1e-15
n_samples_numeric_g = 10
n_samples_numeric_n = 3
px = 0
gvds = [-35e-27]

print(f"Computing the channel-pair NLIN coefficient insides [{dgd1*1e12:.1e}, {dgd2g*1e12:.1e}] ps/m ")
for gvd in gvds:
  for px in [0]:
    if px == 0:
      pulse = GaussianPulse(
          baud_rate=baud_rate,
          num_symbols=1e2,
          samples_per_symbol=2**5,
          rolloff=0.0,
      )
    else:
      pulse = NyquistPulse(
          baud_rate=baud_rate,
          num_symbols=1e3,
          samples_per_symbol=2**5,
          rolloff=0.0,
      )
      
    n_samples_analytic = 500
    dgd1 = 1e-17
    if px == 0:
      dgd2 = dgd2g
      n_samples_numeric = n_samples_numeric_g
    else: 
      dgd2 = dgd2n
      n_samples_numeric = n_samples_numeric_n
    # 6e-9 for our fiber
    dgds_numeric = np.logspace(np.log10(dgd1), np.log10(dgd2), n_samples_numeric)
    dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)

    zwL_numeric = 1 / (pulse.baud_rate * dgds_numeric * dummy_fiber.length)
    zwL_analytic = 1 / (pulse.baud_rate * dgds_analytic * dummy_fiber.length)

    nlin_max = np.zeros(n_samples_numeric)
    nlin_min = np.zeros(n_samples_numeric)
    a_chan = (-1, -10)
    b_chan = (-1, -100)
    if False:
        for id, dgd in enumerate(dgds_numeric):
            z, I, m = compute_all_collisions_time_integrals(
                a_chan, b_chan, dummy_fiber, wdm, pulse, dgd, gvd)
            # space integrals
            X0mm_max = get_space_integrals_max(m, z, I)
            X0mm_min = get_space_integrals_min(m, z, I)
            nlin_max[id] = np.sum(X0mm_max**2)
            nlin_min[id] = np.sum(X0mm_min**2)
        if px == 0:
          np.save("results/partial_nlin_gaussian_max.npy", nlin_max)
          np.save("results/partial_nlin_gaussian_min.npy", nlin_min)
        else:
          np.save("results/partial_nlin_nyquist_max.npy", nlin_max)
          np.save("results/partial_nlin_nyquist_min.npy", nlin_min)

T = 100e-12
L = dummy_fiber.length
LD_eff = pulse.T0**2/np.abs(gvd)

dgd2 = dgd2g
dgds_analytic = np.linspace(dgd1, dgd2, n_samples_analytic)
analytic_nlin = L / (T * dgds_analytic)
analytic_nlin[analytic_nlin > 1e30] = np.nan
n_samples_numeric = 10
dgds_numeric_g = np.logspace(np.log10(dgd1), np.log10(dgd2g), n_samples_numeric_g)
n_samples_numeric = 3
dgds_numeric_n = np.logspace(np.log10(dgd1), np.log10(dgd2n), n_samples_numeric_n)

fig = plt.figure(figsize=(5, 3.5))  

def antonio_rescale_max(dgd):
  acc = 0.0
  for m in get_m_values(dummy_fiber, wdm, a_chan, b_chan, T, 0, dgd):
    acc += fB_max_function(get_collision_location(m, dummy_fiber, wdm, a_chan, b_chan, pulse, dgd))**2
  return acc

def antonio_rescale_min(dgd):
  acc = 0.0
  for m in get_m_values(dummy_fiber, wdm, a_chan, b_chan, T, 0, dgd):
    acc += fB_min_function(get_collision_location(m, dummy_fiber, wdm, a_chan, b_chan, pulse, dgd))**2
  return acc

# TODO this is very slow
nlin_analytic_max = np.zeros_like(dgds_analytic)
nlin_analytic_min = np.zeros_like(dgds_analytic)
for ix, i in enumerate(dgds_analytic):
  nlin_analytic_max[ix] = antonio_rescale_max(i)/(i**2) * 1e-30
  nlin_analytic_min[ix] = antonio_rescale_min(i)/(i**2) * 1e-30

plt.plot(dgds_analytic * 1e12, nlin_analytic_min, lw = 1, color='red')
plt.plot(dgds_analytic * 1e12, nlin_analytic_max, lw = 1, color='red')

# plt.plot(dgds_analytic * 1e12, np.sum(fB_min_function(get_collision_location(get_m_values(dummy_fiber, wdm, a_chan, b_chan, T, 0, dgds_analytic), dummy_fiber, wdm, a_chan, b_chan, pulse, dgds_analytic))**2)
#          * analytic_nlin * 1e-30, lw = 1, color='red')
# plt.plot(dgds_analytic * 1e12, analytic_nlin * 1e-30, lw = 1, color='red', ls="--")
# Fra LOWER
# plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * 
#          (LD_eff/ (T * np.sqrt(2 * np.pi)) * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', lw=1, label=r'$N^<$')
# plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * 
#          (L/ (T * np.sqrt(2 * np.pi)))**2 * 1e-30, color='blue', lw=1, label=r'$N^<$')
# Fra UPPER
plt.plot(dgds_analytic * 1e12, fB_integral_max**2 * np.ones_like(dgds_analytic) * 1.77 * (LD_eff/ (T * np.sqrt(2 * np.pi))
         * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', lw=1, ls=":", label=r'$N^>$')
plt.plot(dgds_analytic * 1e12, fB_integral_min**2 * np.ones_like(dgds_analytic) * 1.77 * (LD_eff/ (T * np.sqrt(2 * np.pi))
         * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', lw=1, ls=":", label=r'$N^>$')

# # Fra Series
# plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * (LD_eff/ (T * np.sqrt(2 * np.pi))
#          * np.arcsinh(L / LD_eff))**2 * 1e-30, color='blue', ls=":", label=r'$N^>$')
# plt.plot(dgds_analytic * 1e12, np.ones_like(dgds_analytic) * 4/9, color='green', ls="--", lw=1, label='Marco')
# plt.scatter(dgds_numeric_g * 1e12, partial_nlin_gaussian * 1e-30,
#             color='green', label='Gaussian', marker="x")

# plt.scatter(dgds_numeric_n * 1e12, partial_nlin_nyquist * 1e-30,
#             color='green', label='Nyq.'+str(-35e-27), marker="x")

lowest_dgd = 0.0
for mode in ["max", "min"]:
  for ix, gvd in enumerate(gvds):
    partial_B2g = (np.load("results/partial_nlin_gaussian_"+mode+".npy"))
    if ix == 0:
        lowest_dgd = partial_B2g[0]
    print(partial_B2g)
    plt.scatter(dgds_numeric_g * 1e12, partial_B2g * 1e-30,
                label='Gauss.'+str(gvd), color="blue", marker="x")
    
    
    # partial_B2n = (np.load("results/partial_nlin_nyquist_"+mode+".npy"))
    # plt.scatter(dgds_numeric_n * 1e12, partial_B2n * 1e-30,
    #             label='Nyq.'+str(gvd), color="green", marker="x")

print(f"DGD low. num = {lowest_dgd:.3e}, ra < = {(L/ (T * np.sqrt(2 * np.pi)))**2:.3e}")

plt.xlabel('DGD [ps/m]')
# plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'channel-pair NLIN [km$^2$/ps$^2$]')
plt.tight_layout()
plt.savefig(f"media/dispersion/partial_NLIN_fB.png", dpi=dpi)

# TODO start again from here!!