"""
Finds the overall noise on a single channel 
given a fiber link configuration.

Generates figs:

- Statistics of channels over DGD
- NLIN thresholding and approximation

"""
from scripts.modules import cfg
from scripts.modules.load_fiber_values import *
from scripts.modules.load_fiber_values import load_group_delay
from scripts.modules.collision import plot_illustrative, plot_dispersion_analysis
from scripts.modules.validation import get_nlin_threshold
from scripts.modules.system_nlin import plot_case_study_noise, plot_case_study_noise_histogram

from loguru import logger as lg
from scripts.modules.log_init import init_logging
init_logging()

# config_collisions is useful to print the plot of Marco, illustrative one
cf = cfg.load_toml_to_struct("./input/config_collision.toml")
oi_fit = np.load('results/oi_fit.npy')
oi_avg = np.load('results/oi_avg.npy')

beta2 = -pynlin.utils.dispersion_to_beta2(
    cf.dispersion, 1550e-9
)

wdm = pynlin.wdm.WDM(
    spacing=cf.channel_spacing,
    num_channels=cf.n_channels,
    center_frequency=cf.center_frequency
)

fiber = pynlin.fiber.MMFiber(
    effective_area=80e-12,
    overlap_integrals=oi_fit,
    group_delay=load_group_delay(),
    length=cf.fiber_length,
    n_modes=4
)
freqs = wdm.frequency_grid()

s_limit = 1460e-9
l_limit = 1625e-9
s_freq = 3e8 / s_limit
l_freq = 3e8 / l_limit
delta = (s_freq - l_freq) * 1e-12
avg = ((s_freq + l_freq) * 1e-12 / 2)

# ---------------------------------------------
# PLOTTING
# ---------------------------------------------
fig_to_generate = [2]

if -1 in fig_to_generate:
    plot_dispersion_analysis(fiber,
                             wdm,
                             cf,
                             recompute=False)
if 1 in fig_to_generate:
    # the plot of Marco
    plot_illustrative(fiber,
                      wdm,
                      cf,
                      recompute=False)
if 2 in fig_to_generate:
    get_nlin_threshold(recompute=True, 
                       use_fB=True, 
                       fB_simple_interpolation=False)
if 3 in fig_to_generate:
    plot_case_study_noise(use_kappa=True,
               use_smf=True,
               use_fB=True,
               use_dBm_scale=True,
               use_plot_without_x_mode=False)
    plot_case_study_noise(use_kappa=True,
               use_smf=True,
               use_fB=True,
               use_dBm_scale=True,
               use_plot_without_x_mode=True)
if 4 in fig_to_generate:
    plot_case_study_noise_histogram(dgd_threshold=3e-15,
                    use_kappa=True,
                    use_smf=True,
                    use_fB=True,
                    use_dBm_scale=True)