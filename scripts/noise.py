"""
Finds the overall noise on a single channel 
given a fiber link configuration.

Generates figs:

- Statistics of channels over DGD
- NLIN thresholding and approximation

"""
import pynlin.fiber
import pynlin.fiber
import pynlin.wdm
import pynlin.pulses
from scripts.modules import cfg
from scripts.modules.time_integrals import do_time_integrals
from scripts.modules.load_fiber_values import *
from scripts.modules.load_fiber_values import load_group_delay
from scripts.modules.collision import plot_illustrative, plot_dispersion_analysis
from scripts.modules.threshold import get_fig2_raman, get_fig2
from scripts.modules.dgd_nlin import noise_plot, noise_histogram

cf = cfg.load_toml_to_struct("./input/config_collision.toml") # config_collisions is useful to print the plot of Marco, illustrative one
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

# print(s_freq * 1e-12)
# print(l_freq * 1e-12)
delta = (s_freq - l_freq) * 1e-12
avg = ((s_freq + l_freq) * 1e-12 / 2)

###########
fig_to_generate = [-1]
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
                      recompute=True)

if 2 in fig_to_generate:
    get_fig2(recompute=False)
    get_fig2_raman(recompute=False)

if 3 in fig_to_generate:
    # TODO May 2025
    # Plot of the total noise (final)
    # noise_plot(dgd_threshold=3e-15,
    #            use_kappa=True,
    #            use_smf=True,
    #            use_fB=True,
    #            use_dBm_scale=True,
    #            use_plot_without_x_mode=False)

    # This plots the same results as in the paper, but about 2dB lower
    # OK, it seems that the issue is that we were using -1.5dBm launch power for the prefactor

    noise_plot(use_kappa=True,
               use_smf=True,
               use_fB=True,
               use_dBm_scale=True,
               use_plot_without_x_mode=False)

    noise_plot(use_kappa=True,
               use_smf=True,
               use_fB=True,
               use_dBm_scale=True,
               use_plot_without_x_mode=True)

if 4 in fig_to_generate:
    noise_histogram(dgd_threshold=3e-15,
                    use_kappa=True,
                    use_smf=True,
                    use_fB=True,
                    use_dBm_scale=True)