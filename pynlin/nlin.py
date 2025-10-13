import functools
import math
from typing import Tuple, List
from numba import njit, prange

import h5py
import numpy as np

import scipy.integrate
import tqdm
from scipy.constants import nu2lambda
from tqdm.contrib.concurrent import process_map
from itertools import product

from pynlin.fiber import Fiber, SMFiber, MMFiber
from pynlin.pulses import Pulse, RaisedCosinePulse, GaussianPulse, NyquistPulse
from pynlin.wdm import WDM
from pynlin.collisions import get_interfering_frequencies, get_m_values, get_frequency_spacing, get_collision_location, get_z_walkoff, get_dgd, get_gvd
import time
from pynlin.utils import beta2rms

from loguru import logger as lg
from pynlin.log_init import init_logging
init_logging()


def get_interfering_channels(a_chan: Tuple, wdm: WDM, fiber: Fiber):
    b_chans = list(product(range(fiber.n_modes), range(wdm.num_channels)))
    b_chans.remove(a_chan)
    return b_chans


def time_integrals_all_b_chans(
    wdm: WDM,
    fiber: Fiber,
    a_chan: Tuple,
    pulse: Pulse,
    filename: str,
    overwrite=False,
    **compute_collisions_kwargs,
) -> None:
    """Compute the inner time integral of the expression for the XPM
    coefficients Xhkm for each combination of frequencies in the supplied WDM
    grid."""
    if isinstance(fiber, SMFiber):
        assert (a_chan[0] == 0)

    append_write = "a"
    found = False
    try:
        with h5py.File(filename, 'r') as file:
            for gg in file["time_integrals"]:
                lg.trace(gg)

            if f"a_chan_{a_chan}" in file["time_integrals"]:
                lg.trace("A-chan group already present on file. Nothing to do.")
                found = True
    except FileNotFoundError:
        lg.trace(f"File {filename} not found. Creating a new file.")
        append_write = "w"

    if overwrite and found:
        lg.trace(
            "\033[91m warn: \033[0m overwriting by deleting and rewriting all the results file!")
        append_write = "w"
    elif found and not overwrite:
        return -1

    lg.trace("No groups found for this A channel, calculating...")
    file = h5py.File(filename, append_write)

    frequency_grid = wdm.frequency_grid()
    a_freq = frequency_grid[a_chan[1]]
    group_name = f"time_integrals/a_chan_{a_chan}/"
    group = file.create_group(group_name)
    group.attrs["mode"] = a_chan[0]
    group.attrs["frequency"] = a_freq

    b_channels = get_interfering_channels(
        a_chan, wdm, fiber,
    )

    # iterate over all channels of the WDM grid
    for b_num, b_chan in enumerate(b_channels):
        # set up the progress bar to iterate over
        # all interfering channels of the current channel of interest
        pbar = tqdm.tqdm(b_channels)
        pbar.set_description(
            f"A-chan: {a_chan}, B-chan = {b_chan}"
        )

        z, I, M = compute_all_collisions_time_integrals_system(
            a_chan,
            b_chan,
            fiber,
            wdm,
            pulse,
            **compute_collisions_kwargs,
        )

        # in each COI group, create a group for each interfering channel
        # and store the z-array (positions inside the fiber)
        # and the time integrals for each collision.
        b_freq = frequency_grid[b_chan[1]]
        interferer_group_name = group_name + \
            f"b_chan_{b_chan}/"
        interferer_group = file.create_group(interferer_group_name)
        interferer_group.attrs["frequency"] = b_freq
        interferer_group.attrs["mode"] = b_chan[0]

        file.create_dataset(interferer_group_name + "z", data=z)
        file.create_dataset(interferer_group_name + "m", data=M)
        # integrals_group_name = interferer_group_name + "/integrals/"
        # for x, integral in enumerate(I_list):
        file.create_dataset(
            interferer_group_name + f"integrals",
            data=I,
            compression="gzip",
            compression_opts=9,
        )

def compute_all_collisions_time_integrals_system(
    a_chan: Tuple[int, int],
    b_chan: Tuple[int, int],
    fiber: Fiber,
    wdm: WDM,
    pulse: Pulse,
    use_multiprocessing: bool = True,
    partial_collisions_margin: int = 5,
    speedup_pulse_propagation=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    f_grid = wdm.frequency_grid()
    spacing = get_frequency_spacing(a_chan, b_chan, wdm)
    dgd = 0.0
    gvda = 0.0
    gvdb = 0.0
    if fiber.fiber_type == "SMF":
        dgd = fiber.beta2 * spacing
        gvda = fiber.beta2
        gvdb = fiber.beta2
    elif fiber.fiber_type == "MMF":
        dgd = fiber.group_delay.evaluate_beta1(b_chan[0], f_grid[b_chan[1]]) - fiber.group_delay.evaluate_beta1(a_chan[0], f_grid[a_chan[1]])
        gvda = fiber.group_delay.evaluate_beta2(a_chan[0], f_grid[a_chan[1]])
        gvdb = fiber.group_delay.evaluate_beta2(b_chan[0], f_grid[b_chan[1]])
    z_walkoff = get_z_walkoff(pulse, dgd)
    
    lg.trace(f"dispersion data: gvda = {gvda:.3e}, gvdb = {gvdb:.3e}, dgd = {dgd:.3e}")
    lg.debug("==============")
    if dgd is None:
        lg.debug(
            f"a: {a_chan}, b: {b_chan}, a_freq:{f_grid[a_chan[1]]:.5e}, b_freq:{f_grid[b_chan[1]]:.5e}")
        lg.debug(
            f"dgd: {get_dgd(a_chan, b_chan, fiber, wdm):.3e}, z_w: {z_walkoff:.3e}, lenght/z_w:{fiber.length/z_walkoff:.5e}")
    else:
        lg.debug(f"set dgd:{dgd:.2e}, z_walkoff/L = {z_walkoff/fiber.length}")
    lg.debug("==============")
    
    # assert pulse.baud_rate == 35e9
    # assert fiber.length == 70e3
    # assert wdm.num_channels == 250
    # assert wdm.spacing == 40e9
    # assert wdm.central_frequency == 195.94e12
    return compute_all_collisions_time_integrals(
        fiber, 
        pulse,
        dgd,
        gvda, 
        gvdb, 
        use_multiprocessing=use_multiprocessing,
        partial_collisions_margin=partial_collisions_margin)


def compute_all_collisions_time_integrals(
    fiber: Fiber, # only used for the lenght
    pulse:Pulse,
    dgd:float,
    gvda: float, 
    gvdb: float,
    use_multiprocessing: bool = True,
    partial_collisions_margin: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the integrals for all the collisions for the specified pair of
    channels.

    Returns
    -------
    z: np.ndarray
        The sampling points along the fiber
    time_integrals: np.ndarray
        Time integral as a function of the fiber position, one for each collision
    m: np.ndarray
        Array of collision indeces
    """
    z_walkoff = get_z_walkoff(pulse, dgd)
    m_list = get_m_values(
        fiber,
        pulse,
        partial_collisions_margin,
        dgd
    )[::-1]
    # ---
    # estimation of the useful z range 
    # ---
    def i_func(m, z): return m_th_time_integral_Gaussian(
        m,
        z,
        pulse,
        dgd,
        gvda, 
        gvdb
    )
    # Estimate the significant range of each collision:
    #   fill the z_axis_grid with good estimates of
    #   where the pulses start and end the collision
    #   lg.debug(get_frequency_spacing(a_chan, b_chan, wdm))
    n_rough_grid = 50
    n_z_points = 200
    margin = 5
    z_axis_list = []
    if isinstance(pulse, NyquistPulse):
        #   lg.debug("\033[91m warn: \033[0m The pulse is Nyquist (long-tailed): overriding the number of points!")
        n_z_points = 200
        margin = 5
        
    for m in m_list:
        z_m = get_collision_location(m, pulse, dgd)
        z_min = z_m - (z_walkoff / 2 * margin)
        z_max = z_m + (z_walkoff / 2 * margin)
        # lg.debug(f"BEFORE: zmin/L = {z_min/fiber.length:.4e}, zmax/L = {z_max/fiber.length:.4e}")
        i_sample_z_m = i_func(m, z_m)
        dz = z_walkoff / n_rough_grid
        threshold = i_sample_z_m / 100
        # lg.debug(f"threshold = {threshold:.4e}")
        z = 0
        i_sample = i_func(m, z)
        i_sample = i_func(m, z_min)
        # lg.debug(f"left sample = {i_sample:.4e}")
        z = 0
        while i_sample > threshold and z_min > 0:
            z_min -= dz
            i_sample = i_func(m, z_min)

        i_sample = i_func(m, z_max)
        # lg.debug(f"right sample = {i_sample:.4e}")
        while i_sample > threshold and z_max < fiber.length:
            z_max += dz
            i_sample = i_func(m, z_max)
        z_min = max(z_min, 0)
        z_max = min(z_max, fiber.length)
        if z_min > z_max:
            # lg.debug(f"\033[91m warn: \033[0m delimitation of integral diverged at m = {m:10d}, z_m = {z_m: 5.4e}!")
            # lg.debug("    Failure to set the pulse region")
            z_axis_list.append(np.linspace(0, fiber.length, n_z_points))
        else:
            z_axis_list.append(np.linspace(z_min, z_max, n_z_points))
        lg.debug(
                f"    z_axis = ({z_axis_list[-1][0]:.2e}, {z_axis_list[-1][-1]:.2e}, {len(z_axis_list[-1]):5d})")
    if pulse.num_symbols != n_z_points:  # does this make sense?
        lg.warning(f"\033[91m warn: \033[0m pulse num_symbols ({pulse.num_symbols}) != n_z_points ({n_z_points})!")
        lg.warning("\033[91m warn: \033[0m overriding the pulse number of samples!")
        pulse.num_symbols = n_z_points
    lg.debug("  Done.")
    
    # --- 
    # parallel compute the integrals
    # ---
    # # build a partial function otherwise multiprocessing
    # complains about not being able to pickle stuff
    partial_function = functools.partial(
        m_th_time_integral, pulse=pulse, dgd=dgd, gvda=gvda, gvdb=gvdb
    )
    lg.debug("Computing the integrals for every m...")
    start = time.time()
    if use_multiprocessing:
        integrals_list = process_map(
            partial_function, m_list, z_axis_list, leave=False, chunksize=1, max_workers=14,
            desc=f"Iterating over m values {len(m_list)} total, {partial_collisions_margin} margins on each size",
        )
    else:
        integrals_list = process_map(
            partial_function, m_list, z_axis_list, leave=False, chunksize=1, max_workers=1
        )
    end = time.time()
    lg.debug(f"  Done in {(end-start)*1e3:.0e} ms.")

    # convert the list of arrays in a 2d array, since the shape is the same
    z_axis_list_2d = np.stack(z_axis_list)
    integrals_list_2d = np.stack(integrals_list)
    return z_axis_list_2d, integrals_list_2d, m_list


# ---------------------------------------------
#  Fundamental time integrals
# ---------------------------------------------
# Multiprocessing wrapper
def m_th_time_integral(
    m: int,
    z: np.ndarray, # we would have great advantage in doing a dynamical allocation here.
    pulse: Pulse,
    dgd,  # for manual operation of the DGD
    gvda=None,  # additional parameters for full specification
    gvdb=None,
):
    lg.trace(f"Computing integral for m = {m:10d}, z = ({z[0]:.2e}, {z[-1]:.2e}, {len(z):5d})")
    lg.trace(f"  gvda = {gvda:.3e}, gvdb = {gvdb:.3e}, dgd = {dgd:.3e}")
    if isinstance(pulse, GaussianPulse):
        return m_th_time_integral_Gaussian(
            m, z, pulse, dgd, gvda, gvdb)
    else:
        return m_th_time_integral_general(m, z, pulse, dgd, gvda, gvdb)

def m_th_time_integral_Gaussian(
    m: int,
    z: np.ndarray,
    pulse: Pulse,
    dgd: float,
    gvda: float,
    gvdb: float,
) -> float:
    # Apply the fully analytical formula
    rms_gvd = beta2rms(gvda, gvdb)
    if rms_gvd == 0:
        rms_ld = 1e100 # very large number
    else:
        rms_ld = 1 / (np.abs(rms_gvd) * (pulse.baud_rate)**2)
    factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
    factor2 = 1 / np.sqrt(1 + (z / rms_ld)**2)
    exponent = -((m + pulse.baud_rate * dgd * z)**2) / \
        (2 * (1 + (z / rms_ld)**2))
    with np.errstate(over='ignore'): # prevent underflow
        total = factor1 * factor2 * np.exp(exponent)
    return total


def m_th_time_integral_Nyquist(
    m: int,
    z: np.ndarray,
    pulse: Pulse,
    dgd:float,
    gvda:float,
    gvdb:float,
) -> float:
    # Nakazawa formula for propagation and then integration??
    # Integrate in spectral domain?
    raise (NotImplementedError) # here we suppose that the MMF case easily includes the SMF one
    l_da = 1 / \
        (fiber.group_delay.evaluate_beta2(
            a_chan[0], wdm.frequency_grid()[a_chan[1]])(pulse.baud_rate)**2)
    l_db = 1 / \
        (fiber.group_delay.evaluate_beta2(
            b_chan[0], wdm.frequency_grid()[b_chan[1]])(pulse.baud_rate)**2)
    dgd = fiber.group_delay.evaluate_beta1(b_chan[0], wdm.frequency_grid(
    )[b_chan[1]]) - fiber.group_delay.evaluate_beta1(a_chan[0], wdm.frequency_grid()[a_chan[1]])
    avg_l_d = (l_da * l_db) / (l_da + l_db) / 2
    factor1 = pulse.baud_rate / (np.sqrt(2 * np.pi))
    factor2 = 1 / np.sqrt(1 + (z / avg_l_d)**2)
    exponent = -((m / pulse.baud_rate + dgd * z)**2) / \
        (2 * (1 + (z / avg_l_d)**2))
    return factor1 * factor2 * np.exp(exponent)

# apply_chromatic take channel inside
# @njit(parallel=False, cache=True)
def m_th_time_integral_general(
    m: int,
    z_axis: np.ndarray,
    pulse: Pulse,
    dgd : float,
    gvda: float,
    gvdb: float,
) -> np.ndarray:
    I_list = np.zeros_like(z_axis, dtype=np.complex64)
    dt = pulse.T0/pulse.samples_per_symbol
    for iz in prange(len(z_axis)):
        delay = m / pulse.baud_rate + dgd * z_axis[iz]
        g1 = apply_chromatic_dispersion(
            gvda, pulse, z_axis[iz] , 0.0)
        g2 = np.conj(g1)
        g3 = apply_chromatic_dispersion(
            gvdb, pulse, z_axis[iz], delay)
        g4 = np.conj(g3)
        I_list[iz] = scipy.integrate.trapezoid(g1 * g2 * g3 * g4, dx=dt)
    return I_list


def X0mm_space_integral(
        z: np.ndarray, time_integrals, amplification_function=None, axis=-1) -> np.ndarray:
    """Compute the X0mm XPM coefficients specifying the inner time integral as
    input.

    Useful to compare different amplification schemes without re-
    computing the time integral.
    """
    if callable(amplification_function):
        X = scipy.integrate.trapezoid(
            time_integrals * amplification_function(z), z, axis=axis)
    else:
        # if the amplification function is not supplied, assume perfect distributed
        # amplification
        amplification_function = np.ones_like(z)
        X = scipy.integrate.trapezoid(
            time_integrals * amplification_function, z, axis=axis)
    return X


# essential function
def apply_chromatic_dispersion(
        gvd: float, 
        pulse: Pulse, 
        z: float, 
        delay: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return the propagated pulse shape.
    Optionally apply a delay in time.
    """

    g, t = pulse.data()
    dt = t[1] - t[0]
    nsamples = len(g)
    freq = np.fft.fftfreq(nsamples, d=dt)
    omega = 2 * np.pi * freq
    omega = np.fft.fftshift(omega)
    gf = np.fft.fftshift(np.fft.fft(g))

    propagator = -1j * gvd / 2 * omega**2 * z
    delay = np.exp(-1j * delay * omega)

    gf_propagated = gf * np.exp(propagator) * delay
    g_propagated = np.fft.ifft(np.fft.fftshift(gf_propagated))

    return g_propagated
