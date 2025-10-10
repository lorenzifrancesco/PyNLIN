import numpy as np
from typing import Generator, List, Tuple
from pynlin.fiber import Fiber
import math
from pynlin.fiber import Fiber, SMFiber, MMFiber
from pynlin.wdm import WDM
from pynlin.pulses import Pulse

def get_interfering_frequencies(
    channel_of_interest: float,
    frequency_grid: np.ndarray,
) -> List[float]:
    """Given a channel frequency and an  iterable of frequencies, generate a
    list of interfering channel frequencies."""
    combinations = []
    for x in frequency_grid:
        if x != channel_of_interest:
            combinations.append(x)
    return combinations

def get_frequency_spacing(a_chan, b_chan, wdm):
  frequency_grid = wdm.frequency_grid()
  try:
    return frequency_grid[b_chan[1]] - frequency_grid[a_chan[1]]
  except:
    # print("Inadequate channel choice to calculate spacing...")
    return 0
  
def get_m_values(
    fiber: Fiber,
    pulse: Pulse,
    partial_collisions_start:int,
    dgd: float 
) -> np.ndarray:
    """Get values of the m indeces to compute the X0mm XPM coefficients for.

    Computes those indeces for which the collisions fall inside the
    fiber. By default, 10 extra partial collisions at each end of the
    fiber are computed. This parameter can be controlled by the
    `partial_collisions_start` and `partial_collisions_end` kwargs.
    """
    partial_collisions_end = partial_collisions_start
    m_max = -(fiber.length * dgd) * pulse.baud_rate
    if m_max < 0:
        m_max = math.ceil(m_max)
        return np.arange(m_max - partial_collisions_start, partial_collisions_end + 1)
    else:
        m_max = math.floor(m_max)
        return np.arange(-partial_collisions_start, m_max + partial_collisions_end + 1)


def get_collision_location(m, 
                           pulse: Pulse,
                           dgd:float) -> float:
    if dgd == 0:
        assert m == 0, "m should be zero if dgd is zero"
        return 0.0
    return -m / (pulse.baud_rate * dgd)
  
  
def get_dgd(a_chan, b_chan, fiber, wdm) -> float:
    freq_grid = wdm.frequency_grid()
    if isinstance(fiber, SMFiber):
        assert (a_chan[0] == 0 and b_chan[0] == 0)
        return fiber.beta2 * 2 * np.pi * (freq_grid[b_chan[1]] - freq_grid[a_chan[1]])
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta1(b_chan[0], freq_grid[b_chan[1]]) \
      - fiber.group_delay.evaluate_beta1(a_chan[0], freq_grid[a_chan[1]])


def get_gvd(b_chan, fiber, wdm) -> float:
    if isinstance(fiber, SMFiber):
        return fiber.beta2
    elif isinstance(fiber, MMFiber):
        return fiber.group_delay.evaluate_beta2(b_chan[0], wdm.frequency_grid()[b_chan[1]])
    pass
  
  
def get_z_walkoff(
  pulse: Pulse,
  dgd: float):
  if dgd != 0:
    return np.abs(1 / (pulse.baud_rate * dgd))
  else:
    return 1e100 # very large number