"""
Optimize the pumps and calculate signal and ASE power evolution

Generates figs: 

- Power profiles

"""
import os
import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.constants import lambda2nu, nu2lambda
from matplotlib.cm import viridis
from scripts.modules.load_fiber_values import load_group_delay
from scripts.modules import cfg

import pynlin
import pynlin.wdm
import pynlin.pulses
import pynlin.nlin
import pynlin.utils
import pynlin.fiber
from pynlin.raman.pytorch.gain_optimizer import GainOptimizer
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier
from pynlin.raman.solvers import MMFRamanAmplifier as NumpyMMFRamanAmplifier
from pynlin.utils import dBm2watt, watt2dBm
import pynlin.constellations
from scripts.modules.plot_optimization import plot_profiles, analyze_optimization

def ct_solver(fiber, 
              wdm, 
              power_per_pump, # dBm
              pump_band_a,
              pump_band_b,
              learning_rate,
              epochs,
              lock_wavelengths,
              batch_size = 1,
              use_precomputed=False,
              optimize=False, 
              use_avg_oi=False
              ):
    """
      Script for solve a single instance of pump optimization.
      All the system variables are set in the config toml file.
      The initial states for optimization are taken as arguments
      
      Input files: 
        config.toml
      
      Output files:
        opt_pump_wavelengths.npy
        opt_pump_powers.npy
    """
    cf = cfg.load_toml_to_struct("./input/config.toml")
    #
    integration_steps = 300
    z_max = np.linspace(0, fiber.length, integration_steps)
    np.save("z_max.npy", z_max)
    #
    print(
        f"> Running optimization for Pin = {cf.launch_power:.2e} dBm, and gain = {cf.raman_gain:.2e} dB.\n")
    if use_precomputed and os.path.exists("results/pump_solution_ct_power" + str(cf.launch_power) + "_opt_gain_" + str(cf.raman_gain) + ".npy"):
        print("Result already computed for power: ",
              cf.launch_power, " and gain: ", cf.raman_gain)
        return
    else:
        print("Computing the power: ", cf.launch_power, " and gain: ", cf.raman_gain)
    initial_pump_frequencies = lambda2nu(
        np.linspace(pump_band_a, pump_band_b, cf.n_pumps))
    #
    signal_wavelengths = wdm.wavelength_grid()
    torch_amplifier_ct = MMFRamanAmplifier(
        fiber.length,
        integration_steps,
        cf.n_pumps,
        signal_wavelengths,
        dBm2watt(cf.launch_power), # W
        fiber,
        counterpumping=True
    )
    #
    if use_precomputed:
        try:
            initial_pump_wavelengths = np.load("results/opt_pump_wavelengths.npy")
            initial_pump_powers = np.load("results/opt_pump_powers.npy")
        except:
            print("The precomputed values misbehave...")
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    ## trivial initializaiton
    initial_pump_wavelengths = nu2lambda(initial_pump_frequencies[:cf.n_pumps])
    initial_pump_powers = np.ones_like(initial_pump_wavelengths) * power_per_pump
    initial_pump_powers = initial_pump_powers.repeat(cf.n_modes, axis=0)
    print(initial_pump_wavelengths.shape)
    print(initial_pump_powers.shape)
    ## initialize to the configuration that is not bad
    initial_pump_wavelengths = np.array([1.3844927e-06, 1.3975118e-06, 1.4131243e-06, 1.4286949e-06, 1.4559689e-06, 1.4575429e-06], dtype=np.float32)
    if cf.n_modes == 4:
      initial_pump_powers = np.array([
        [-31.527662 , -32.781242 , -15.580993 , -18.018877 ],
        [-25.519464 , -23.685694 , -10.617271 , -13.667908 ],
        [-24.974928 , -23.210556 ,  -6.3097696,  -9.496441 ],
        [-25.57634  , -22.774395 ,  -1.7358053,  -4.677509 ],
        [-20.778175 , -17.052673 ,   1.0048871,   5.156198 ],
        [-21.905903 , -19.641636 ,   8.71186  ,   0.5589079]],      dtype=np.float32)
      initial_pump_powers = initial_pump_powers.reshape((24,)) 
    else: 
      initial_pump_powers = np.array([
        [-15.580993 ],
        [-10.617271 ],
        [ -6.309769 ],
        [ -1.735805 ],
        [  1.004887 ],
        [  8.71186  ]],      dtype=np.float32) - 10
      initial_pump_powers = initial_pump_powers.reshape((6,)) 
    # to subtract to the pumps in the 0 dBm launch power setup to prevent RK4 blowup
    if cf.launch_power > -5:
        initial_pump_powers = initial_pump_powers - 3
    print(initial_pump_wavelengths.shape)
    print(initial_pump_powers.shape)
    # adapt to torch
    initial_pump_wavelengths_tensor = torch.from_numpy(initial_pump_wavelengths).to(device)
    initial_pump_powers_tensor =           torch.from_numpy(initial_pump_powers).to(device)
    #
    optimizer = GainOptimizer(
        torch_amplifier_ct,
        initial_pump_wavelengths_tensor,
        initial_pump_powers_tensor, # in dBm
        batch_size=batch_size
    )
    # all in dBm here
    signal_powers = np.ones_like(signal_wavelengths) * cf.launch_power
    signal_powers = signal_powers[:, None].repeat(cf.n_modes, axis=1)
    target_spectrum = signal_powers[None, :, :] + cf.raman_gain
    #
    if optimize:
        pump_wavelengths, pump_powers = optimizer.optimize(
            target_spectrum=target_spectrum,
            epochs=epochs,
            learning_rate=learning_rate,
            lock_wavelengths=lock_wavelengths,
        )
        np.save("results/opt_pump_wavelengths.npy", pump_wavelengths)
        np.save("results/opt_pump_powers.npy", pump_powers)
    else:
        pump_wavelengths = initial_pump_wavelengths
        pump_powers = initial_pump_powers
    #
    amplifier = NumpyMMFRamanAmplifier(fiber)
    pump_powers = pump_powers.reshape((cf.n_pumps, cf.n_modes))
    pump_solution, signal_solution, ase_solution = amplifier.solve( # this should work in Watt
        dBm2watt(signal_powers),
        signal_wavelengths,
        dBm2watt(pump_powers),
        pump_wavelengths,
        z_max,
        fiber,
        ase=False,
        counterpumping=True,
        reference_bandwidth=cf.baud_rate
    )
    #
    return pump_solution, signal_solution, ase_solution, pump_wavelengths, pump_powers


def repropagate_numpy(fiber, 
                signal_wavelengths, 
                pump_wavelengths, 
                pump_powers,
                cf,
                output_file
                ):
    print("Repropagating with Numpy amplifier...")
    amplifier = NumpyMMFRamanAmplifier(fiber)
    pump_powers = pump_powers.reshape((cf.n_pumps, cf.n_modes))
    integration_steps = 300
    z_max = np.linspace(0, fiber.length, integration_steps)
    signal_powers = np.ones_like(signal_wavelengths) * cf.launch_power
    signal_powers = signal_powers[:, None].repeat(cf.n_modes, axis=1)
    pump_sol, signal_sol, ase_sol = amplifier.solve( # this should work in Watt
        dBm2watt(signal_powers),
        signal_wavelengths,
        dBm2watt(pump_powers),
        pump_wavelengths,
        z_max,
        fiber,
        ase=True,
        counterpumping=True,
        reference_bandwidth=nu2lambda(cf.baud_rate), # BEWARE, it is now a bandidth, but a wavelength interval!
        temperature=300,
    )
    print("Done repropagation.")
    variables_dict = {
        name: value 
        for name, value in locals().items() 
        if name in ['pump_sol', 'signal_sol', 'ase_sol', 'pump_wavelengths', 'pump_powers']
    }
    np.save(output_file, variables_dict)
    print("Results saved to file: ", output_file)
    return


if __name__ == "__main__":    
    # Configuration
    recompute   = False
    repropagate = True
    use_smf     = False
    use_avg_oi  = False
    signal_powers = [-5]
    signal_powers = [-2]
    # -10 -> true
    # -5  -> true OI
    # 0   -> true OI
  
    oi_fit = np.load('results/oi_fit.npy')
    oi_avg = np.load('results/oi_avg.npy')
   
    # prepare the definitions of fiber and wdm
    if use_smf:
      cf = cfg.load_toml_to_struct("./input/smf.toml")
    else:
      cf = cfg.load_toml_to_struct("./input/mmf.toml")
    num_original_modes = oi_avg[0].shape[0]
    matrix_avg = oi_avg
    matrix_zeros = np.tile(np.zeros((num_original_modes, num_original_modes))[
                          None, :, :], (5, 1, 1))
    oi_avg_complete = np.stack((*matrix_zeros, matrix_avg), axis=0)
    if use_avg_oi:
        oi_set = oi_avg_complete
    else:
        oi_set = oi_fit
    oi_fit = oi_avg_complete
    #
    fiber = pynlin.fiber.MMFiber(
        effective_area=80e-12,
        n_modes=cf.n_modes,
        overlap_integrals=oi_set,
        group_delay=load_group_delay()
    )
    wdm = pynlin.wdm.WDM(
        spacing=cf.channel_spacing,
        num_channels=cf.n_channels,
        center_frequency=cf.center_frequency
    )
    
    for signal_power in signal_powers:
        cf.launch_power = signal_power
        cfg.save_struct_to_toml("./input/config.toml", cf)
        if use_smf:
          agg = "_SMF"
        else:
          agg=""
        output_file = f"results/ct_solution{signal_power}_gain_{cf.raman_gain}"+agg+".npy"
  
        signal_wavelengths = wdm.wavelength_grid()
        
        if not os.path.exists(output_file) or recompute:
            raise("DO NOT OVERWRITE!!")
            # assert(cf.n_modes == 1)
            pump_sol, signal_sol, ase_sol, pump_wavelengths, pump_powers = ct_solver(
                fiber,
                wdm, 
                power_per_pump   = -20,
                pump_band_a      = 1385e-9,
                pump_band_b      = 1465e-9,
                learning_rate    = 5e-2,
                epochs           = 5000,
                lock_wavelengths = 2000,
                batch_size       = 1,
                use_precomputed  = False,
                optimize         = True, 
                use_avg_oi       = False
            )
            variables_dict = {
                name: value 
                for name, value in locals().items() 
                if name in ['pump_sol', 'signal_sol', 'ase_sol', 'pump_wavelengths', 'pump_powers']
            }
            np.save(output_file, variables_dict)
            print("Results saved to file: ", output_file)
        else:
            print(f"File {output_file} already exists. Loading data...")
        
        variables_dict = np.load(output_file, allow_pickle=True).item() 
        if repropagate:
          repropagate_numpy(
            fiber              = fiber,
            signal_wavelengths = signal_wavelengths,
            pump_wavelengths   = variables_dict['pump_wavelengths'],
            pump_powers        = variables_dict['pump_powers'],
            cf                 = cf,
            output_file        = output_file
          )
        variables_dict = np.load(output_file, allow_pickle=True).item()
        
        plot_profiles(
            signal_wavelengths = wdm.wavelength_grid(),
            signal_solution    = variables_dict['signal_sol'],
            ase_solution       = variables_dict['ase_sol'],
            pump_wavelengths   = variables_dict['pump_wavelengths'],
            pump_powers        = variables_dict['pump_powers'],
            pump_solution      = variables_dict['pump_sol'],
            cf                 = cf
        )
        analyze_optimization(
            signal_wavelengths = wdm.wavelength_grid(),
            signal_solution    = variables_dict['signal_sol'],
            ase_solution       = variables_dict['ase_sol'],
            pump_wavelengths   = variables_dict['pump_wavelengths'],
            pump_solution      = variables_dict['pump_sol'],
            pump_powers        = variables_dict['pump_powers'],
            cf                 = cf
        )