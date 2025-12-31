from typing import Tuple

import numpy as np
import torch
import tqdm
from torch import nn
from torch.nn import MSELoss
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from pynlin.utils import dBm2watt, watt2dBm
from pynlin.raman.pytorch.solvers import MMFRamanAmplifier


def dBm(x: torch.Tensor) -> torch.Tensor:
    """Convert a tensor from Watt to dBm."""
    return 10 * torch.log10(x) + 30

def watt(x: torch.Tensor) -> torch.Tensor:
    """Convert a tensor from dBm to Watt."""
    return torch.pow(10, (x - 30) / 10)


class GainOptimizer(nn.Module):
    """PyTorch module for finding the power and wavelength of each pump of a
    Raman amplifier to obtain a target output signal spectrum."""

    def __init__(
        self,
        raman_torch_solver: MMFRamanAmplifier,
        initial_pump_wavelengths: torch.Tensor,
        initial_pump_powers: torch.Tensor, # in dBm
        batch_size: int = 1,
    ):
        super(GainOptimizer, self).__init__()
        self.raman_solver = raman_torch_solver
        scaled_wavelengths, self.wavelength_scaling = self.scale(
            initial_pump_wavelengths.float()
        )
        self.pump_powers = nn.Parameter(initial_pump_powers.float())
        self.pump_wavelengths = nn.Parameter(scaled_wavelengths)
        self.batch_size = batch_size

    def forward(self, wavelengths: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
        """Compute the output spectrum of the Raman amplifier given pump
        parameters.
        All power units in this function are in Watt."""
        x = torch.cat((wavelengths, powers)).view(1, -1).float().repeat(self.batch_size, 1)
        return self.raman_solver(x).float()

    def scale(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Scale the input tensor to the range [0, 1]."""
        m = torch.min(x)
        d = torch.max(x) - m
        return (x - m) / d, (m, d)

    def unscale(self, x: torch.Tensor, m: float, d: float) -> torch.Tensor:
        """Unscale the input tensor to the original range."""
        return (x * d) + m

    def optimize(
        self,
        target_spectrum: np.ndarray = None,
        epochs: int = 100,
        learning_rate: float = 2e-2,
        lock_wavelengths: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the optimization algorithm."""

        if target_spectrum is None:
            _target_spectrum = dBm(
                self.raman_solver.signal_power
                * torch.ones_like(self.raman_solver.signal_wavelengths).view(1, -1)
            ).float()
        else:
            print("Overriding the loading of the target spectrum")
            _target_spectrum = torch.from_numpy(target_spectrum).float()
            # _target_spectrum = torch.from_numpy(target_spectrum).view(1, -1).float()

        torch_optimizer = Adam(
            self.parameters(),
            lr=learning_rate,          # Learning rate
            betas=(0.9, 0.999), # (beta1, beta2)
            eps=1e-08,         # Epsilon
            weight_decay=0,    # Weight decay
            amsgrad=False      # AMSGrad
        )
        # SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        loss_function = MSELoss()

        best_loss = torch.inf
        best_wavelengths = torch.clone(self.pump_wavelengths)
        best_powers = torch.clone(self.pump_powers)
        best_flatness = torch.inf
        
        # do not optimize wavelengths for the first `lock_wavelengths` epochs
        self.pump_wavelengths.requires_grad = False
        reg_lambda = 0.0
        # pbar = tqdm.trange(epochs)
        # pbar = tqdm.trange(epochs)
        for epoch in range(epochs):
            if best_loss > 0.001 or flatness > 0.001:
                if epoch > lock_wavelengths:
                    self.pump_wavelengths.requires_grad = True
                pump_wavelengths = self.unscale(
                    self.pump_wavelengths, *self.wavelength_scaling
                )
                signal_spectrum = dBm(self.forward( # the physics works in Watt
                    pump_wavelengths, watt(self.pump_powers))) 
                loss = loss_function(signal_spectrum, _target_spectrum) # the loss function is in dBm
                loss.backward()
                torch_optimizer.step()
                torch_optimizer.zero_grad()

                with torch.no_grad():
                    flatness = (
                        torch.max(signal_spectrum) - torch.min(signal_spectrum)
                    ).item()
                print(
                    f"\n({epoch:4d}/{epochs:4d}) RMSE: {np.sqrt(loss.item()):10.4f} | Best: {np.sqrt(best_loss):10.4f} | Flat: {flatness:6.2f} dB"
                )
                print(
                  f"            Signal:  {torch.mean(signal_spectrum):.2f}  | Target:  {torch.mean(_target_spectrum):.2f}  | Pump:  {torch.mean(self.pump_powers):.2f}"
                )
                print(f"            Wavel [um] : {pump_wavelengths.detach().numpy()*1e6}")
                print(f"            Pow. [dBm] : {self.pump_powers.detach().numpy()}")
                np.save("results/gain_walker.npy", signal_spectrum.detach().numpy()[0])
                # np.save("results/pump_wavelengths.npy", pump_wavelengths.detach().numpy())

                if loss.item() < best_loss:
                    pump_wavelengths = self.unscale(
                        self.pump_wavelengths, * self.wavelength_scaling
                    )
                    best_wavelengths = torch.clone(pump_wavelengths)
                    best_powers = torch.clone(self.pump_powers)
                    best_loss = loss.item()
                    best_flatness = flatness
                    
        print(f"\nPow. : {best_flatness}")
        print(f"\nFlatness: {flatness:.2f} dB")
        return (
            best_wavelengths.detach().numpy().squeeze(),
            best_powers.detach().numpy().squeeze(),
        )
