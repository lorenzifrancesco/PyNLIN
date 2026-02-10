Theory Notes
============

This page sketches the main mathematical models used across ``pynlin`` and the accompanying analysis scripts. It is intentionally light on implementation details; see the API docs for signatures and defaults.

1.  Raman amplification
-------------------

We model Raman gain with longitudinal gain/loss profiles and overlap integrals:

.. math::

   \frac{\mathrm{d}P_s(z)}{\mathrm{d}z}
   = -\alpha_s P_s(z) + g_R \frac{\rho}{A_\mathrm{eff}} P_p(z) P_s(z),

where :math:`\alpha_s` is the signal attenuation, :math:`g_R` the Raman gain coefficient, :math:`\rho` a polarization factor, and :math:`A_\mathrm{eff}` the effective area. Counter-propagating pumps follow

.. math::

   P_p(z) = P_{p,\mathrm{in}} \exp[-\alpha_p (L - z)],

yielding closed-form solutions for :math:`P_s(z)` under undepleted pump assumptions (see ``analysis/undepleted_fB.py``).

Effective parameters governing the undepleted-pump model are:

- :math:`g_{\mathrm{eff}} = \rho g_R / A_\mathrm{eff}` (effective Raman gain coefficient)
- :math:`\alpha_s, \alpha_p` (signal and pump attenuation, in Np/m)
- :math:`L` (fiber length), :math:`P_{p,\mathrm{in}}` (pump launch power), :math:`P_s(0)` (signal launch power)

Under the undepleted-pump approximation the pump is decoupled,

.. math::

   \frac{\mathrm{d}P_p}{\mathrm{d}z} = -\alpha_p P_p,

and the signal has the closed-form solution

.. math::

   P_s(z) = P_s(0)\exp\left(-\alpha_s z + g_{\mathrm{eff}}\int_0^z P_p(z')\,\mathrm{d}z'\right).

For a co-propagating pump, :math:`P_p(z)=P_{p,\mathrm{in}}\exp(-\alpha_p z)`,

.. math::

   P_s(z)=P_s(0)\exp\left(-\alpha_s z + \frac{g_{\mathrm{eff}}P_{p,\mathrm{in}}}{\alpha_p}\left(1-e^{-\alpha_p z}\right)\right).

For a counter-propagating pump launched at :math:`z=L`,

.. math::

   P_p(z)=P_{p,\mathrm{in}}\exp[-\alpha_p(L-z)],

.. math::

   P_s(z)=P_s(0)\exp\left(-\alpha_s z + \frac{g_{\mathrm{eff}}P_{p,\mathrm{in}}e^{-\alpha_p L}}{\alpha_p}\left(e^{\alpha_p z}-1\right)\right).

2. Nonlinear interference (NLIN)
-----------------------------

The ``pynlin`` package implements a semi-analytical framework designed to estimate Non-Linear Interference (NLI) in optical fiber systems, with specialized support for Multi-Mode Fibers (MMF) and Ultra-Wideband (UWB) transmission. Unlike traditional models that rely on continuous numerical integration of the Manakov equation, ``pynlin`` utilizes a "Collision Coefficient" approach combined with a Softplus fitting law.

The Perturbative Foundation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The total NLI variance $\sigma^2_{NLI, A}$ for a channel under test (A) is modeled as the sum of nonlinear interactions with all interfering channels (B) across all fiber modes:

.. math::

   \sigma^2_{NLI, A} = \mathcal{P} \sum_{m_A, \nu_A} \sum_{m_B, \nu_B} \mathcal{N}_{AB} \cdot \kappa^2_{m_A, m_B} \cdot \Theta(m_A, m_B)

Where:
* $\mathcal{P} = \frac{P_{in}^3 \gamma^2}{R_s^2}$ is the constant prefactor determined by launch power $P_{in}$, the nonlinear coefficient $\gamma$, and the symbol rate $R_s$.
* $\kappa^2_{m_A, m_B}$ represents the squared coupling matrix between modes $m_A$ and $m_B$.
* $\Theta(m_A, m_B)$ is a multiplicity prefactor accounting for the modulation format's excess kurtosis $\mu_0$.
* $\mathcal{N}_{AB}$ is the **Channel-Pair Collision Coefficient**.



The Softplus Scaling Law
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To achieve high computational efficiency, ``pynlin`` avoids direct numerical integration of the nonlinear kernel for every channel pair. Instead, it maps the efficiency $\mathcal{N}_{AB}$ to a normalized walk-off parameter $d = |\beta_{1A} - \beta_{1B}| \cdot L \cdot R_s$ using a three-parameter **Softplus function**:

.. math::

   \text{softplus}(x, a, b, c) = a \cdot \left(1 + \left(\frac{x}{b}\right)^{1/c}\right)^{-c}

The parameters define the physical behavior of the interference:
* **$a$ (Plateau):** The Low-Order (LO) regime efficiency where walk-off is negligible and noise is signal-dependent.
* **$b$ (Turning Point):** The walk-off value marking the transition between the signal-dependent plateau and the Gaussian Noise (GN) regime.
* **$c$ (Slope):** The curvature of the transition.

UWB and Stimulated Raman Scattering (SRS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Ultra-Wideband systems, the power profile $f_B(z)$ is non-exponential due to SRS. The ``uwb`` branch introduces corrections to the Softplus parameters to account for this longitudinal power variation.

High-Walk-off (HI) Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the HI regime, the efficiency is scaled by the Raman HI integral:

.. math::

   \mathcal{R}_{HI} = \frac{1}{L} \int_{0}^{L} f_B^2(z) dz

Low-Walk-off (LO) Correction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the LO regime, the plateau value is recomputed by integrating time-integral profiles $I_{m_{lo}}(z)$ against the Raman gain profile $f_B(z)$:

.. math::

   a_{fB} = \sum_{m_{lo}=0}^{M} \left( \frac{1}{L} \int_{0}^{L} I_{m_{lo}}(z) f_B(z) dz \right)^2

Since $f_B(z)$ is channel-dependent, the implementation uses linear interpolation between pre-computed minimum and maximum Raman envelopes:

.. math::

   a_{fB} \approx \frac{R_{LO, fB} - R_{LO, min}}{R_{LO, max} - R_{LO, min}} (a_{max} - a_{min}) + a_{min}

Computational Bottlenecks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the data flow in ``nlin_estimator.py`` and ``lo_correction.py``, the performance of the package is constrained by the following factors:

1. **Raman Grid Pre-computation:** The function ``build_lookup_integral_table_with_raman`` must perform a double-nested loop over GVD space (typically $20 \times 20$ samples). For each point, it executes a numerical ``scipy.integrate.quad`` operation, which is repeated for every low-order collision term $m_{lo}$.
2. **$O(N^2)$ Complexity:** Even with Softplus fitting, the initial system-wide collision coefficient calculation involves $N_{modes} \times N_{freqs}$ squared interactions.
3. **I/O Overhead:** Multiprocessing workers must frequently load large pre-computed ``.npz`` tables for interpolation, which can bottleneck the CPU on systems with slower storage.

Poggiolini PCFM/GN model (analysis/poggiolini_nlin.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow in ``analysis/poggiolini_nlin.py`` computes per-channel NLI with the Poggiolini PCFM/GN formulation
(SCI+XCI only; MCI disabled) using Raman-derived signal power profiles. For a channel under test :math:`i` with
baud rate :math:`B`, the launch PSD is :math:`g_i = P_i/B` and the normalized signal power profile is
:math:`p_i(z) = P_i(z)/P_i(0)` (optionally including lumped losses by applying stepwise factors
:math:`10^{-\ell_k/10}` for :math:`z \ge z_k`). The NLI PSD is

.. math::

   G_{\mathrm{NLI},i} = G_{\mathrm{SCI},i} + \sum_{j \ne i} G_{\mathrm{XCI},ij}

with

.. math::

   G_{\mathrm{SCI},i} = \frac{16}{27} \, p_i(L)\, g_i^3\, \gamma_i^2\, K_{\mathrm{SCI},i},

.. math::

   G_{\mathrm{XCI},ij} = \frac{16}{27} \, p_i(L)\, g_i^2\, g_j\, \gamma_{ij}^2\, K_{\mathrm{XCI},ij}.

The nonlinear coefficients are

.. math::

   \gamma_i = \frac{2\pi f_i}{c}\frac{n_2}{A_{\mathrm{eff},i}}, \qquad
   \gamma_{ij} = \frac{2\pi f_i}{c}\frac{2n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}},

with :math:`n_2 = 2.6\times 10^{-20}\,\mathrm{m^2/W}` in the implementation. The NLI power is then
:math:`P_{\mathrm{NLI},i} = G_{\mathrm{NLI},i}\, B`.

The kernel terms are evaluated as:

.. math::

   K_{\mathrm{SCI},i} =
   \int_{-B/2}^{B/2}\!\!\int_{-B/2}^{B/2}
   \left|\left(\int_{0}^{L} p_i(z)\, e^{j 4\pi^2 \beta_{2,i} f_1 f_2 z}\, \mathrm{d}z\right)\right|^2
   \mathrm{d}f_1\, \mathrm{d}f_2,

.. math::

   K_{\mathrm{XCI},ij} =
   \int_{-B/2}^{B/2}\!\!\int_{\Delta f_{ij}-B/2}^{\Delta f_{ij}+B/2}
   \left|\left(\int_{0}^{L} p_j(z)\, e^{j 4\pi^2 \beta_{2,j} f_1 f_2 z}\, \mathrm{d}z\right)\right|^2
   \mathrm{d}f_1\, \mathrm{d}f_2,

where :math:`\Delta f_{ij} = f_j - f_i`. In ``poggiolini_nlin.py``, :math:`K_{\mathrm{SCI}}` is always computed
numerically, while :math:`K_{\mathrm{XCI}}` defaults to the closed-form PCFM approximation

.. math::

   K_{\mathrm{XCI},ij} \approx \frac{L}{2\pi |\beta_{2,j}|}
   \ln\!\left(\frac{|\Delta f_{ij}| + B/2}{|\Delta f_{ij}| - B/2}\right)
   \sum_{n,k} \frac{a_n a_k}{n+k+1},

with :math:`p_j(z/L) \approx \sum_n a_n (z/L)^n` from a polynomial fit to the normalized profile.

3. Four-wave mixing (FWM)
----------------------

Phase matching is assessed via polynomial fits of :math:`\beta_0(\omega)` per mode and the plane

.. math::

   a \omega_1 + b \omega_2 + c \omega_3 + d = 0,

constructed from mode tuples and permutation signs :math:`p`. Intersections of this plane with the frequency cube identify FWM-relevant combinations (see ``analysis/phase_matching.py`` and ``analysis/fwm_efficiency.py``).

4. Fiber properties
-----------------

1. Overlap integrals
^^^^^^^^^^
Spatial overlap integrals (OI) between modes are precomputed from numerical field solutions and fitted with low-order polynomials in wavelength. The tensor :math:`\mathrm{OI}(\lambda_1,\lambda_2)` feeds NLIN and Raman estimators (see ``analysis/oi_fit.py``).

2. Dispersion and group delay
^^^^^^^^^^

Group delay and dispersion are represented by mode-wise polynomials in angular frequency:

.. math::

   \beta_1(\omega) \approx p_1 \omega^2 + p_2 \omega + p_3, \qquad
   \beta_2(\omega) = \frac{\mathrm{d}\beta_1}{\mathrm{d}\omega},

with coefficients derived from MATLAB fits and converted to SI units

3. Optimization workflows
^^^^^^^^^^

Pump optimization solves for pump wavelengths/powers that flatten on–off gain across modes/channels. A typical loop:

1. Load config (WDM grid, fiber, target gain).
2. Initialize pumps, run gradient-based optimizer (PyTorch) or reuse cached solutions.
3. Repropagate with a NumPy solver for verification.
4. Plot signal/pump profiles and flatness metrics.

See ``analysis/optimize.py`` and plotting helpers in ``analysis/components/plot_optimization.py``.
