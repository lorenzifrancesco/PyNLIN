Theory Notes
============

This page sketches the main mathematical models used across ``pynlin`` and the accompanying analysis scripts. It is intentionally light on implementation details; see the API docs for signatures and defaults.


1. Raman model
--------------

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

2. Time-domain NLI model (implementation-exact)
-----------------------------------------------

This section documents the equations currently executed by the Poggiolini workflow
(``analysis/poggiolini/workflow.py``), which is also exposed by the compatibility
entrypoint ``analysis/poggiolini_nlin.py``.

Collision-coefficient path and units
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow computes collision coefficients via ``collision_coeffs_system_uwb`` and
then converts them to SI in ``total_nlin_uwb`` using

.. math::

   y_\mathrm{norm} = \frac{1}{(L R_s)^2}, \qquad
   \mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
   = \frac{\mathcal{N}_{m_A,\nu_A,m_B,\nu_B}}{y_\mathrm{norm}}.

With per-channel launch powers :math:`P_{\nu}` (broadcast across modes when needed),
the channel-wise nonlinear coefficient is

.. math::

   \gamma_\nu = \frac{n_2\,2\pi f_\nu}{A_{\mathrm{eff},\nu} c},

and the cubic prefactor used by TD is

.. math::

   \mathcal{P}_\nu = \frac{P_\nu^3 \gamma_\nu^2}{R_s^2}.

The implementation constant is :math:`n_2 = 2.6\times 10^{-20}\,\mathrm{m^2/W}`.

Multiplicity and mode-coupling factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each pair :math:`(m_A,m_B)`, the mode-coupling weight is
:math:`\kappa^2_{m_A,m_B}` from ``get_kappa2_matrix_uwb``.
In the current workflow call, ``use_kappa=False`` and ``use_x_mode=True``, therefore
:math:`\kappa^2_{m_A,m_B}=1` for all mode pairs.

The multiplicity prefactor is

.. math::

   \Theta(m_A,m_B) =
   \begin{cases}
   \mu_0\left(2S_{m_A}+3\right)-4, & m_A=m_B,\\[4pt]
   2S_{m_B}(\mu_0-1), & m_A\neq m_B,
   \end{cases}

where ``SPATIAL_MODES = [1,2,2,1]`` and :math:`\mu_0` is the constellation
kurtosis factor. In ``total_nlin_uwb`` this :math:`\mu_0` is fixed to
64-QAM (``MU0 = qam_mu0(64)``).

TD NLIN expression actually used in workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-mode/per-channel TD NLIN before workflow scaling is

.. math::

   P^{\mathrm{TD,raw}}_{m_A,\nu_A}
   =
   \mathcal{P}_{\nu_A}
   \sum_{m_B,\nu_B}
   \mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}
   \kappa^2_{m_A,m_B}
   \Theta(m_A,m_B).

Then ``analysis/poggiolini/workflow.py`` applies a manual factor

.. math::

   P^{\mathrm{TD}}_{m_A,\nu_A}
   =
   \frac{16}{9}\,P^{\mathrm{TD,raw}}_{m_A,\nu_A}.

This is implemented by ``_apply_poggiolini_manakov_scaling`` and is currently active.

Modulation sweep path (TD decomposition)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For modulation sweeps, the workflow computes
``(constant_prefactor, sum_a, sum_b)`` in ``analysis/poggiolini/td.py``:

.. math::

   \mathrm{sum\_a}_{m_A,\nu_A}
   =
   \sum_{m_B}
   \kappa^2_{m_A,m_B}
   \left(\sum_{\nu_B}\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}\right)
   a(m_A,m_B),

.. math::

   \mathrm{sum\_b}_{m_A,\nu_A}
   =
   \sum_{m_B}
   \kappa^2_{m_A,m_B}
   \left(\sum_{\nu_B}\mathcal{N}^{(\mathrm{SI})}_{m_A,\nu_A,m_B,\nu_B}\right)
   b(m_A,m_B),

with

.. math::

   a(m_A,m_B), b(m_A,m_B)
   =
   \begin{cases}
   \left(2S_{m_A}+3,\ -4\right), & m_A=m_B,\\[4pt]
   \left(2S_{m_B},\ -2S_{m_B}\right), & m_A\neq m_B.
   \end{cases}

For a target modulation with kurtosis :math:`\mu_0`, the TD estimate is

.. math::

   P^{\mathrm{TD}}_{m_A,\nu_A}(\mu_0)
   =
   \frac{16}{9}\,
   \mathcal{P}_{\nu_A}
   \left(\mu_0\,\mathrm{sum\_a}_{m_A,\nu_A}
   + \mathrm{sum\_b}_{m_A,\nu_A}\right).

The workflow evaluates this for 16/64/256-QAM plus Gaussian
(:math:`\mu_0=2` in ``constellation_stats.gaussian_mu0``).

TD normalization used in exports/plots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The signal-power denominator for all GSNR/NLI-normalized outputs is
:math:`P_{\mathrm{sig},i}(L)` from ``_resolve_signal_power``:

1. If profile is available and valid, :math:`P_{\mathrm{sig},i}(L)` is the
   last-z profile sample.
2. Otherwise it falls back to launch powers.

Then

.. math::

   \mathrm{GSNR}^{\mathrm{TD}}_i
   = 10\log_{10}\!\left(\frac{P_{\mathrm{sig},i}(L)}
   {\max(P^{\mathrm{TD}}_{\mathrm{NLI},i}, 10^{-18})}\right).

3. PCFM/GN model (implementation-exact)
---------------------------------------

Runtime flow and power-profile handling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow supports these profile modes:
``flat``, ``cached``, ``recompute``, ``cached_no_profile_launch``,
``recompute_no_profile_launch``.

In flat mode, it writes a synthetic profile

.. math::

   P_i(z_k) = P_{i,\mathrm{launch}}, \quad \forall k,

thus :math:`p_i(z)=1`.

When using real Raman profiles, the code validates them by checking finite values
and requiring

.. math::

   \max_{i,z} P_i(z) \le 10\ \mathrm{W}.

Launch-power resolution and validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-channel launch powers are resolved with priority profile -> CSV -> TOML,
depending on run mode.

All dBm-to-W conversions use

.. math::

   P[\mathrm{W}] = 10^{(P_{\mathrm{dBm}}-30)/10}.

When a CSV is used, powers are interpolated in dBm over frequency.
When profile launch powers are used, they are checked against current settings
with tolerance

.. math::

   \max_i |P^{\mathrm{profile}}_{i,\mathrm{dBm}} - P^{\mathrm{expected}}_{i,\mathrm{dBm}}|
   \le 0.1\ \mathrm{dB}.

Signal profile loading and normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``load_signal_profiles`` returns channel profiles :math:`P_i(z)`.
For 3D stored tensors, the implementation sums one axis to obtain channel powers
(this may aggregate polarization/mode components, depending on file layout).

Optional lumped losses are applied as cumulative steps:

.. math::

   \widetilde{P}_i(z)
   =
   P_i(z)\prod_k 10^{-\ell_k/10\cdot H(z-z_k)},

then normalized:

.. math::

   p_i(z)=\frac{\widetilde{P}_i(z)}{\widetilde{P}_i(0)}.

The normalized profile is clipped to ``[0, MAX_SPP]`` with ``MAX_SPP=1e3``.

Polynomial profile representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For PCFM and GN-numeric, the profile is fit in normalized distance
:math:`u=(z-z_0)/L \in [0,1]`:

.. math::

   p_i(u)\approx \sum_{n=0}^{N} a_{i,n}u^n.

The closed-form XCI helper term is

.. math::

   S_i = \sum_{n,k}\frac{a_{i,n}a_{i,k}}{n+k+1},

implemented via coefficient convolution (``poly_sum``).

Dispersion model used in phase terms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``use_beta2_eff=True`` and beta-spline derivatives are available, the code
uses Poggiolini Eq. (28):

.. math::

   \beta_{2,\mathrm{eff}}(f_m,f_k)
   =
   \beta_2
   + \pi\beta_3(f_m+f_k-2f_c)
   + \frac{2}{3}\pi^2\beta_4
   \left[(f_m-f_c)^2 + (f_m-f_c)(f_k-f_c) + (f_k-f_c)^2\right].

Otherwise it falls back to sampled channel :math:`\beta_2`.

Numerical SCI/XCI kernels
^^^^^^^^^^^^^^^^^^^^^^^^^

The phase model is

.. math::

   \phi(f_1,f_2,z)=C_\phi\,\beta_{2,\mathrm{eff}}\,f_1f_2z,

with :math:`C_\phi=4\pi^2` by default.

SCI kernel:

.. math::

   K_{\mathrm{SCI},i}
   =
   \int_{-B/2}^{B/2}\!\!\int_{-B/2}^{B/2}
   \left|
   \int_0^L p_i(z)e^{j\phi(f_1,f_2,z)}\,dz
   \right|^2
   df_2\,df_1.

XCI kernel:

.. math::

   K_{\mathrm{XCI},ij}
   =
   \int_{-B/2}^{B/2}\!\!\int_{\Delta f_{ij}-B/2}^{\Delta f_{ij}+B/2}
   \left|
   \int_0^L p_j(z)e^{j\phi(f_1,f_2,z)}\,dz
   \right|^2
   df_2\,df_1.

All integrals are evaluated numerically with trapezoidal integration on
uniform grids. The ``direct`` GN path uses sampled :math:`p(z)` directly
instead of polynomial fits.

PCFM and GN PSD equations in code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let

.. math::

   g_i = \frac{P_{i,\mathrm{launch}}}{B},

.. math::

   \gamma_i = \frac{2\pi f_i}{c}\frac{n_2}{A_{\mathrm{eff},i}}, \qquad
   \gamma_{ij} = \frac{2\pi f_i}{c}\frac{2n_2}{A_{\mathrm{eff},i}+A_{\mathrm{eff},j}}.

Then for every CUT :math:`i`:

.. math::

   G_{\mathrm{SCI},i}
   = \frac{16}{27}g_i^3\gamma_i^2K_{\mathrm{SCI},i},

.. math::

   G_{\mathrm{XCI},ij}
   = \frac{32}{27}g_ig_j^2\gamma_{ij}^2K_{\mathrm{XCI},ij},

.. math::

   G_{\mathrm{NLI},i}
   = G_{\mathrm{SCI},i} + \sum_{j\neq i}G_{\mathrm{XCI},ij}.

Only interferers satisfying :math:`|\Delta f_{ij}| > B/2` are included.
For PCFM closed-form XCI (when ``use_numeric_xci=False``):

.. math::

   K_{\mathrm{XCI},ij}
   \approx
   \frac{L}{2\pi\max(|\beta_{2,\mathrm{eff}}|,10^{-30})}
   \ln\!\left(\frac{|\Delta f_{ij}|+B/2}{|\Delta f_{ij}|-B/2}\right)S_j.

The endpoint factor :math:`p_i(L)` is computed but currently not applied
in the final PSD expression (intentional, with FIXME note in code).

Per-polarization output correction (current implementation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PCFM, GN-numeric, and GN-direct all apply a final output correction:

.. math::

   P^{\mathrm{model}}_{\mathrm{NLI},i}
   =
   \frac{G_{\mathrm{NLI},i}B}{2}.

The same :math:`1/2` factor is also applied to returned SCI and XCI power
components.

This correction is implemented as a final-formula conversion only
(``POLARIZATION_COUNT = 2.0`` in ``pcfm_gn.py``), not as an input launch-power
renormalization.

GSNR and normalized outputs in workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each model output vector :math:`P_{\mathrm{NLI},i}`, the workflow computes

.. math::

   \mathrm{GSNR}_i
   =
   10\log_{10}\!\left(
   \frac{P_{\mathrm{sig},i}(L)}
   {\max(P_{\mathrm{NLI},i},10^{-18})}
   \right).

The denominator :math:`P_{\mathrm{sig},i}(L)` is obtained from
``_resolve_signal_power`` and is not additionally divided by 2 in this path.

The optional GN-direct plotting branch also stores

.. math::

   \eta^{\mathrm{GNdir}}_i = \frac{P^{\mathrm{GNdir}}_{\mathrm{NLI},i}}{P_{\mathrm{sig},i}(L)}

for convenience in ratio plots.

TD-vs-PCFM(XCI) diagnostic currently computed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow prints (Gaussian-modulation TD against PCFM-XCI):

.. math::

   \Delta_i = P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i} - P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},
   \qquad
   r_i = \frac{P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i}}
   {\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})},

plus dB-domain differences:

.. math::

   \Delta^{\mathrm{dB}}_i =
   10\log_{10}\!\big(\max(P^{\mathrm{TD,Gauss}}_{\mathrm{NLI},i},10^{-30})\big)
   -
   10\log_{10}\!\big(\max(P^{\mathrm{PCFM,XCI}}_{\mathrm{NLI},i},10^{-30})\big).

Supporting models and data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Four-wave mixing (FWM)
~~~~~~~~~~~~~~~~~~~~~~

Phase matching is assessed via polynomial fits of :math:`\beta_0(\omega)` per mode and the plane

.. math::

   a \omega_1 + b \omega_2 + c \omega_3 + d = 0,

constructed from mode tuples and permutation signs :math:`p`. Intersections of this plane with the frequency cube identify FWM-relevant combinations (see ``analysis/phase_matching.py`` and ``analysis/fwm_efficiency.py``).

Fiber properties
~~~~~~~~~~~~~~~~

**Overlap integrals.** Spatial overlap integrals (OI) between modes are precomputed from numerical field
solutions and fitted with low-order polynomials in wavelength. The tensor :math:`\mathrm{OI}(\lambda_1,\lambda_2)`
feeds NLIN and Raman estimators (see ``analysis/oi_fit.py``).

**Dispersion and group delay.** Group delay and dispersion are represented by mode-wise polynomials in angular
frequency:

.. math::

   \beta_1(\omega) \approx p_1 \omega^2 + p_2 \omega + p_3, \qquad
   \beta_2(\omega) = \frac{\mathrm{d}\beta_1}{\mathrm{d}\omega},

with coefficients derived from MATLAB fits and converted to SI units

**Optimization workflows.** Pump optimization solves for pump wavelengths/powers that flatten on–off gain across
modes/channels. A typical loop:

1. Load config (WDM grid, fiber, target gain).
2. Initialize pumps, run gradient-based optimizer (PyTorch) or reuse cached solutions.
3. Repropagate with a NumPy solver for verification.
4. Plot signal/pump profiles and flatness metrics.

See ``analysis/optimize.py`` and plotting helpers in ``analysis/components/plot_optimization.py``.

4. Direct Monte-Carlo TD integration (Dar NLIN)
-----------------------------------------------

The direct Monte-Carlo time-domain (TD) integration used for Dar NLIN is implemented in ``src/darnlin/nlin.py``
(NumPy port of the original MATLAB). It estimates the frequency-domain integrals by random sampling of phase
variables and returns NLIN variance for a single interferer.

Implementation mapping (Dar NLIN):

* Monte-Carlo sampling draws :math:`R \sim \mathcal{U}(-\pi,\pi)` in 4 or 5 dimensions, depending on the term.
* Inter-channel variance uses ``calc_interChannel`` which computes :math:`\chi_1` and :math:`\chi_2` and combines
  them as :math:`\sigma^2 = \chi_1 + (\mu_0-2)\chi_2` (plus a polarization-multiplexing correction when
  ``pol_mux=1``). The function returns an error estimate from sample variance.
* Additional inter-channel terms are evaluated in ``calc_interChannel_addTerms`` (:math:`X_{21}\ldots X_{24}`).
* Intra-channel terms are evaluated in ``calc_intraChannel`` (:math:`X_1, X_0, X_2, X_{21}, X_3`) and combined in
  ``_intra_var``.
* Normalization in ``main`` converts physical inputs to normalized units using :math:`T = 1/R_s` and scales
  :math:`\beta_2`, :math:`\alpha`, and :math:`\Delta f` accordingly before calling the Monte-Carlo kernels.
* The geometric sum across spans is modeled by the factor :math:`(1-e^{i n_{span} \beta_2 \Delta z})/(1-e^{i \beta_2 \Delta z})`
  inside each sampled term.
