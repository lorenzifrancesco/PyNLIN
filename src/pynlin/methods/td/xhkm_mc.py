from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class XPMTaylorDispersion:
    """Channel-local target/interferer dispersion through fourth order.

    Arrays contain the target first and the interferer second. Frequencies in
    the Dar integrals are normalized by ``baud_rate``.
    """

    beta0: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    baud_rate: float
    beta3: np.ndarray | None = None
    beta4: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("beta0", "beta1", "beta2", "beta3", "beta4"):
            values = getattr(self, name)
            if values is None:
                continue
            array = np.asarray(values, dtype=float)
            if array.shape != (2,):
                raise ValueError(f"{name} must contain target and interferer values")
            object.__setattr__(self, name, array)
        if float(self.baud_rate) <= 0.0:
            raise ValueError("baud_rate must be positive")

    def evaluate(self, channel: int, omega_normalized: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega_normalized, dtype=float) * float(self.baud_rate)
        beta3 = 0.0 if self.beta3 is None else float(self.beta3[channel])
        beta4 = 0.0 if self.beta4 is None else float(self.beta4[channel])
        return (
            float(self.beta0[channel])
            + float(self.beta1[channel]) * omega
            + 0.5 * float(self.beta2[channel]) * omega**2
            + (beta3 / 6.0) * omega**3
            + (beta4 / 24.0) * omega**4
        )

    def mismatch(
        self,
        channel_a: int,
        omega_a: np.ndarray,
        channel_b: int,
        omega_b: np.ndarray,
        channel_c: int,
        omega_c: np.ndarray,
        channel_d: int,
        omega_d: np.ndarray,
    ) -> np.ndarray:
        return (
            self.evaluate(channel_a, omega_a)
            + self.evaluate(channel_b, omega_b)
            - self.evaluate(channel_c, omega_c)
            - self.evaluate(channel_d, omega_d)
        )


@dataclass(frozen=True)
class XhkmMCSums:
    """Prefactor-free Monte Carlo estimates of Dar-style Xhkm aggregates.

    ``n1`` and ``n2`` are direct aggregate MC estimates.  The collision-sector
    fields are experimental residual estimators reconstructed by subtracting
    independently sampled aggregates; unlike the exact sectors returned by
    :func:`pynlin.methods.td.xhkm_sums.compute_xhkm_sums`, a finite-sample
    residual can be negative and must not be presented as a resolved
    nonnegative sum of ``|X[h,r,m]|**2`` terms.
    """

    n1: float
    n2: float
    n_2pc: float
    n_3pc_total: float
    n_3pca: float
    n_3pcb: float
    n_4pc: float
    n1_stderr: float
    n2_stderr: float
    n_2pc_stderr: float
    n_3pc_total_stderr: float
    n_3pca_stderr: float
    n_3pcb_stderr: float
    n_4pc_stderr: float
    n_samples: int
    seed: int | None
    metadata: dict[str, object]

    @property
    def n2_over_n1(self) -> float:
        if self.n1 <= 0.0:
            return float("nan")
        return self.n2 / self.n1


def assert_flat_signal_power_profile(system: object) -> None:
    """Reject system configs that request non-flat signal power profiles."""
    raw = getattr(system, "raw_config", None)
    if not isinstance(raw, Mapping):
        return

    profiles = raw.get("profiles")
    if isinstance(profiles, Mapping):
        mode = str(profiles.get("mode", "flat")).strip().lower()
        if mode != "flat":
            raise ValueError(
                "Prefactor-free Dar MC validation supports only flat signal power profiles; "
                f"got profiles.mode={mode!r}."
            )

    pcfm = raw.get("pcfm")
    run = pcfm.get("run") if isinstance(pcfm, Mapping) else None
    if isinstance(run, Mapping) and "power_profiles_mode" in run:
        mode = str(run["power_profiles_mode"]).strip().lower()
        if mode != "flat":
            raise ValueError(
                "Prefactor-free Dar MC validation supports only flat signal power profiles; "
                f"got pcfm.run.power_profiles_mode={mode!r}."
            )


def _draw_uniform_phases_with_rng(
    rng: np.random.Generator, dim: int, n_samples: int
) -> np.ndarray:
    return 2.0 * np.pi * (rng.random((int(dim), int(n_samples))) - 0.5)


def _safe_mean_stderr(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan")
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def _span_factor(phase: np.ndarray, nspan: int, sign: int) -> np.ndarray:
    signed = 1j * int(sign) * np.asarray(phase, dtype=float)
    numerator = 1.0 - np.exp(nspan * signed)
    denominator = 1.0 - np.exp(signed)
    out = np.empty_like(numerator, dtype=complex)
    near_zero = np.abs(denominator) < 1e-14
    out[near_zero] = complex(nspan)
    out[~near_zero] = numerator[~near_zero] / denominator[~near_zero]
    return out


def _link_function(
    arg: np.ndarray,
    *,
    beta2: float,
    alpha: float,
    length: float,
    nspan: int,
    phase_delay: float,
) -> np.ndarray:
    """Dimensionless Nyquist link function used by the Dar MC formulas."""
    arg = np.asarray(arg, dtype=float)
    exponent = 1j * float(beta2) * arg - float(alpha)
    out = np.empty_like(arg, dtype=complex)
    near_zero = np.abs(exponent) < 1e-14
    out[near_zero] = complex(length)
    out[~near_zero] = (np.exp(exponent[~near_zero] * float(length)) - 1.0) / exponent[
        ~near_zero
    ]
    return (
        np.exp(1j * arg * float(phase_delay))
        * out
        * _span_factor(arg * float(beta2) * float(length), int(nspan), sign=1)
    )


def _link_from_delta_beta(
    delta_beta: np.ndarray,
    *,
    alpha: float,
    length: float,
    nspan: int,
) -> np.ndarray:
    """Stable link function for an explicitly evaluated physical mismatch."""
    delta_beta = np.asarray(delta_beta, dtype=float)
    exponent = 1j * delta_beta - float(alpha)
    out = np.empty_like(delta_beta, dtype=complex)
    near_zero = np.abs(exponent) < 1e-14
    out[near_zero] = complex(length)
    out[~near_zero] = (np.exp(exponent[~near_zero] * float(length)) - 1.0) / exponent[
        ~near_zero
    ]
    return out * _span_factor(delta_beta * float(length), int(nspan), sign=1)


def _link(
    arg: np.ndarray,
    *,
    beta2: float,
    delta_beta: np.ndarray | None,
    alpha: float,
    length: float,
    nspan: int,
    phase_delay: float,
) -> np.ndarray:
    if delta_beta is None:
        return _link_function(
            arg,
            beta2=beta2,
            alpha=alpha,
            length=length,
            nspan=nspan,
            phase_delay=phase_delay,
        )
    if phase_delay != 0.0:
        raise ValueError("local Taylor dispersion currently requires phase_delay=0")
    return _link_from_delta_beta(
        delta_beta,
        alpha=alpha,
        length=length,
        nspan=nspan,
    )


def _estimate_nyquist_2pc_and_2pc_plus_3pca_samples(
    *,
    beta2: float,
    alpha: float,
    length: float,
    channel_spacing_over_baud: float,
    nspan: int,
    phase_delay: float,
    n_samples: int,
    rng: np.random.Generator,
    dispersion: XPMTaylorDispersion | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ``N_2PC`` and ``N_2PC + N_3PCa`` from Golani/Dar S1/S2.

    These are the l=l'=0, n=n' Nyquist specializations of Eqs. (18) and
    (16) in Golani et al., JLT 2016, after multiplying by ``T**2`` so the
    normalization matches :func:`estimate_xhkm_sums_mc`.
    """
    q = float(channel_spacing_over_baud)
    omega = 2.0 * np.pi * q

    # A = S1(0,0,n,n) = N_2PC.  The outer frequency difference spans
    # [-2pi, 2pi], whereas the four pulse frequencies span [-pi, pi].
    A_vars = rng.random((5, int(n_samples)))
    x = 4.0 * np.pi * (A_vars[0, :] - 0.5)
    wp = 2.0 * np.pi * (A_vars[1:, :] - 0.5)
    a, u, b, v = wp
    mask_a = (np.abs(x + a) <= np.pi) & (np.abs(u - x) <= np.pi)
    mask_b = (np.abs(x + b) <= np.pi) & (np.abs(v - x) <= np.pi)
    arg_a = x * (x + a - u - omega)
    arg_b = x * (x + b - v - omega)
    delta_a = None
    delta_b = None
    if dispersion is not None:
        delta_a = dispersion.mismatch(1, u - x, 0, x + a, 1, u, 0, a)
        delta_b = dispersion.mismatch(1, v - x, 0, x + b, 1, v, 0, b)
    phi_a = _link(
        arg_a,
        beta2=beta2,
        delta_beta=delta_a,
        alpha=alpha,
        length=length,
        nspan=nspan,
        phase_delay=phase_delay,
    )
    phi_b = _link(
        arg_b,
        beta2=beta2,
        delta_beta=delta_b,
        alpha=alpha,
        length=length,
        nspan=nspan,
        phase_delay=phase_delay,
    )
    n_2pc_samples = np.real(2.0 * mask_a * mask_b * phi_a * np.conj(phi_b))

    # B = S2(0,0,n,n) = N_2PC + N_3PCa.  Use u and v for the two IC
    # frequencies, so x = u-v and the Jacobian is one on [-pi, pi]^4.
    B_vars = _draw_uniform_phases_with_rng(rng, 4, n_samples)
    u, v, a, b = B_vars
    x = u - v
    mask_a = np.abs(x + a) <= np.pi
    mask_b = np.abs(x + b) <= np.pi
    arg_a = x * (a - v - omega)
    arg_b = x * (b - v - omega)
    delta_a = None
    delta_b = None
    if dispersion is not None:
        delta_a = dispersion.mismatch(0, x + a, 1, v, 0, a, 1, u)
        delta_b = dispersion.mismatch(0, x + b, 1, v, 0, b, 1, u)
    phi_a = _link(
        arg_a,
        beta2=beta2,
        delta_beta=delta_a,
        alpha=alpha,
        length=length,
        nspan=nspan,
        phase_delay=phase_delay,
    )
    phi_b = _link(
        arg_b,
        beta2=beta2,
        delta_beta=delta_b,
        alpha=alpha,
        length=length,
        nspan=nspan,
        phase_delay=phase_delay,
    )
    n_2pc_plus_3pca_samples = np.real(mask_a * mask_b * phi_a * np.conj(phi_b))

    return n_2pc_samples, n_2pc_plus_3pca_samples


def _estimate_aggregate_samples(
    *,
    beta2: float,
    alpha: float,
    length: float,
    q: float,
    nspan: int,
    phase_delay: float,
    R: np.ndarray,
    rng: np.random.Generator,
    dispersion: XPMTaylorDispersion | None,
) -> np.ndarray:
    omega = 2.0 * np.pi * q
    w0 = R[0] - R[1] + R[2]
    arg1 = (R[1] - R[2]) * (R[1] + omega - R[0])
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    delta1 = None
    if dispersion is not None:
        delta1 = dispersion.mismatch(0, w0, 1, R[1], 0, R[0], 1, R[2])
    link1 = (
        _link(
            arg1,
            beta2=beta2,
            delta_beta=delta1,
            alpha=alpha,
            length=length,
            nspan=nspan,
            phase_delay=phase_delay,
        )
        * mask1
    )
    n1_samples = np.abs(link1) ** 2

    w3p_local = -R[1] + R[3] + R[2]
    arg2 = (R[1] - R[2]) * (R[3] - R[0] + omega)
    mask2 = (w3p_local > -np.pi) & (w3p_local < np.pi)
    delta2 = None
    if dispersion is not None:
        delta2 = dispersion.mismatch(0, w0, 1, R[3], 0, R[0], 1, w3p_local)
    link2 = (
        _link(
            arg2,
            beta2=beta2,
            delta_beta=delta2,
            alpha=alpha,
            length=length,
            nspan=nspan,
            phase_delay=phase_delay,
        )
        * mask2
    )
    n2_samples = np.real(link1 * np.conj(link2))

    n_2pc, n_2pc_plus_3pca = _estimate_nyquist_2pc_and_2pc_plus_3pca_samples(
        beta2=beta2,
        alpha=alpha,
        length=length,
        channel_spacing_over_baud=q,
        nspan=nspan,
        phase_delay=phase_delay,
        n_samples=R.shape[1],
        rng=rng,
        dispersion=dispersion,
    )
    return np.vstack((n1_samples, n2_samples, n_2pc, n_2pc_plus_3pca))


_SECTOR_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # N1
        [0.0, 1.0, 0.0, 0.0],  # N2
        [0.0, 0.0, 1.0, 0.0],  # 2PC
        [0.0, 0.0, -1.0, 1.0],  # 3PCa
        [0.0, 1.0, -1.0, 0.0],  # 3PCb
        [0.0, 1.0, -2.0, 1.0],  # total 3PC
        [1.0, -1.0, 1.0, -1.0],  # 4PC
    ]
)


def estimate_xhkm_sums_mc(
    *,
    beta2: float,
    alpha: float,
    length: float,
    channel_spacing_over_baud: float,
    n_samples: int,
    nspan: int = 1,
    phase_delay: float = 0.0,
    seed: int | None = None,
    random_variables: np.ndarray | None = None,
    system: object | None = None,
    dispersion: XPMTaylorDispersion | None = None,
    batch_size: int | None = None,
    min_samples: int | None = None,
    target_relative_stderr: float | None = None,
    target_stderr_over_n1: float | None = None,
    sigma_threshold: float = 3.0,
) -> XhkmMCSums:
    """Estimate prefactor-free ``N1`` and ``N2`` using Dar MC integrands.

    This is the stripped-down inter-channel Dar Monte Carlo path: no launch
    powers, no nonlinear coefficient, no modulation cumulants, and no final
    NLIN variance are included.  The returned quantities match the historical
    Dar ``chi1`` and ``chi2`` components after dividing out the common physical
    prefactor ``4*gamma**2*P0**3``.
    """
    if system is not None:
        assert_flat_signal_power_profile(system)
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    beta2 = float(beta2)
    alpha = float(alpha)
    length = float(length)
    phase_delay = float(phase_delay)
    q = float(channel_spacing_over_baud)
    nspan = int(nspan)
    if dispersion is not None and phase_delay != 0.0:
        raise ValueError("local Taylor dispersion currently requires phase_delay=0")
    if sigma_threshold <= 0.0:
        raise ValueError("sigma_threshold must be positive")
    for name, tolerance in (
        ("target_relative_stderr", target_relative_stderr),
        ("target_stderr_over_n1", target_stderr_over_n1),
    ):
        if tolerance is not None and tolerance <= 0.0:
            raise ValueError(f"{name} must be positive")

    rng = np.random.default_rng(seed)
    adaptive = target_relative_stderr is not None or target_stderr_over_n1 is not None
    if batch_size is None:
        if adaptive:
            raise ValueError("adaptive stopping requires batch_size")
        if random_variables is None:
            R = _draw_uniform_phases_with_rng(rng, 4, n_samples)
        else:
            R = np.asarray(random_variables, dtype=float)
            if R.shape != (4, n_samples):
                raise ValueError(f"random_variables must have shape (4, {n_samples})")
        aggregate_samples = _estimate_aggregate_samples(
            beta2=beta2,
            alpha=alpha,
            length=length,
            q=q,
            nspan=nspan,
            phase_delay=phase_delay,
            R=R,
            rng=rng,
            dispersion=dispersion,
        )
        sector_samples = _SECTOR_TRANSFORM @ aggregate_samples
        means = np.mean(sector_samples, axis=1)
        stderrs = np.std(sector_samples, axis=1, ddof=1) / np.sqrt(n_samples)
        actual_samples = n_samples
        n_batches = 1
        stop_reason = "fixed_budget"
        covariance_method = "paired_samples"
    else:
        if random_variables is not None:
            raise ValueError(
                "random_variables cannot be combined with batched estimation"
            )
        batch_size = int(batch_size)
        if batch_size <= 0 or n_samples % batch_size != 0:
            raise ValueError("batch_size must be positive and divide n_samples")
        minimum = batch_size * 2 if min_samples is None else int(min_samples)
        if minimum < 2 * batch_size or minimum > n_samples or minimum % batch_size != 0:
            raise ValueError(
                "min_samples must be a multiple of batch_size between 2 batches and n_samples"
            )
        batch_aggregates: list[np.ndarray] = []
        stop_reason = "maximum_budget"
        for used in range(batch_size, n_samples + 1, batch_size):
            R = _draw_uniform_phases_with_rng(rng, 4, batch_size)
            samples = _estimate_aggregate_samples(
                beta2=beta2,
                alpha=alpha,
                length=length,
                q=q,
                nspan=nspan,
                phase_delay=phase_delay,
                R=R,
                rng=rng,
                dispersion=dispersion,
            )
            batch_aggregates.append(np.mean(samples, axis=1))
            if not adaptive or used < minimum:
                continue
            transformed = (_SECTOR_TRANSFORM @ np.asarray(batch_aggregates).T).T
            current_means = np.mean(transformed, axis=0)
            current_stderrs = np.std(transformed, axis=0, ddof=1) / np.sqrt(
                transformed.shape[0]
            )
            n1_scale = abs(float(current_means[0]))
            resolved = []
            for index in (2, 3, 4, 6):
                mean = current_means[index]
                stderr = current_stderrs[index]
                positive = mean - float(sigma_threshold) * stderr > 0.0
                relative_ok = (
                    target_relative_stderr is not None
                    and mean > 0.0
                    and stderr / mean <= target_relative_stderr
                )
                absolute_ok = (
                    target_stderr_over_n1 is not None
                    and n1_scale > 0.0
                    and stderr / n1_scale <= target_stderr_over_n1
                )
                resolved.append(positive and (relative_ok or absolute_ok))
            if all(resolved):
                stop_reason = "target_precision"
                break
        batch_matrix = np.asarray(batch_aggregates, dtype=float)
        transformed = (_SECTOR_TRANSFORM @ batch_matrix.T).T
        means = np.mean(transformed, axis=0)
        stderrs = np.std(transformed, axis=0, ddof=1) / np.sqrt(transformed.shape[0])
        n_batches = transformed.shape[0]
        actual_samples = n_batches * batch_size
        covariance_method = "batch_means_linear_transform"

    n1, n2, n_2pc, n_3pca, n_3pcb, n_3pc_total, n_4pc = map(float, means)
    (
        n1_stderr,
        n2_stderr,
        n_2pc_stderr,
        n_3pca_stderr,
        n_3pcb_stderr,
        n_3pc_total_stderr,
        n_4pc_stderr,
    ) = map(float, stderrs)
    return XhkmMCSums(
        n1=n1,
        n2=n2,
        n_2pc=n_2pc,
        n_3pc_total=n_3pc_total,
        n_3pca=n_3pca,
        n_3pcb=n_3pcb,
        n_4pc=n_4pc,
        n1_stderr=n1_stderr,
        n2_stderr=n2_stderr,
        n_2pc_stderr=n_2pc_stderr,
        n_3pc_total_stderr=n_3pc_total_stderr,
        n_3pca_stderr=n_3pca_stderr,
        n_3pcb_stderr=n_3pcb_stderr,
        n_4pc_stderr=n_4pc_stderr,
        n_samples=actual_samples,
        seed=seed,
        metadata={
            "calculation": "prefactor_free_dar_mc_xhkm_sums",
            "sector_estimator": "aggregate_residuals_covariance_aware",
            "sector_estimates_are_direct_modulus_squared_sums": False,
            "sector_calculation": "nyquist_golani_jlt2016_s1_s2_l0",
            "beta2": beta2,
            "alpha": alpha,
            "length": length,
            "channel_spacing_over_baud": q,
            "nspan": nspan,
            "phase_delay": phase_delay,
            "dispersion_model": "local_taylor_beta4"
            if dispersion is not None
            else "scalar_beta2",
            "batch_size": batch_size,
            "n_batches": n_batches,
            "maximum_samples": n_samples,
            "minimum_samples": min_samples,
            "target_relative_stderr": target_relative_stderr,
            "target_stderr_over_n1": target_stderr_over_n1,
            "sigma_threshold": float(sigma_threshold),
            "covariance_method": covariance_method,
            "stop_reason": stop_reason,
        },
    )


# Branch-A sampling Jacobian: outer x drawn uniformly on (-2pi, 2pi) while the two
# inner frequencies are drawn on (-pi, pi). Empirically 2.0 to MC precision across
# every sector (see tests), matching the explicit 2.0 in the Nyquist 2PC sampler.
_DIRECT_SECTOR_CONST = 2.0


def _direct_sector_kernel(
    x: np.ndarray,
    a: np.ndarray,
    u: np.ndarray,
    *,
    beta2: float,
    dispersion: XPMTaylorDispersion | None,
    alpha: float,
    length: float,
    nspan: int,
    phase_delay: float,
    omega: float,
) -> np.ndarray:
    """One masked link kernel ``K(a, u)`` at shared outer frequency ``x``.

    Variable map to the Golani frequencies: target in ``w1=a``, target out
    ``w2=x+a``, interferer ``w3=u``, interferer other ``w4=u-x``; the outer
    ``x`` is the target in/out difference shared by the two ``|X|**2`` copies
    (the collision ``m`` sum).  The support mask keeps ``w2`` and ``w4`` inside
    the principal band ``[-pi, pi]``.
    """
    w2 = x + a
    w4 = u - x
    mask = (np.abs(w2) <= np.pi) & (np.abs(w4) <= np.pi)
    if dispersion is not None:
        delta = dispersion.mismatch(0, w2, 1, w4, 0, a, 1, u)
        kernel = _link(
            np.zeros_like(x),
            beta2=0.0,
            delta_beta=delta,
            alpha=alpha,
            length=length,
            nspan=nspan,
            phase_delay=phase_delay,
        )
    else:
        arg = -x * (u - x + omega - a)
        kernel = _link(
            arg,
            beta2=beta2,
            delta_beta=None,
            alpha=alpha,
            length=length,
            nspan=nspan,
            phase_delay=phase_delay,
        )
    return kernel * mask


def estimate_xhkm_sectors_direct_mc(
    *,
    beta2: float,
    alpha: float,
    length: float,
    channel_spacing_over_baud: float,
    n_samples: int,
    nspan: int = 1,
    phase_delay: float = 0.0,
    seed: int | None = None,
    dispersion: XPMTaylorDispersion | None = None,
    random_variables: np.ndarray | None = None,
) -> XhkmMCSums:
    """Direct common-random-number estimator for the XPM collision sectors.

    Unlike :func:`estimate_xhkm_sums_mc`, which samples four aggregates and
    reconstructs the sectors through the alternating-difference
    :data:`_SECTOR_TRANSFORM` (variance set by the *aggregate*), this estimator
    evaluates all sectors from a single draw of one outer frequency ``x`` and
    two independent inner pairs ``(a1, u1)`` and ``(a2, u2)``.  With the four
    kernels ``A=K(a1,u1)``, ``B=K(a2,u2)``, ``Cc=K(a2,u1)`` (shared interferer),
    ``Dd=K(a1,u2)`` (shared target):

    * ``N1 = E|A|**2``, ``2PC = E[A conj(B)]`` (both direct, non-negative),
    * ``3PCa = E[A conj(Cc-B)]``, ``3PCb = E[A conj(Dd-B)]``,
      ``4PC  = E[A conj(A-Cc-Dd+B)]``.

    The last three are common-random-number paired differences whose per-sample
    value scales with the *sector*, not the aggregate, so the residual sectors
    (3PCa/3PCb/4PC) are resolved 7-19x more cheaply at high walk-off where the
    transform estimator leaves them in the noise.  Derived from the Golani/Dar
    ``X[h,r,m]`` definition by Poisson summation (``m`` -> shared ``x``, ``h`` ->
    target-out frequency, ``r`` -> interferer frequency).

    ``random_variables``, if given, must have shape ``(5, n_samples)`` with row 0
    the outer ``x`` on ``(-2pi, 2pi)`` and rows 1-4 the inner ``a1, u1, a2, u2``
    on ``(-pi, pi)``.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    beta2 = float(beta2)
    alpha = float(alpha)
    length = float(length)
    phase_delay = float(phase_delay)
    q = float(channel_spacing_over_baud)
    nspan = int(nspan)
    if dispersion is not None and phase_delay != 0.0:
        raise ValueError("local Taylor dispersion currently requires phase_delay=0")
    omega = 2.0 * np.pi * q

    if random_variables is None:
        rng = np.random.default_rng(seed)
        draws = rng.random((5, n_samples))
        x = 2.0 * np.pi * (2.0 * draws[0] - 1.0)
        a1, u1, a2, u2 = (np.pi * (2.0 * draws[i] - 1.0) for i in range(1, 5))
    else:
        R = np.asarray(random_variables, dtype=float)
        if R.shape != (5, n_samples):
            raise ValueError(f"random_variables must have shape (5, {n_samples})")
        x, a1, u1, a2, u2 = R

    kwargs = dict(
        beta2=beta2, dispersion=dispersion, alpha=alpha, length=length,
        nspan=nspan, phase_delay=phase_delay, omega=omega,
    )
    A = _direct_sector_kernel(x, a1, u1, **kwargs)
    B = _direct_sector_kernel(x, a2, u2, **kwargs)
    Cc = _direct_sector_kernel(x, a2, u1, **kwargs)
    Dd = _direct_sector_kernel(x, a1, u2, **kwargs)

    per_sample = {
        "n1": _DIRECT_SECTOR_CONST * np.real(A * np.conj(A)),
        "n2": _DIRECT_SECTOR_CONST * np.real(A * np.conj(Dd)),
        "n_2pc": _DIRECT_SECTOR_CONST * np.real(A * np.conj(B)),
        "n_3pca": _DIRECT_SECTOR_CONST * np.real(A * np.conj(Cc - B)),
        "n_3pcb": _DIRECT_SECTOR_CONST * np.real(A * np.conj(Dd - B)),
        "n_4pc": _DIRECT_SECTOR_CONST * np.real(A * np.conj(A - Cc - Dd + B)),
    }
    means = {k: float(np.mean(v)) for k, v in per_sample.items()}
    stderrs = {
        k: float(np.std(v, ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else float("nan")
        for k, v in per_sample.items()
    }
    n_3pc_total = means["n_3pca"] + means["n_3pcb"]
    n_3pc_total_stderr = float(
        np.std(per_sample["n_3pca"] + per_sample["n_3pcb"], ddof=1) / np.sqrt(n_samples)
        if n_samples > 1 else float("nan")
    )

    return XhkmMCSums(
        n1=means["n1"],
        n2=means["n2"],
        n_2pc=means["n_2pc"],
        n_3pc_total=n_3pc_total,
        n_3pca=means["n_3pca"],
        n_3pcb=means["n_3pcb"],
        n_4pc=means["n_4pc"],
        n1_stderr=stderrs["n1"],
        n2_stderr=stderrs["n2"],
        n_2pc_stderr=stderrs["n_2pc"],
        n_3pc_total_stderr=n_3pc_total_stderr,
        n_3pca_stderr=stderrs["n_3pca"],
        n_3pcb_stderr=stderrs["n_3pcb"],
        n_4pc_stderr=stderrs["n_4pc"],
        n_samples=n_samples,
        seed=seed,
        metadata={
            "calculation": "prefactor_free_dar_mc_xhkm_sums",
            "sector_estimator": "direct_crn_paired_projections",
            "sector_estimates_are_direct_modulus_squared_sums": False,
            "sector_calculation": "golani_poisson_hr_projections",
            "beta2": beta2,
            "alpha": alpha,
            "length": length,
            "channel_spacing_over_baud": q,
            "nspan": nspan,
            "phase_delay": phase_delay,
            "dispersion_model": "local_taylor_beta4"
            if dispersion is not None
            else "scalar_beta2",
            "sampling_constant": _DIRECT_SECTOR_CONST,
        },
    )


__all__ = [
    "XhkmMCSums",
    "XPMTaylorDispersion",
    "assert_flat_signal_power_profile",
    "estimate_xhkm_sums_mc",
    "estimate_xhkm_sectors_direct_mc",
]
