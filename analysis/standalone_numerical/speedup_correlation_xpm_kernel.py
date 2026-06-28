import numpy as np
from time import perf_counter


def next_pow_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def shift_zero(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Return y[n] = x[n - shift] with zero padding.

    Positive shift moves the array to the right.
    """
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x
    return y


def direct_lag_integrals(C, D, lags, dt):
    """
    Q[lag] = dt * sum_n C[n] D[n - lag]
    with zero padding outside the grid.
    """
    Q = np.empty(len(lags), dtype=complex)

    for i, lag in enumerate(lags):
        D_shifted = shift_zero(D, lag)  # D[n - lag]
        Q[i] = dt * np.sum(C * D_shifted)

    return Q


def fft_lag_integrals(C, D, lags, dt):
    """
    Compute the same Q[lag] = dt * sum_n C[n] D[n - lag]
    using FFT convolution.

    Since
        Q[lag] = sum_n C[n] D[n - lag],
    this is the linear convolution of C with reversed D:

        conv = C * D[::-1]

    and
        Q[lag] = conv[N - 1 + lag].
    """
    N = len(C)
    nfft = next_pow_two(2 * N - 1)

    conv = np.fft.ifft(np.fft.fft(C, nfft) * np.fft.fft(D[::-1], nfft))
    conv = conv[: 2 * N - 1]

    indices = (N - 1) + lags
    return dt * conv[indices]


def main():
    # Grid
    N = 2**15
    dt = 1.0
    t = (np.arange(N) - N // 2) * dt

    # Symbol period in samples
    sps = 16
    T_samples = sps

    # Toy dispersed/chirped pulse
    width = 120.0
    chirp = 0.003
    g = np.exp(-0.5 * (t / width) ** 2) * np.exp(1j * chirp * t**2)

    # Choose one (h, r) family.
    # In the NLIN notation:
    #   C_h(t) = g*(t) g(t - hT)
    #   D_r(t) = g*(t - rT) g(t)
    h = 2
    r = -3

    g_h = shift_zero(g, h * T_samples)      # g(t - hT)
    g_r = shift_zero(g, r * T_samples)      # g(t - rT)

    C = np.conj(g) * g_h
    D = np.conj(g_r) * g

    # These are the m-values.
    # Lag = mT, here restricted to integer sample shifts.
    m_values = np.arange(-800, 801)
    lags = m_values * T_samples

    # Direct calculation
    t0 = perf_counter()
    Q_direct = direct_lag_integrals(C, D, lags, dt)
    t1 = perf_counter()

    # FFT-correlation calculation
    t2 = perf_counter()
    Q_fft = fft_lag_integrals(C, D, lags, dt)
    t3 = perf_counter()

    # Error
    abs_err = np.max(np.abs(Q_direct - Q_fft))
    rel_err = abs_err / np.max(np.abs(Q_direct))

    print("Number of time samples:", N)
    print("Number of m-values:", len(m_values))
    print()
    print(f"Direct time-domain lag + integrate: {t1 - t0:.4f} s")
    print(f"FFT correlation method:            {t3 - t2:.4f} s")
    print(f"Speedup:                           {(t1 - t0) / (t3 - t2):.1f}x")
    print()
    print(f"Maximum absolute error: {abs_err:.3e}")
    print(f"Maximum relative error: {rel_err:.3e}")

    # Optional: show a few values
    for idx in [0, len(m_values)//2, -1]:
        m = m_values[idx]
        print()
        print(f"m = {m}")
        print(f"direct = {Q_direct[idx]}")
        print(f"fft    = {Q_fft[idx]}")


if __name__ == "__main__":
    main()