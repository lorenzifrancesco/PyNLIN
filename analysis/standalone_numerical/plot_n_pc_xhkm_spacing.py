from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pynlin.methods.td.time_integrals import X0mm_space_integral, m_th_time_integral_general
from pynlin.methods.td.xpm_kernel import classify_collision, compute_x0mm_fft, compute_xpm_kernel_fft
from pynlin.pulses import RootRaisedCosinePulse


OUT_DIR = Path("media/n-PC")
OUT_PDF = OUT_DIR / "xhkm_spacing_sweep.pdf"
OUT_PNG = OUT_DIR / "xhkm_spacing_sweep.png"
OUT_CSV = OUT_DIR / "xhkm_spacing_sweep.csv"

FIBER_LENGTH_M = 100e3
BAUD_RATE_HZ = 32e9
BETA2_S2_PER_M = 21e-27
GAMMA_W_INV_M = 1.3e-3
SPACING_GHZ = np.linspace(32.0, 200.0, 80)
Z = np.linspace(0.0, FIBER_LENGTH_M, 801)

PULSE = RootRaisedCosinePulse(
    baud_rate=BAUD_RATE_HZ,
    num_symbols=320,
    samples_per_symbol=16,
    rolloff=0.2,
)

# compute_xpm_kernel_fft stores X[h, r, m] = X_{h,m+r,m}.
REQUESTS = {
    "X_0_10_10_fft": (0, 0, 1),
    "X_0_10_11_fft": (0, -1, 1),
    "X_1_10_10_fft": (1, 0, 1),
    "X_1_11_10_fft": (1, 1, 1),
}


def _extract(result, h: int, r: int, m: int) -> complex:
    h_idx = int(np.where(result.h_values == h)[0][0])
    r_idx = int(np.where(result.r_values == r)[0][0])
    m_idx = int(np.where(result.m_values == m)[0][0])
    return complex(result.X[h_idx, r_idx, m_idx])


def _legacy_x0mm(m: int, dgd: float) -> complex:
    time_integral = m_th_time_integral_general(
        m,
        Z,
        PULSE,
        dgd,
        BETA2_S2_PER_M,
        BETA2_S2_PER_M,
    )
    return complex(X0mm_space_integral(Z, time_integral))


def _request_label(h: int, r: int, m: int) -> str:
    return rf"$|X_{{{h},{m + r},{m}}}|$"


def _request_class(h: int, r: int, m: int) -> str:
    return classify_collision(h, r, m)


def _request_asymptote_exponent(h: int, r: int, m: int) -> float:
    return {"2pc": -1.0, "3pc": -2.0, "4pc": -3.0}[_request_class(h, r, m)]


def _unique_request_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_values = np.array(sorted({h for h, _, _ in REQUESTS.values()}), dtype=int)
    r_values = np.array(sorted({r for _, r, _ in REQUESTS.values()}), dtype=int)
    m_values = np.array(sorted({m for _, _, m in REQUESTS.values()}), dtype=int)
    return h_values, r_values, m_values


def _two_pc_request_name() -> str:
    return next(name for name, (h, r, m) in REQUESTS.items() if _request_class(h, r, m) == "2pc")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spacing_hz = SPACING_GHZ * 1e9
    values = {name: np.empty(SPACING_GHZ.size, dtype=complex) for name in REQUESTS}
    two_pc_name = _two_pc_request_name()
    two_pc_m = REQUESTS[two_pc_name][2]
    legacy_name = f"{two_pc_name}_legacy_direct"
    values[legacy_name] = np.empty(SPACING_GHZ.size, dtype=complex)
    fft_2pc_times = np.empty(SPACING_GHZ.size, dtype=float)
    legacy_2pc_times = np.empty(SPACING_GHZ.size, dtype=float)

    h_values, r_values, m_values = _unique_request_axes()

    for idx, spacing in enumerate(spacing_hz):
        # Eq. (4) uses g(t-kT-beta*Omega*z). With our delay convention this
        # collision corresponds to a negative walkoff for positive spacing.
        dgd = -BETA2_S2_PER_M * 2.0 * np.pi * float(spacing)
        t0 = perf_counter()
        x0mm_fft = compute_x0mm_fft(
            PULSE,
            Z,
            np.array([two_pc_m], dtype=int),
            dgd=dgd,
            gvda=BETA2_S2_PER_M,
            gvdb=BETA2_S2_PER_M,
        )[0]
        fft_2pc_times[idx] = perf_counter() - t0

        result = compute_xpm_kernel_fft(
            PULSE,
            Z,
            h_values,
            r_values,
            m_values,
            dgd=dgd,
            gvda=BETA2_S2_PER_M,
            gvdb=BETA2_S2_PER_M,
        )
        for name, (h, r, m) in REQUESTS.items():
            values[name][idx] = _extract(result, h, r, m)
        values[two_pc_name][idx] = x0mm_fft

        t0 = perf_counter()
        values[legacy_name][idx] = _legacy_x0mm(two_pc_m, dgd)
        legacy_2pc_times[idx] = perf_counter() - t0

    _save_csv(values)
    average_speedup = float(np.mean(legacy_2pc_times / np.maximum(fft_2pc_times, 1e-12)))
    _plot(values, average_speedup)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


def _save_csv(values: dict[str, np.ndarray]) -> None:
    columns = [SPACING_GHZ]
    header = ["spacing_ghz"]
    for name, arr in values.items():
        columns.extend([np.abs(arr), arr.real, arr.imag])
        header.extend([f"abs_{name}", f"real_{name}", f"imag_{name}"])
    np.savetxt(
        OUT_CSV,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
    )


def _tail_matched_asymptote(x: np.ndarray, y: np.ndarray, exponent: float, n_tail: int = 8) -> np.ndarray:
    positive = y > 0.0
    x_tail = x[positive][-n_tail:]
    y_tail = y[positive][-n_tail:]
    if x_tail.size == 0:
        return np.full_like(x, np.nan, dtype=float)
    scale = np.exp(np.mean(np.log(y_tail) - exponent * np.log(x_tail)))
    return scale * x**exponent


def _plot(values: dict[str, np.ndarray], average_speedup: float) -> None:
    fig, ax = plt.subplots()
    # Dar et al. Fig. 3 plots gamma |X| in 1/W/ps.
    y_scale = GAMMA_W_INV_M * 1e-12
    for name, (h, r, m) in REQUESTS.items():
        y = y_scale * np.abs(values[name])
        (line,) = ax.loglog(
            SPACING_GHZ,
            y,
            label=rf"{_request_label(h, r, m)} FFT {_request_class(h, r, m).upper()}",
        )
        ax.loglog(
            SPACING_GHZ,
            _tail_matched_asymptote(SPACING_GHZ, y, _request_asymptote_exponent(h, r, m)),
            ":",
            color=line.get_color(),
            alpha=0.8,
        )

    legacy_name = next(name for name in values if name.endswith("_legacy_direct"))
    ax.loglog(
        SPACING_GHZ,
        y_scale * np.abs(values[legacy_name]),
        "--",
        label=rf"{_request_label(*REQUESTS[_two_pc_request_name()])} direct legacy",
    )
    ax.set_xlabel("Channel spacing [GHz]")
    ax.set_ylabel(r"$\gamma |X_{h,k,m}|$ [1/W/ps]")
    ax.text(
        0.03,
        0.04,
        f"single-$m$ 2PC FFT/direct: {average_speedup:.1f}x",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.02, 0.12), framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
