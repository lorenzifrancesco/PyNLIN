from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from pynlin.constellations import QAM


@dataclass(frozen=True)
class ConstellationStats:
    label: str
    family: str
    order: int | None
    e2: float
    e4: float
    e6: float
    mu0: float
    mu0_minus_1: float
    mu0_minus_2: float
    abs2_variance: float
    kur: float
    kur3: float
    intra_cubic_term: float


def _stats_from_symbols(
    symbols: np.ndarray,
    *,
    label: str,
    family: str,
    order: int | None,
) -> ConstellationStats:
    arr = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label}: empty symbol set.")
    abs2 = np.abs(arr) ** 2
    e2 = float(np.mean(abs2))
    if e2 <= 0.0 or (not np.isfinite(e2)):
        raise ValueError(f"{label}: invalid E[|b|^2]={e2}.")
    e4 = float(np.mean(abs2**2))
    e6 = float(np.mean(abs2**3))
    mu0 = e4 / (e2**2)
    kur = mu0
    kur3 = e6 / (e2**3)
    return ConstellationStats(
        label=label,
        family=family,
        order=order,
        e2=e2,
        e4=e4,
        e6=e6,
        mu0=mu0,
        mu0_minus_1=mu0 - 1.0,
        mu0_minus_2=mu0 - 2.0,
        abs2_variance=e4 - e2**2,
        kur=kur,
        kur3=kur3,
        intra_cubic_term=kur3 - 9.0 * kur + 12.0,
    )


@lru_cache(maxsize=None)
def qam_stats(order: int) -> ConstellationStats:
    order = int(order)
    if order < 4:
        raise ValueError(f"QAM order must be >= 4, got {order}.")
    symbols = QAM(order).symbols()
    return _stats_from_symbols(
        symbols,
        label=f"QAM-{order}",
        family="QAM",
        order=order,
    )


@lru_cache(maxsize=None)
def psk_stats(order: int) -> ConstellationStats:
    order = int(order)
    if order < 2:
        raise ValueError(f"PSK order must be >= 2, got {order}.")
    # Constant-modulus uniform M-PSK has |b|=1 exactly for every symbol.
    return ConstellationStats(
        label=f"PSK-{order}",
        family="PSK",
        order=order,
        e2=1.0,
        e4=1.0,
        e6=1.0,
        mu0=1.0,
        mu0_minus_1=0.0,
        mu0_minus_2=-1.0,
        abs2_variance=0.0,
        kur=1.0,
        kur3=1.0,
        intra_cubic_term=4.0,
    )


@lru_cache(maxsize=1)
def gaussian_stats() -> ConstellationStats:
    # Circular complex Gaussian, unit average power.
    e2 = 1.0
    e4 = 2.0
    e6 = 6.0
    mu0 = e4 / (e2**2)
    kur = mu0
    kur3 = e6 / (e2**3)
    return ConstellationStats(
        label="Gaussian",
        family="Gaussian",
        order=None,
        e2=e2,
        e4=e4,
        e6=e6,
        mu0=mu0,
        mu0_minus_1=mu0 - 1.0,
        mu0_minus_2=mu0 - 2.0,
        abs2_variance=e4 - e2**2,
        kur=kur,
        kur3=kur3,
        intra_cubic_term=kur3 - 9.0 * kur + 12.0,
    )


def qam_mu0(order: int) -> float:
    return float(qam_stats(order).mu0)


def qam_kur_kur3(order: int) -> tuple[float, float]:
    stats = qam_stats(order)
    return float(stats.kur), float(stats.kur3)


def gaussian_mu0() -> float:
    return float(gaussian_stats().mu0)


def gaussian_kur_kur3() -> tuple[float, float]:
    stats = gaussian_stats()
    return float(stats.kur), float(stats.kur3)


def build_uniform_constellation_stats(
    qam_orders: Sequence[int] = (4, 16, 32, 64, 128, 256, 512, 1024),
    psk_orders: Sequence[int] = (2, 4, 8, 16, 32, 64),
    include_gaussian: bool = True,
) -> list[ConstellationStats]:
    rows: list[ConstellationStats] = []
    if include_gaussian:
        rows.append(gaussian_stats())
    rows.extend(psk_stats(m) for m in psk_orders)
    rows.extend(qam_stats(m) for m in qam_orders)
    return rows


def format_stats_table(rows: Iterable[ConstellationStats]) -> str:
    rows_list = list(rows)
    headers = [
        "label",
        "family",
        "M",
        "E2",
        "E4",
        "E6",
        "mu0",
        "mu0-1",
        "mu0-2",
        "Var(|b|^2)",
        "kur3",
        "kur3-9kur+12",
    ]

    def _fmt(val: float) -> str:
        if abs(val) < 1e-14:
            return "0"
        return f"{val:.12g}"

    line_rows: list[list[str]] = []
    for row in rows_list:
        line_rows.append(
            [
                row.label,
                row.family,
                "-" if row.order is None else str(row.order),
                _fmt(row.e2),
                _fmt(row.e4),
                _fmt(row.e6),
                _fmt(row.mu0),
                _fmt(row.mu0_minus_1),
                _fmt(row.mu0_minus_2),
                _fmt(row.abs2_variance),
                _fmt(row.kur3),
                _fmt(row.intra_cubic_term),
            ]
        )

    widths = [len(h) for h in headers]
    for vals in line_rows:
        widths = [max(w, len(v)) for w, v in zip(widths, vals)]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)
    data_lines = [" | ".join(v.ljust(w) for v, w in zip(vals, widths)) for vals in line_rows]
    return "\n".join([header_line, sep_line, *data_lines])


def print_uniform_constellation_mu_table(
    qam_orders: Sequence[int] = (4, 16, 32, 64, 128, 256, 512, 1024),
    psk_orders: Sequence[int] = (2, 4, 8, 16, 32, 64),
    include_gaussian: bool = True,
) -> None:
    rows = build_uniform_constellation_stats(
        qam_orders=qam_orders,
        psk_orders=psk_orders,
        include_gaussian=include_gaussian,
    )
    print("\nUniform constellation statistical factors (analytical):")
    print(format_stats_table(rows))
    print("")


def _parse_orders(values: Sequence[str]) -> tuple[int, ...]:
    if not values:
        return tuple()
    return tuple(int(v) for v in values)


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print analytical constellation statistics and mu-term decomposition "
            "for uniform constellations."
        )
    )
    parser.add_argument(
        "--qam",
        nargs="*",
        default=[4, 16, 32, 64, 128, 256, 512, 1024],
        help="QAM orders to include (space-separated).",
    )
    parser.add_argument(
        "--psk",
        nargs="*",
        default=[2, 4, 8, 16, 32, 64],
        help="PSK orders to include (space-separated).",
    )
    parser.add_argument(
        "--no-gaussian",
        action="store_true",
        help="Exclude Gaussian reference row.",
    )
    args = parser.parse_args()
    qam_orders = _parse_orders(args.qam)
    psk_orders = _parse_orders(args.psk)
    print_uniform_constellation_mu_table(
        qam_orders=qam_orders,
        psk_orders=psk_orders,
        include_gaussian=not args.no_gaussian,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
