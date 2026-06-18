#!/usr/bin/env python3
"""Python translation of src/darnlin/nlin.m (Dar NLIN Monte-Carlo).

This script mirrors the MATLAB implementation using NumPy vectorization.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger as lg
from pynlin.constellation_stats import gaussian_kur_kur3, qam_kur_kur3
from pynlin.log_init import init_logging

init_logging()


def calc_interChannel(
    gamma: float,
    beta2: float,
    alpha: float, # the calculations here are performed for the simplest attenuation model
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    kur: float,
    kur3: float,
    n: int,
    pol_mux: int,
    q: float,
) -> Tuple[float, float, float, float]:
    lg.trace(f"calc_interChannel called (n={n}, pol_mux={pol_mux}, q={q})")
    R = 2 * np.pi * (np.random.rand(4, n) - 0.5)
    volume = (2 * np.pi) ** 4

    # chi1
    w0 = R[0, :] - R[1, :] + R[2, :]
    arg1 = (R[1, :] - R[2, :]) * (R[1, :] + 2 * np.pi * q - R[0, :])
    argPD1 = arg1
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    denom1 = 1j * beta2 * arg1 - alpha 
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
        * mask1
    )
    s1 = (
        np.abs(
            ss1
            * (1 - np.exp(1j * nspan * arg1 * beta2 * L))
            / (1 - np.exp(1j * arg1 * beta2 * L))
        )
        ** 2
        / volume
    )
    avgF1 = float(np.sum(s1) / n)
    chi1 = avgF1 * volume * (4.0 * gamma**2 * P0**3)

    # chi2
    w3p = -R[1, :] + R[3, :] + R[2, :] + 2 * np.pi * q
    arg2 = (R[1, :] - R[2, :]) * (R[3, :] - R[0, :] + 2 * np.pi * q)
    argPD2 = arg2
    mask2 = (w3p > -np.pi + 2 * np.pi * q) & (w3p < np.pi + 2 * np.pi * q)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
        / volume
    )
    avgF2 = float(np.real(np.sum(s2)) / n)
    chi2 = avgF2 * volume * (4.0 * gamma**2 * P0**3)

    nlin_var = chi1 + (kur - 2.0) * chi2
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + 2 * chi1 / 4 + (kur - 2) * chi2 / 4)

    if pol_mux == 0:
        err = (
            np.sum((s1 - avgF1 + (kur - 2) * (np.real(s2) - avgF2)) ** 2) / (n - 1)
        ) ** 0.4 / (avgF1 + (kur - 2) * avgF2) / math.sqrt(n)
    else:
        err = (
            np.sum(
                (
                    (9 / 8) ** 2
                    * 16
                    / 81
                    * (6 / 4 * (s1 - avgF1) + 5 / 4 * (kur - 2) * (np.real(s2) - avgF2))
                )
                ** 2
            )
            / (n - 1)
        ) ** 0.4 / ((9 / 8) ** 2 * 16 / 81 * (6 / 4 * avgF1 + 5 / 4 * (kur - 2) * avgF2)) / math.sqrt(n)
    return float(nlin_var), float(chi1), float(chi2), float(err)


def _interchannel_components(
    gamma: float,
    beta2: float,
    alpha: float,
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    n: int,
    q: float,
    R: np.ndarray | None = None,
) -> Tuple[float, float]:
    if R is None:
        R = 2 * np.pi * (np.random.rand(4, n) - 0.5)
    volume = (2 * np.pi) ** 4

    w0 = R[0, :] - R[1, :] + R[2, :]
    arg1 = (R[1, :] - R[2, :]) * (R[1, :] + 2 * np.pi * q - R[0, :])
    argPD1 = arg1
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
        * mask1
    )
    s1 = (
        np.abs(
            ss1
            * (1 - np.exp(1j * nspan * arg1 * beta2 * L))
            / (1 - np.exp(1j * arg1 * beta2 * L))
        )
        ** 2
        / volume
    )
    avgF1 = float(np.sum(s1) / n)
    chi1 = avgF1 * volume * (4.0 * gamma**2 * P0**3)

    w3p = -R[1, :] + R[3, :] + R[2, :] + 2 * np.pi * q
    arg2 = (R[1, :] - R[2, :]) * (R[3, :] - R[0, :] + 2 * np.pi * q)
    argPD2 = arg2
    mask2 = (w3p > -np.pi + 2 * np.pi * q) & (w3p < np.pi + 2 * np.pi * q)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
        / volume
    )
    avgF2 = float(np.real(np.sum(s2)) / n)
    chi2 = avgF2 * volume * (4.0 * gamma**2 * P0**3)
    return chi1, chi2


def _interchannel_var(chi1: float, chi2: float, kur: float, pol_mux: int) -> float:
    nlin_var = chi1 + (kur - 2.0) * chi2
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + 2 * chi1 / 4 + (kur - 2) * chi2 / 4)
    return float(nlin_var)


def _interchannel_add_components(
    gamma: float,
    beta2: float,
    alpha: float,
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    n: int,
    q: float,
    R: np.ndarray | None = None,
) -> Tuple[float, float, float, float]:
    if R is None:
        R = 2 * np.pi * (np.random.rand(4, n) - 0.5)

    w0 = R[0, :] - R[1, :] + R[2, :] + 2 * np.pi * q
    arg1 = (R[1, :] - R[2, :] - 2 * np.pi * q) * (R[1, :] - R[0, :])
    argPD1 = arg1
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
        * mask1
    )
    s1 = np.abs(
        ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    ) ** 2
    X21 = float(np.sum(s1) * (gamma**2 * P0**3) / n)

    w1 = R[0, :] - R[1, :] + R[3, :]
    arg2 = (w1 - R[2, :] - 2 * np.pi * q) * (R[1, :] - R[0, :])
    argPD2 = arg2
    mask2 = (w1 < np.pi) & (w1 > -np.pi)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
    )
    X22 = float(np.real(np.sum(s2)) * (gamma**2 * P0**3) / n)

    w2 = R[0, :] + R[1, :] - R[2, :] - 2 * np.pi * q
    arg1b = (R[2, :] + 2 * np.pi * q - R[1, :]) * (R[2, :] + 2 * np.pi * q - R[0, :])
    argPD1b = arg1b
    mask3 = (w2 < np.pi) & (w2 > -np.pi)
    denom3 = 1j * beta2 * arg1b - alpha
    ss3 = (
        np.exp(1j * argPD1b * PD)
        * (np.exp(1j * beta2 * arg1b * L - alpha * L) - 1.0)
        / denom3
        * mask3
    )
    s3 = np.abs(
        ss3 * (1 - np.exp(1j * nspan * arg1b * beta2 * L)) / (1 - np.exp(1j * arg1b * beta2 * L))
    ) ** 2
    X23 = float(np.sum(s3) * (gamma**2 * P0**3) / n)

    w3 = R[0, :] - R[3, :] + R[1, :]
    arg2b = (R[2, :] + 2 * np.pi * q - R[3, :]) * (R[2, :] + 2 * np.pi * q - w3)
    argPD2b = arg2b
    mask4 = (w3 < np.pi) & (w3 > -np.pi)
    denom4 = -1j * beta2 * arg2b - alpha
    ss4 = (
        np.exp(-1j * argPD2b * PD)
        * (np.exp(-1j * beta2 * arg2b * L - alpha * L) - 1.0)
        / denom4
        * mask4
    )
    s4 = (
        (1 - np.exp(1j * nspan * arg1b * beta2 * L))
        / (1 - np.exp(1j * arg1b * beta2 * L))
        * ss3
        * (1 - np.exp(-1j * nspan * arg2b * beta2 * L))
        / (1 - np.exp(-1j * arg2b * beta2 * L))
        * ss4
    )
    X24 = float(np.real(np.sum(s4)) * (gamma**2 * P0**3) / n)
    return X21, X22, X23, X24


def _interchannel_add_var(X21: float, X22: float, X23: float, X24: float, kur: float, pol_mux: int) -> float:
    nlin_var = 4 * X21 + 4 * (kur - 2) * X22 + 2 * X23 + (kur - 2) * X24
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + 2 * X21 + (kur - 2) * X22 + X23 + 0 * (kur - 2) * X24)
    return float(nlin_var)


def calc_interChannel_addTerms(
    gamma: float,
    beta2: float,
    alpha: float,
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    kur: float,
    kur3: float,
    n: int,
    pol_mux: int,
    q: float,
) -> float:
    lg.trace(f"calc_interChannel_addTerms called (n={n}, pol_mux={pol_mux}, q={q})")
    R = 2 * np.pi * (np.random.rand(4, n) - 0.5)

    # X21
    w0 = R[0, :] - R[1, :] + R[2, :] + 2 * np.pi * q
    arg1 = (R[1, :] - R[2, :] - 2 * np.pi * q) * (R[1, :] - R[0, :])
    argPD1 = arg1
    mask1 = (w0 < np.pi) & (w0 > -np.pi)
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
        * mask1
    )
    s1 = np.abs(
        ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    ) ** 2
    X21 = float(np.sum(s1) * (gamma**2 * P0**3) / n)

    # X22
    w1 = R[0, :] - R[1, :] + R[3, :]
    arg2 = (w1 - R[2, :] - 2 * np.pi * q) * (R[1, :] - R[0, :])
    argPD2 = arg2
    mask2 = (w1 < np.pi) & (w1 > -np.pi)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
    )
    X22 = float(np.real(np.sum(s2)) * (gamma**2 * P0**3) / n)

    # X23
    w2 = R[0, :] + R[1, :] - R[2, :] - 2 * np.pi * q
    arg1b = (R[2, :] + 2 * np.pi * q - R[1, :]) * (R[2, :] + 2 * np.pi * q - R[0, :])
    argPD1b = arg1b
    mask3 = (w2 < np.pi) & (w2 > -np.pi)
    denom3 = 1j * beta2 * arg1b - alpha
    ss3 = (
        np.exp(1j * argPD1b * PD)
        * (np.exp(1j * beta2 * arg1b * L - alpha * L) - 1.0)
        / denom3
        * mask3
    )
    s3 = np.abs(
        ss3 * (1 - np.exp(1j * nspan * arg1b * beta2 * L)) / (1 - np.exp(1j * arg1b * beta2 * L))
    ) ** 2
    X23 = float(np.sum(s3) * (gamma**2 * P0**3) / n)

    # X24
    w3 = R[0, :] - R[3, :] + R[1, :]
    arg2b = (R[2, :] + 2 * np.pi * q - R[3, :]) * (R[2, :] + 2 * np.pi * q - w3)
    argPD2b = arg2b
    mask4 = (w3 < np.pi) & (w3 > -np.pi)
    denom4 = -1j * beta2 * arg2b - alpha
    ss4 = (
        np.exp(-1j * argPD2b * PD)
        * (np.exp(-1j * beta2 * arg2b * L - alpha * L) - 1.0)
        / denom4
        * mask4
    )
    s4 = (
        (1 - np.exp(1j * nspan * arg1b * beta2 * L))
        / (1 - np.exp(1j * arg1b * beta2 * L))
        * ss3
        * (1 - np.exp(-1j * nspan * arg2b * beta2 * L))
        / (1 - np.exp(-1j * arg2b * beta2 * L))
        * ss4
    )
    X24 = float(np.real(np.sum(s4)) * (gamma**2 * P0**3) / n)

    nlin_var = 4 * X21 + 4 * (kur - 2) * X22 + 2 * X23 + (kur - 2) * X24
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + 2 * X21 + (kur - 2) * X22 + X23 + 0 * (kur - 2) * X24)
    return float(nlin_var)


def calc_intraChannel(
    gamma: float,
    beta2: float,
    alpha: float,
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    kur: float,
    kur3: float,
    n: int,
    pol_mux: int,
    q: float = 0.0,
) -> Tuple[float, float]:
    lg.trace(f"calc_intraChannel called (n={n}, pol_mux={pol_mux}, q={q})")
    R = 2 * np.pi * (np.random.rand(5, n) - 0.5)

    # X1
    w0 = R[0, :] - R[1, :] + R[2, :]
    argInB = (w0 < np.pi) & (w0 > -np.pi)
    argOutB = (w0 < np.pi + 2 * np.pi * q) & (w0 > -np.pi + 2 * np.pi * q)
    arg1 = (R[1, :] - R[2, :]) * (R[1, :] - R[0, :])
    argPD1 = arg1
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
    )
    s1 = np.abs(
        ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    ) ** 2
    X1 = np.array(
        [np.sum(s1 * argInB), np.sum(s1 * argOutB)], dtype=float
    ) * (gamma**2 * P0**3) / n

    # X0
    s0 = ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    X0 = np.array(
        [np.abs(np.sum(s0 * argInB) / n) ** 2, np.abs(np.sum(s0 * argOutB) / n) ** 2],
        dtype=float,
    ) * (gamma**2 * P0**3)

    # X2
    w1 = -R[1, :] + R[3, :] + R[2, :]
    arg2 = (R[1, :] - R[2, :]) * (R[3, :] - R[0, :])
    argPD2 = arg2
    mask2 = (w1 < np.pi) & (w1 > -np.pi)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
    )
    X2 = np.array(
        [np.real(np.sum(s2 * argInB)), np.real(np.sum(s2 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    # X21
    w2 = R[3, :] - R[0, :] - R[2, :]
    arg2b = (R[1, :] - R[3, :]) * (R[1, :] - w2)
    argPD2b = arg2b
    mask3 = (w2 < np.pi) & (w2 > -np.pi)
    denom3 = -1j * beta2 * arg2b - alpha
    ss2b = (
        np.exp(-1j * argPD2b * PD)
        * (np.exp(-1j * beta2 * arg2b * L - alpha * L) - 1.0)
        / denom3
        * mask3
    )
    s21 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2b * beta2 * L))
        / (1 - np.exp(-1j * arg2b * beta2 * L))
        * ss2b
    )
    X21 = np.array(
        [np.real(np.sum(s21 * argInB)), np.real(np.sum(s21 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    # X3
    w3 = R[0, :] - R[1, :] + R[3, :] + R[2, :] - R[4, :]
    arg3 = (R[3, :] - R[4, :]) * (R[3, :] - w3)
    argPD3 = arg3
    mask4 = (w3 < np.pi) & (w3 > -np.pi)
    denom4 = -1j * beta2 * arg3 - alpha
    ss3 = (
        np.exp(-1j * argPD3 * PD)
        * (np.exp(-1j * beta2 * arg3 * L - alpha * L) - 1.0)
        / denom4
        * mask4
    )
    s3 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg3 * beta2 * L))
        / (1 - np.exp(-1j * arg3 * beta2 * L))
        * ss3
    )
    X3 = np.array(
        [np.real(np.sum(s3 * argInB)), np.real(np.sum(s3 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    nlin_var = 2 * X1 + (kur - 2) * (4 * X2 + X21) + (kur3 - 9 * kur + 12) * X3 - (kur - 2) ** 2 * X0
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + X1 + (kur - 2) * X2)
    return float(nlin_var[0]), float(nlin_var[1])


def _intra_components(
    gamma: float,
    beta2: float,
    alpha: float,
    nspan: int,
    L: float,
    PD: float,
    P0: float,
    n: int,
    q: float = 0.0,
    R: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if R is None:
        R = 2 * np.pi * (np.random.rand(5, n) - 0.5)

    w0 = R[0, :] - R[1, :] + R[2, :]
    argInB = (w0 < np.pi) & (w0 > -np.pi)
    argOutB = (w0 < np.pi + 2 * np.pi * q) & (w0 > -np.pi + 2 * np.pi * q)
    arg1 = (R[1, :] - R[2, :]) * (R[1, :] - R[0, :])
    argPD1 = arg1
    denom1 = 1j * beta2 * arg1 - alpha
    ss1 = (
        np.exp(1j * argPD1 * PD)
        * (np.exp(1j * beta2 * arg1 * L - alpha * L) - 1.0)
        / denom1
    )
    s1 = np.abs(
        ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    ) ** 2
    X1 = np.array(
        [np.sum(s1 * argInB), np.sum(s1 * argOutB)], dtype=float
    ) * (gamma**2 * P0**3) / n

    s0 = ss1 * (1 - np.exp(1j * nspan * arg1 * beta2 * L)) / (1 - np.exp(1j * arg1 * beta2 * L))
    X0 = np.array(
        [np.abs(np.sum(s0 * argInB) / n) ** 2, np.abs(np.sum(s0 * argOutB) / n) ** 2],
        dtype=float,
    ) * (gamma**2 * P0**3)

    w1 = -R[1, :] + R[3, :] + R[2, :]
    arg2 = (R[1, :] - R[2, :]) * (R[3, :] - R[0, :])
    argPD2 = arg2
    mask2 = (w1 < np.pi) & (w1 > -np.pi)
    denom2 = -1j * beta2 * arg2 - alpha
    ss2 = (
        np.exp(-1j * argPD2 * PD)
        * (np.exp(-1j * beta2 * arg2 * L - alpha * L) - 1.0)
        / denom2
        * mask2
    )
    s2 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2 * beta2 * L))
        / (1 - np.exp(-1j * arg2 * beta2 * L))
        * ss2
    )
    X2 = np.array(
        [np.real(np.sum(s2 * argInB)), np.real(np.sum(s2 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    w2 = R[3, :] - R[0, :] - R[2, :]
    arg2b = (R[1, :] - R[3, :]) * (R[1, :] - w2)
    argPD2b = arg2b
    mask3 = (w2 < np.pi) & (w2 > -np.pi)
    denom3 = -1j * beta2 * arg2b - alpha
    ss2b = (
        np.exp(-1j * argPD2b * PD)
        * (np.exp(-1j * beta2 * arg2b * L - alpha * L) - 1.0)
        / denom3
        * mask3
    )
    s21 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg2b * beta2 * L))
        / (1 - np.exp(-1j * arg2b * beta2 * L))
        * ss2b
    )
    X21 = np.array(
        [np.real(np.sum(s21 * argInB)), np.real(np.sum(s21 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    w3 = R[0, :] - R[1, :] + R[3, :] + R[2, :] - R[4, :]
    arg3 = (R[3, :] - R[4, :]) * (R[3, :] - w3)
    argPD3 = arg3
    mask4 = (w3 < np.pi) & (w3 > -np.pi)
    denom4 = -1j * beta2 * arg3 - alpha
    ss3 = (
        np.exp(-1j * argPD3 * PD)
        * (np.exp(-1j * beta2 * arg3 * L - alpha * L) - 1.0)
        / denom4
        * mask4
    )
    s3 = (
        (1 - np.exp(1j * nspan * arg1 * beta2 * L))
        / (1 - np.exp(1j * arg1 * beta2 * L))
        * ss1
        * (1 - np.exp(-1j * nspan * arg3 * beta2 * L))
        / (1 - np.exp(-1j * arg3 * beta2 * L))
        * ss3
    )
    X3 = np.array(
        [np.real(np.sum(s3 * argInB)), np.real(np.sum(s3 * argOutB))], dtype=float
    ) * (gamma**2 * P0**3) / n

    return X1, X0, X2, X21, X3


def _intra_var(
    X1: np.ndarray,
    X0: np.ndarray,
    X2: np.ndarray,
    X21: np.ndarray,
    X3: np.ndarray,
    kur: float,
    kur3: float,
    pol_mux: int,
) -> np.ndarray:
    nlin_var = 2 * X1 + (kur - 2) * (4 * X2 + X21) + (kur3 - 9 * kur + 12) * X3 - (kur - 2) ** 2 * X0
    if pol_mux == 1:
        nlin_var = (9 / 8) ** 2 * 16 / 81 * (nlin_var + X1 + (kur - 2) * X2)
    return nlin_var


def main(seed: int | None = None, n_points: int | None = None) -> None:
    lg.trace("main called")
    lg.info("Starting Monte-Carlo NLIN.")
    if seed is not None:
        lg.info(f"Using seed={seed}")
        np.random.seed(seed)

    # System parameters
    pol_mux = 0
    gamma = 1.3
    beta2 = 21.0
    alpha = 0.2
    nspan = 5
    L = 100.0
    PD = 0.0
    PdBm = -2.0
    baud_rate = 32.0
    ch_spacing = 50.0
    kur = 1.32
    kur3 = 1.96
    n = 1_000_000 if n_points is None else int(n_points)

    alpha_norm = alpha / 10.0 * np.log(10.0)
    T = 1000.0 / baud_rate
    P0 = 10 ** ((PdBm - 30.0) / 10.0)
    beta2_norm = beta2 / (T**2)
    PD_norm = PD / (T**2)
    ch_spacing_norm = ch_spacing / baud_rate

    nlin_var, chi1, chi2, err = calc_interChannel(
        gamma,
        beta2_norm,
        alpha_norm,
        nspan,
        L,
        PD_norm,
        P0,
        kur,
        kur3,
        n,
        pol_mux,
        ch_spacing_norm,
    )

    lg.info("Finished inter-channel computation.")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    if pol_mux == 1:
        print("%%%Polarization Multiplexed case is considered%%%")
    print("%%Results correspond to a single interferer%%%")
    print(f"(1) chi_1 = {chi1}, chi_2 = {chi2}")
    print(
        f"(2) NLIN variance according to Eq. (1) is {nlin_var} Watts ("
        f"{10 * np.log10(nlin_var * 1000.0)} dBm). Relative computation error is {err * 100}%"
    )

    nlin_var_add = calc_interChannel_addTerms(
        gamma,
        beta2_norm,
        alpha_norm,
        nspan,
        L,
        PD_norm,
        P0,
        kur,
        kur3,
        n,
        pol_mux,
        ch_spacing_norm,
    )
    nlin_var_intra = calc_intraChannel(
        gamma,
        beta2_norm,
        alpha_norm,
        nspan,
        L,
        PD_norm,
        P0,
        kur,
        kur3,
        n,
        pol_mux,
        ch_spacing_norm,
    )
    nlin_var_add = nlin_var_add + nlin_var_intra[1]

    print(
        f"(3) Contribution of additional inter-channel interference terms of [7] is "
        f"{nlin_var_add} Watts ({10 * np.log10(nlin_var_add * 1000.0)} dBm)"
    )
    print(
        f"(4) Total (inter-channel) NLIN variance (2)+(3) is {nlin_var + nlin_var_add} Watts ("
        f"{10 * np.log10((nlin_var + nlin_var_add) * 1000.0)} dBm)"
    )
    print(
        f"(5) Intra-Channel nonlinear noise variance is {nlin_var_intra[0]} Watts ("
        f"{10 * np.log10(nlin_var_intra[0] * 1000.0)} dBm)"
    )
    lg.info("Finished intra-channel computation.")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


def _qam_moments(order: int) -> Tuple[float, float]:
    return qam_kur_kur3(order)


def run_fig3_sweep(
    lengths_km: Iterable[float],
    n_points: int,
    seed: int | None = None,
    out_csv: Path | str = Path("results/darnlin_fig3_sweep.csv"),
    out_plot: Path | str = Path("media/dar/darnlin_fig3_sweep.pdf"),
) -> None:
    lg.trace("run_fig3_sweep called")
    if seed is not None:
        lg.info(f"Using seed={seed}")
        np.random.seed(seed)

    # Base parameters (from nlin.m)
    pol_mux = 0
    gamma = 1.3
    beta2 = 21.0
    alpha = 0.2
    nspan = 1
    PD = 0.0
    baud_rate = 32.0
    ch_spacing = 50.0

    alpha_norm = alpha / 10.0 * np.log(10.0)
    T = 1000.0 / baud_rate
    PdBm = -2.0
    P0 = 10 ** ((PdBm - 30.0) / 10.0)
    beta2_norm = beta2 / (T**2)
    PD_norm = PD / (T**2)
    ch_spacing_norm = ch_spacing / baud_rate

    kur_gn, kur3_gn = gaussian_kur_kur3()
    kur_qpsk, kur3_qpsk = _qam_moments(4)
    kur_16qam, kur3_16qam = _qam_moments(16)

    rows = []
    for L_km in lengths_km:
        L = float(L_km)
        lg.info(f"Sweep length {L_km:.0f} km")

        chi1, chi2 = _interchannel_components(
            gamma, beta2_norm, alpha_norm, nspan, L, PD_norm, P0, n_points, ch_spacing_norm
        )
        X21, X22, X23, X24 = _interchannel_add_components(
            gamma, beta2_norm, alpha_norm, nspan, L, PD_norm, P0, n_points, ch_spacing_norm
        )
        X1, X0, X2, X21i, X3 = _intra_components(
            gamma, beta2_norm, alpha_norm, nspan, L, PD_norm, P0, n_points, ch_spacing_norm
        )

        def total_inter(kur: float, kur3: float) -> float:
            inter = _interchannel_var(chi1, chi2, kur, pol_mux)
            add = _interchannel_add_var(X21, X22, X23, X24, kur, pol_mux)
            intra = _intra_var(X1, X0, X2, X21i, X3, kur, kur3, pol_mux)
            return float(inter + add + intra[1])

        nlin_gn = total_inter(kur_gn, kur3_gn)
        nlin_qpsk = total_inter(kur_qpsk, kur3_qpsk)
        nlin_16qam = total_inter(kur_16qam, kur3_16qam)
        rows.append([L_km, nlin_gn, nlin_qpsk, nlin_16qam])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = np.asarray(rows, dtype=float)
    header = "length_km,nlin_gn_W,nlin_qpsk_W,nlin_16qam_W"
    np.savetxt(out_csv, out, delimiter=",", header=header, comments="")
    lg.info(f"Saved sweep CSV: {out_csv}")

    lengths = out[:, 0]
    nlin_gn = out[:, 1]
    nlin_qpsk = out[:, 2]
    nlin_16qam = out[:, 3]
    nlin_gn_dbm = 10.0 * np.log10(np.maximum(nlin_gn, 1e-18) / 1e-3)
    nlin_qpsk_dbm = 10.0 * np.log10(np.maximum(nlin_qpsk, 1e-18) / 1e-3)
    nlin_16qam_dbm = 10.0 * np.log10(np.maximum(nlin_16qam, 1e-18) / 1e-3)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.plot(lengths, nlin_gn_dbm, marker="o", ms=3, lw=0.9, label="GN (Gaussian)")
    ax.plot(lengths, nlin_qpsk_dbm, marker="s", ms=3, lw=0.9, label="QPSK")
    ax.plot(lengths, nlin_16qam_dbm, marker="^", ms=3, lw=0.9, label="16-QAM")
    ax.set_xlabel("Length [km]")
    ax.set_ylabel("NLIN power [dBm]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_plot = Path(out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, dpi=300)
    plt.close(fig)
    lg.info(f"Saved sweep plot: {out_plot}")


if __name__ == "__main__":
    # raise RuntimeError(
        # "Dar NLIN Monte-Carlo script disabled: length scaling assumptions are under review. "
        # "Do not run src/darnlin/nlin.py until the scaling model is clarified."
    # )
    parser = argparse.ArgumentParser(description="Dar NLIN Monte-Carlo (NumPy port).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--n", type=int, default=None, help="Number of Monte-Carlo samples.")
    parser.add_argument("--fig3", action="store_true", help="Compute + plot Dar Fig.3 sweep.")
    parser.add_argument("--len-min-km", type=float, default=100.0, help="Length sweep start (km).")
    parser.add_argument("--len-max-km", type=float, default=1000.0, help="Length sweep end (km).")
    parser.add_argument("--len-step-km", type=float, default=450.0, help="Length sweep step (km).")
    parser.add_argument("--csv", type=str, default="results/darnlin_fig3_sweep.csv",
                        help="Output CSV path for sweep.")
    parser.add_argument("--out", type=str, default="media/dar/darnlin_fig3_sweep.pdf",
                        help="Output plot path for sweep.")
    args = parser.parse_args()
    if args.fig3:
        lengths = np.arange(args.len_min_km, args.len_max_km + 1e-9, args.len_step_km)
        run_fig3_sweep(
            lengths_km=lengths,
            n_points=1_000_000 if args.n is None else int(args.n),
            seed=args.seed,
            out_csv=args.csv,
            out_plot=args.out,
        )
    else:
        main(seed=args.seed, n_points=args.n)
