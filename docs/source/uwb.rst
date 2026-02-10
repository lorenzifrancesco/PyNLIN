UWB Transmission Notes
======================

This note organizes the core experimental data from the study by Puttnam
et al. (2025) regarding ultra-wideband transmission across the full
low-loss window of standard single-mode fiber (SMF).

1. Global Transmission Benchmarks
---------------------------------

The study achieved record-breaking aggregate data rates by utilizing up
to 1505 channels with 25 GHz spacing covering the O, E, S, C, L, and
U-bands.

===================== ====================== ======================
Parameter             50 km Transmission     100 km Transmission
===================== ====================== ======================
**Total Bandwidth**   37.6 THz               36.6 THz
**GMI Data-Rate**     402.2 Tb/s             339.1 Tb/s
**Decoded Data-Rate** 378.9 Tb/s             322.8 Tb/s
**Spectral Range**    1281.2 nm to 1649.9 nm 1281.2 nm to 1649.9 nm
===================== ====================== ======================


2. Per-Band Performance Comparison (100 km)
-------------------------------------------

The following data reflects the 100 km span results, showing the impact
of fiber loss and amplifier limitations on specific spectral regions.

.. list-table::
   :header-rows: 1
   :widths: 18 10 16 16 16

   * - Band
     - Channels
     - Bandwidth (THz)
     - GMI Data-Rate (Tb/s)
     - % Decrease (vs 50km)
   * - O-Band
     - 275
     - 6.9
     - 42.9
     - 21.9%
   * - E-Band
     - 310
     - 7.8
     - 76.4
     - 13.4%
   * - S-Band
     - 296
     - 7.4
     - 74.6
     - 27.5%
   * - C-Band
     - 195
     - 4.9
     - 53.9
     - 11.7%
   * - L-Band
     - 245
     - 6.1
     - 67.1
     - 13.1%
   * - U-Band
     - 119
     - 3.0
     - 24.1
     - 33.2%


3. Amplification Hardware Strategy
----------------------------------

The system combined six variants of doped-fiber amplifiers (DFAs) with
discrete Raman U-band amplifiers and distributed Raman amplification.

.. list-table::
   :header-rows: 1
   :widths: 16 18 12 24

   * - Band
     - Amplifier Type
     - Noise Figure (NF)
     - Performance Notes
   * - **O-Band**
     - P-DFA / O-BDFA
     - 4.5 – 7.0 dB
     - 18 or 23 dBm variants; strong wavelength dependent gain.
   * - **E-Band**
     - Bismuth (B-DFA)
     - 5.0 – 6.0 dB
     - Higher output power reaching up to 25 dBm.
   * - **S-Band**
     - Thulium (T-DFA)
     - < 7.0 dB
     - Based on thulium-doped fluoride fibers.
   * - **C-Band**
     - Erbium (EDFA)
     - ~ 5.0 dB
     - Typical silica-host EDFAs; 13 to 22 dBm output.
   * - **L-Band**
     - 3-Stage EDFA
     - ~ 6.0 dB
     - Optimized for extended long-wavelength performance.
   * - **U-Band**
     - Discrete Raman
     - 4.2 – 13.7 dB*
     - Gain variation of ~8 dB across 25 nm spectrum.

*\*U-band NF depends on Raman gain fiber: 4.2 dB for HNLF vs. 8.1–13.7 dB for HNLDSF.*


4. Transmission Environment & Launch Powers
-------------------------------------------

The experiment utilized suppressed OH-peak SMF. High fiber loss in O and
U bands presented the most significant challenges for longer spans.

**Fiber Span Loss (at 100 km)**

- **O-Band (1300 nm):** 0.35 dB/km
- **E-Band (1440 nm):** 0.24 dB/km
- **C-Band (1560 nm):** 0.19 dB/km
- **U-Band (1625 nm):** 0.21 dB/km

**Per-Band Launch Powers (100 km test)**

- **O-Band:** 20.0 dBm
- **E-Band:** 21.5 dBm
- **S-Band:** 20.0 dBm
- **C-Band:** 17.0 dBm
- **L-Band:** 17.5 dBm
- **U-Band:** 18.5 dBm


5. Modulation Formats & Signal Specs
------------------------------------

Signal quality was estimated from GMI and LDPC decoding.

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Category
     - Description
   * - S, C, L-Bands
     - DP-256QAM (50 km) / DP-64QAM (100 km)
   * - E, U-Bands
     - DP-64QAM
   * - O-Band
     - DP-16QAM (limited by high modulator loss and bias issues)
   * - Symbol Rate
     - 24.5 GBaud DP-QAM root-raised cosine shaped signals
