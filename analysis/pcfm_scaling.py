import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from loguru import logger as lg
except ModuleNotFoundError:  # pragma: no cover - fallback for lightweight CLI usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    lg = logging.getLogger(__name__)


def _parse_float_list(values: str) -> list[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def _prompt_target() -> str:
    prompt = (
        "What do you want to calculate?\n"
        "  1) channel spacing\n"
        "  2) baud rate\n"
        "  3) length\n"
        "  4) all\n"
        "Selection: "
    )
    mapping = {
        "1": "channel-spacing",
        "channel": "channel-spacing",
        "channel spacing": "channel-spacing",
        "channel-spacing": "channel-spacing",
        "2": "baud-rate",
        "baud": "baud-rate",
        "baud rate": "baud-rate",
        "baud-rate": "baud-rate",
        "3": "length",
        "length": "length",
        "4": "all",
        "all": "all",
    }
    while True:
        choice = input(prompt).strip().lower()
        target = mapping.get(choice)
        if target is not None:
            return target
        print("Invalid selection. Choose 1, 2, 3, 4, or the corresponding name.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive launcher for PCFM scaling sweeps."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="input/pcfm_struct.toml",
        help="Path to system TOML.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("channel-spacing", "baud-rate", "length", "all"),
        default=None,
        help="Scaling sweep to run. If omitted, ask interactively at startup.",
    )
    parser.add_argument(
        "--channel-spacing-ghz",
        type=str,
        default="100,500,700",
        help="Comma-separated channel spacing values in GHz.",
    )
    parser.add_argument(
        "--baud-rates-gbaud",
        type=str,
        default="10,50,100",
        help="Comma-separated baud rates in GBaud.",
    )
    parser.add_argument(
        "--lengths-km",
        type=str,
        default="10,25,50,100,200,400",
        help="Comma-separated fiber lengths in km.",
    )
    parser.add_argument(
        "--launch-dbm",
        type=float,
        default=None,
        help="Optional uniform launch power override in dBm.",
    )
    parser.add_argument(
        "--pcfm-numeric-xci",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].pcfm_numeric_xci.",
    )
    parser.add_argument(
        "--recompute-td",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].td_mode.",
    )
    parser.add_argument(
        "--recompute-pcfm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].pcfm_mode.",
    )
    parser.add_argument(
        "--exclude-self-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override [pcfm.run].td_exclude_self_channel.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/debug/pcfm_scaling",
        help="Root output directory for generated profiles, caches, and reports.",
    )
    args = parser.parse_args()

    target = args.target or _prompt_target()
    cfg_path = Path(args.config)
    out_root = Path(args.out_dir)
    launch_dbm = args.launch_dbm
    pcfm_numeric_xci = args.pcfm_numeric_xci
    recompute_td = args.recompute_td
    recompute_pcfm = args.recompute_pcfm
    exclude_self_channel = args.exclude_self_channel

    results: list[tuple[str, Path, Path]] = []

    from analysis.pcfm_baud_scaling import run_baud_sweep
    from analysis.pcfm_channel_spacing_scaling import (
        run_spacing_sweep as run_channel_spacing_sweep,
    )
    from analysis.pcfm_length_scaling import run_length_sweep

    if target in {"channel-spacing", "all"}:
        csv_path, summary_path = run_channel_spacing_sweep(
            cfg_path=cfg_path,
            spacing_ghz_values=_parse_float_list(args.channel_spacing_ghz),
            out_dir=out_root / "channel_spacing",
            launch_dbm=launch_dbm,
            pcfm_numeric_xci=pcfm_numeric_xci,
            recompute_td=recompute_td,
            recompute_pcfm=recompute_pcfm,
            exclude_self_channel=exclude_self_channel,
        )
        results.append(("channel spacing", csv_path, summary_path))

    if target in {"baud-rate", "all"}:
        csv_path, summary_path = run_baud_sweep(
            cfg_path=cfg_path,
            baud_rates_gbaud=_parse_float_list(args.baud_rates_gbaud),
            out_dir=out_root / "baud_rate",
            launch_dbm=launch_dbm,
            pcfm_numeric_xci=pcfm_numeric_xci,
            recompute_td=recompute_td,
            recompute_pcfm=recompute_pcfm,
            exclude_self_channel=exclude_self_channel,
        )
        results.append(("baud rate", csv_path, summary_path))

    if target in {"length", "all"}:
        csv_path, summary_path = run_length_sweep(
            cfg_path=cfg_path,
            lengths_km=_parse_float_list(args.lengths_km),
            out_dir=out_root / "length",
            launch_dbm=launch_dbm,
            pcfm_numeric_xci=pcfm_numeric_xci,
            recompute_td=recompute_td,
            recompute_pcfm=recompute_pcfm,
            exclude_self_channel=exclude_self_channel,
        )
        results.append(("length", csv_path, summary_path))

    for label, csv_path, summary_path in results:
        lg.success(f"{label}: CSV -> {csv_path}")
        lg.success(f"{label}: summary -> {summary_path}")


if __name__ == "__main__":
    main()
