from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.config import _load_pcfm_runtime_config, load_studies_runtime_config
from analysis.subset import resolve_subset


class _DummyWDM:
    def __init__(self, freqs):
        self._freqs = np.asarray(freqs, dtype=float)
        self.num_channels = self._freqs.size

    def frequency_grid(self):
        return self._freqs


def _system(raw_config, freqs=(193.0e12, 193.05e12, 193.1e12, 193.15e12, 193.2e12)):
    wdm = _DummyWDM(freqs)
    return SimpleNamespace(raw_config=raw_config, wdm=wdm, n_channels=wdm.num_channels)


def test_load_studies_runtime_config_new_schema():
    system = _system(
        {
            "profiles": {
                "mode": "flat",
                "path": "results/test_profile.npy",
                "launch_csv": "results/launch.csv",
            },
            "methods": {
                "td": {"mode": "recompute", "exclude_self_channel": False, "m_lo_truncation": 3},
                "pcfm": {"mode": "cached", "numeric_xci": True, "degree": 7},
                "mc": {"mode": "off", "n_trials": 4, "rng_seed": 99},
            },
            "studies": {
                "center_refined": {
                    "type": "subset",
                    "methods": ["td", "pcfm", "mc"],
                    "out_dir": "results/studies/test_center",
                    "subset": {"mode": "center_window", "center": "auto", "half_width": 1},
                }
            },
        }
    )

    cfg = load_studies_runtime_config(system)

    assert cfg.profiles.mode == "flat"
    assert str(cfg.profiles.path) == "results/test_profile.npy"
    assert str(cfg.profiles.launch_csv) == "results/launch.csv"
    assert cfg.methods.td.mode == "recompute"
    assert cfg.methods.td.exclude_self_channel is False
    assert cfg.methods.td.m_lo_truncation == 3
    assert cfg.methods.pcfm.numeric_xci is True
    assert cfg.methods.pcfm.degree == 7
    assert cfg.methods.mc.n_trials == 4
    assert cfg.methods.mc.rng_seed == 99
    assert len(cfg.studies) == 1
    assert cfg.studies[0].name == "center_refined"
    assert cfg.studies[0].methods == ("td", "pcfm", "mc")
    assert cfg.studies[0].subset.half_width == 1

    pcfm_runtime = _load_pcfm_runtime_config(system)
    assert pcfm_runtime["pcfm_degree"] == 7


def test_resolve_center_window_subset_auto_center():
    system = _system({})
    cfg = load_studies_runtime_config(
        _system(
            {
                "studies": {
                    "center": {
                        "type": "subset",
                        "methods": ["td"],
                        "subset": {"mode": "center_window", "center": "auto", "half_width": 1},
                    }
                }
            }
        )
    )

    subset = resolve_subset(system, cfg.studies[0].subset)

    assert subset.cut_indices == (2,)
    assert subset.interferer_indices == (1, 2, 3)
    assert subset.center_index == 2
    assert subset.tag == "cuts2_ints1-2-3_sci1"


def test_resolve_explicit_subset_without_sci():
    system = _system({})
    cfg = load_studies_runtime_config(
        _system(
            {
                "studies": {
                    "explicit": {
                        "type": "subset",
                        "methods": ["td"],
                        "subset": {
                            "mode": "explicit",
                            "cut_indices": [2],
                            "interferer_indices": [1, 2, 3],
                            "include_sci": False,
                        },
                    }
                }
            }
        )
    )

    subset = resolve_subset(system, cfg.studies[0].subset)

    assert subset.cut_indices == (2,)
    assert subset.interferer_indices == (1, 3)
    assert subset.tag == "cuts2_ints1-3_sci0"
