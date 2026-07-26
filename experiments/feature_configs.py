"""Registry of feature-set configurations and helpers to instantiate them.

Configs are kept as plain dicts (with `low`/`high` as Python lists) so they
pickle cleanly when shipped across process boundaries by
``experiments.sweep``.

Each entry has at least:

    name              str
    kind              "tile" | "rbf" | "poly"
    feature_indices   list[int]
    low, high         list[float] of equal length to feature_indices
    alpha             float  (per-tiling for tile coding, raw for others)

plus the kind-specific knobs.

NOTE on observation layout. The env currently exposes an 11-D obs vector
with the following indices:

    0..2  prices (3 assets)
    3..5  allocation per asset
    6     cash ratio
    7     RSI of portfolio value
    8     avg correlation across assets
    9     portfolio return std (vol)
    10    MACD histogram of portfolio value, normalized by port_val

The low/high ranges below were calibrated before the
portfolio-value switch; they are
good enough to start but should be recalibrated against the current obs
layout for best results.
Adding a new config is one dict entry.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from envs.market import MarketEnv
from utils.features import TileCoder, RBFFeatures, PolynomialFeatures, RawRepresentation


# ---------------------------------------------------------------------------
# Training schedule (mirrors the notebook's cell 8)
# ---------------------------------------------------------------------------

N_EPISODES: int = 2000
EVAL_EVERY: int = 50
EVAL_EPISODES: int = 20
MAX_STEPS_PER_EP: int = 250
EVAL_MAX_STEPS: int = 250
MASTER_SEED: int = 42

# The env's stochastic path is fixed across all "seed" runs to match the
# notebook's evaluation protocol (notebook section 5 / 8). Vary the
# `--env-seed` flag if you want to test generalisation to unseen paths.
ENV_SEED: int = MASTER_SEED

NOMINAL_ALPHA: float = 0.1
TC_N_TILES: List[int] = [5, 5, 5, 5]
TC_N_TILINGS: int = 4
TC_ALPHA: float = NOMINAL_ALPHA / TC_N_TILINGS

RBF_N_CENTERS: List[int] = [4, 4, 4, 4]
RBF_SIGMA: float = 0.1
RBF_ALPHA: float = NOMINAL_ALPHA

POLY_ALPHA: float = 0.0001


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

def _as_list(x) -> List[float]:
    return list(np.asarray(x, dtype=float).ravel())


def _safe_name(s: str) -> str:
    """Filesystem-safe version of a string (alphanumeric + underscores)."""
    import re
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return s.strip("_")


FEATURE_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "Set A - Raw (prices)",
        "kind": "raw",
        "feature_indices": [0, 1, 2],
        "low": [0, 0, 0],
        "high": [1500, 1500, 1500],
        "alpha": 0.0001,
    },
    # ---- Set 1: raw signals (tile coding) -------------------------------
    {
        "name": "Set A - TileCoder (Raw Prices)",
        "kind": "tile",
        "feature_indices": [0, 1, 2],
        "low": _as_list([0, 0, 0]),
        "high": _as_list([1500, 1500, 1500]),
        "n_tiles": list([5, 5, 5]),
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    {
        "name": "Set B - TileCoder (asset allocation)",
        "kind": "tile",
        "feature_indices": [3, 4, 5, 6],
        "low": _as_list([0, 0, 0, 0]),
        "high": _as_list([1, 1, 1, 1]),
        "n_tiles": list(TC_N_TILES),
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    # ---- Set M: indicators (sma, RSI, avg_corr) ------------------------
    {
        "name": "Set C - TileCoder (port_std)",
        "kind": "tile",
        "feature_indices": [9],
        "low": _as_list([0]),
        "high": _as_list([0.03]),
        "n_tiles": [8],
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    # ---- Set M: indicators (sma, RSI, avg_corr) ------------------------
    {
        "name": "Set D - TileCoder (MACD, port_std)",
        "kind": "tile",
        "feature_indices": [10,9],
        "low": _as_list([-0.015,0]),
        "high": _as_list([0.015,0.03]),
        "n_tiles": [6,6],
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    # ---- Set I: indicators + MACD histogram (tile coding) -------------
    {
        "name": "Set E - TileCoder (MACD, port_std, RSI)",
        "kind": "tile",
        "feature_indices": [10, 9, 7],   # MACD, port_std, RSI
        "low": _as_list([-0.015, 0, 0]),
        "high": _as_list([0.015, 0.03, 100]),
        "n_tiles": [4, 5, 4],
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    # ---- Set I: indicators + MACD histogram (tile coding) -------------
    {
        "name": "Set F - TileCoder (avg_corr, RSI, MACD)",
        "kind": "tile",
        "feature_indices": [8, 7, 10],   # avg_corr, RSI, MACD
        "low": _as_list([-0.5, 0,  -0.015]),
        "high": _as_list([0.75, 100, 0.015]),
        "n_tiles": [4, 5, 4],
        "n_tilings": TC_N_TILINGS,
        "alpha": TC_ALPHA,
    },
    # ---- Set 3: polynomial features (degree 2) -------------------------
    {
        "name": "Set E - Polynomial (RSI, MACD, port_std)",
        "kind": "poly",
        "feature_indices": [7, 10, 9], # RSI, MACD, port_std
        "low": _as_list([0, -0.015, 0]),
        "high": _as_list([100, 0.015, 0.03]),
        "degree": 2,
        "alpha": POLY_ALPHA,
    },
    # ---- Set 2: hand-crafted indicators (RBF) --------------------------
    {
        "name": "Set D - RBF (MACD, port_std)",
        "kind": "rbf",
        "feature_indices": [10,9],
        "low": _as_list([-0.015,0]),
        "high": _as_list([0.015,0.03]),
        "n_centers": [4, 4],
        "sigma": RBF_SIGMA,
        "normalize": True,
        "alpha": RBF_ALPHA,
    },
    # ---- Set I: indicators + MACD histogram (tile coding) -------------
    {
        "name": "Set E - RBF (port_std, RSI, MACD)",
        "kind": "rbf",
        "feature_indices": [9, 7, 10],   # port_std, RSI, MACD
        "low": _as_list([0, 0,  -0.015]),
        "high": _as_list([0.03, 100, 0.015]),
        "n_centers": [4, 4, 4],
        "sigma": RBF_SIGMA,
        "normalize": True,
        "alpha": RBF_ALPHA,
    },
    # ---- Set 3: polynomial features (degree 2) -------------------------
    {
        "name": "Set B - Polynomial (asset allocation)",
        "kind": "poly",
        "feature_indices": [3, 4, 5, 6], 
        "low": _as_list([0, 0, 0, 0]),
        "high": _as_list([1, 1, 1, 1]),
        "degree": 2,
        "alpha": POLY_ALPHA,
    }

]


CONFIG_NAMES: List[str] = [c["name"] for c in FEATURE_CONFIGS]


def get_config(name: str) -> Dict[str, Any]:
    for c in FEATURE_CONFIGS:
        if c["name"] == name:
            return c
    raise KeyError(f"Unknown feature config: {name!r}. "
                   f"Available: {CONFIG_NAMES}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def make_feature_extractor(cfg: Dict[str, Any], seed: int):
    """Instantiate a fresh feature extractor for ``cfg`` with a given ``seed``.

    The seed only matters for tile coding (offset jitter); RBF and polynomial
    extractors are deterministic.
    """
    low = np.asarray(cfg["low"], dtype=np.float64)
    high = np.asarray(cfg["high"], dtype=np.float64)
    indices = list(cfg["feature_indices"])

    kind = cfg["kind"]
    if kind == "tile":
        return TileCoder(
            low=low, high=high,
            n_tiles=cfg["n_tiles"],
            n_tilings=cfg["n_tilings"],
            feature_indices=indices,
            seed=seed,
        )
    if kind == "rbf":
        return RBFFeatures(
            low=low, high=high,
            n_centers=cfg["n_centers"],
            sigma=cfg["sigma"],
            normalize=cfg.get("normalize", True),
            feature_indices=indices,
        )
    if kind == "poly":
        return PolynomialFeatures(
            state_dim=11,
            degree=cfg["degree"],
            feature_indices=indices,
            low=low, high=high,
        )
    if kind == "raw":
        return RawRepresentation(
            state_dim=11,
            feature_indices=indices,
            low=low, high=high,
        )
    raise ValueError(f"Unknown feature kind: {kind!r}")


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(df_prices: pd.DataFrame, *, stochastic: bool = True,
             regime_changing: bool = True) -> MarketEnv:
    """Build a fresh env. Stochastic + regime_changing matches notebook
    section 8 (the harder variant) and is the default for the sweep."""
    return MarketEnv(
        df_prices=df_prices,
        stochastic=stochastic,
        regime_changing=regime_changing,
    )


def load_data(path: str) -> pd.DataFrame:
    """Load a price frame from parquet or csv based on extension.

    For CSV, the index is parsed as a date only when the first column
    *looks* like a date — otherwise it is kept as-is (int/str). This avoids
    a ``dateutil`` warning on plain integer-indexed test data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path!r}. "
            "Run the notebook's data-prep cell to dump df_normalized to "
            "data/df_normalized.csv, or pass --data-path explicitly."
        )
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        # Peek at the first data row (line 2) to decide whether to parse
        # the index as dates. The header line is just "Date,...".
        with open(path, "r") as f:
            f.readline()  # skip header
            first = f.readline().split(",", 1)[0].strip()
        import re
        parse_dates = bool(re.match(r"^\d{4}-\d{2}-\d{2}", first))
        return pd.read_csv(path, index_col=0, parse_dates=parse_dates)
    raise ValueError(f"Unsupported data extension: {path}")


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def results_path(out_dir: str = "results") -> str:
    return os.path.join(out_dir, "feature_sweep.csv")


def load_results(path: Optional[str] = None) -> "pd.DataFrame":
    """Load the sweep results CSV (empty DataFrame if it does not exist)."""
    import pandas as pd
    if path is None:
        path = results_path()
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp", "config", "seed", "alpha", "n_features",
            "mode", "train_return", "eval_return", "weight_norm", "elapsed_s",
        ])
    return pd.read_csv(path)


def summarize(df: "pd.DataFrame", by: str = "config", mode: Optional[str] = None):
    """Group results by ``by`` and return mean/std/count of eval_return."""
    import pandas as pd
    sub = df if mode is None else df[df["mode"] == mode]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(by)["eval_return"]
           .agg(["mean", "std", "count"])
           .sort_values("mean", ascending=False)
    )
