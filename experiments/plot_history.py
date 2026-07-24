"""Plot training trajectories saved by ``experiments.sweep``.

Each (config, seed, env_mode, timestamp) run writes a small NPZ file
under ``results/histories/<env_mode>__<timestamp>/`` containing the full
per-episode and per-eval-point history. This CLI loads those files and
produces a three-row figure:

    top:    training return vs episode (raw per-episode + 20-ep moving
            average mean ± std across seeds)
    middle: eval return vs episode (mean ± std across seeds)
    bottom: weight norm vs episode (mean ± std across seeds)

The weight-norm subplot makes it easy to spot overfitting: when the
weight norm keeps growing but the eval return plateaus or drops, the
features are over-parameterised for the available data.

Usage:

    # One config to one PNG (reads from results/histories/stoch_regime__<latest>/
    # by default; bare env-mode auto-resolves to the latest timestamped subdir)
    python -m experiments.plot_history --config "Set H - ..." --out results/plots/SetH.png

    # Pick a specific env mode
    python -m experiments.plot_history --config "Set A - ..." --env-mode det --out /tmp/a.png

    # Pick a specific run (env mode + timestamp)
    python -m experiments.plot_history --config "Set A - ..." --env-mode stoch_regime__20260722_094500

    # All configs in the registry
    python -m experiments.plot_history --all --out results/plots/
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow `python -m experiments.plot_history` from the project root
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # safe for headless / non-interactive use
import matplotlib.pyplot as plt

from experiments.feature_configs import CONFIG_NAMES, _safe_name  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Three env modes (the deterministic env has no hidden regime, so det
# has no regime suffix). Each is also a prefix for timestamped subdirs.
_KNOWN_PREFIXES = ("stoch_regime", "stoch_noregime", "det")


def _resolve_history_dir(env_mode: str) -> str:
    """Resolve a (possibly bare) env-mode tag to a concrete history subdir.

    - 'stoch_regime'                      → latest 'stoch_regime__*/' subdir,
                                            or 'stoch_regime' itself if none exist
    - 'stoch_regime__20260722_094500'     → that exact subdir (no resolution)
    - anything else (unknown tag)         → returned as-is
    """
    # Exact timestamped tag: pass through
    if any(env_mode.startswith(p + "__") for p in _KNOWN_PREFIXES) and env_mode not in _KNOWN_PREFIXES:
        return env_mode
    # Bare prefix: pick the latest timestamped subdir
    if env_mode in _KNOWN_PREFIXES:
        candidates = sorted(glob.glob(os.path.join("results", "histories", f"{env_mode}__*")))
        if candidates:
            return os.path.basename(candidates[-1])
    return env_mode


def _history_files_for(cfg_name: str, env_mode: str = "stoch_regime") -> List[str]:
    """Return all NPZ history files for a given config + env mode, sorted by seed.

    The ``env_mode`` arg is resolved via ``_resolve_history_dir`` first, so
    bare tags like ``stoch_regime`` auto-pick the latest timestamped subdir.
    """
    safe = _safe_name(cfg_name)
    subdir = _resolve_history_dir(env_mode)
    pattern = os.path.join("results", "histories", subdir, f"{safe}__seed*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        # Try with absolute path
        files = sorted(glob.glob(os.path.join(os.getcwd(), pattern)))
    return files


def _load_history(path: str) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in np.load(path).items()}


def _seed_from_filename(path: str) -> int:
    base = os.path.basename(path)
    # <safe_name>__seed<N>.npz
    seed_str = base.split("__seed")[-1].split(".")[0]
    try:
        return int(seed_str)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stack_seeds(histories: List[Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align histories on a common eval-episode grid and stack per-seed.

    Returns:
        eval_grid: common 1-D array of eval episodes
        eval_mean: shape (n_seeds, n_evals), NaN-padded where shorter
        eval_std:  same shape
        ep_returns: list of per-episode arrays (variable length — returned as list)
        weight_norms: same as ep_returns
    """
    eval_grids = [h["eval_episodes"] for h in histories if len(h["eval_episodes"]) > 0]
    if not eval_grids:
        # Fall back: no eval points at all. Use episode indices for everything.
        max_len = max(len(h["episode_returns"]) for h in histories)
        eval_grid = np.arange(0, max_len + 1, max(1, max_len // 10))
        eval_mean = np.full((len(histories), len(eval_grid)), np.nan)
        eval_std = np.full_like(eval_mean, np.nan)
        for i, h in enumerate(histories):
            er = h["episode_returns"]
            for j, ep in enumerate(eval_grid):
                idx = min(int(ep), len(er) - 1)
                eval_mean[i, j] = er[idx] if idx >= 0 else np.nan
                eval_std[i, j] = 0.0
        ep_returns = [h["episode_returns"] for h in histories]
        weight_norms = [h["weight_norms"] for h in histories]
        return eval_grid, eval_mean, eval_std, ep_returns, weight_norms

    eval_grid = np.unique(np.concatenate(eval_grids))
    eval_mean = np.full((len(histories), len(eval_grid)), np.nan)
    eval_std = np.full_like(eval_mean, np.nan)
    for i, h in enumerate(histories):
        for j, ep in enumerate(eval_grid):
            mask = h["eval_episodes"] == ep
            if mask.any():
                eval_mean[i, j] = h["eval_mean_returns"][mask][0]
                eval_std[i, j] = h["eval_std_returns"][mask][0]
    ep_returns = [h["episode_returns"] for h in histories]
    weight_norms = [h["weight_norms"] for h in histories]
    return eval_grid, eval_mean, eval_std, ep_returns, weight_norms


def _plot_one(cfg_name: str, out_path: str,
              env_mode: str = "stoch_regime") -> Optional[Dict[str, float]]:
    files = _history_files_for(cfg_name, env_mode)
    if not files:
        print(f"  [skip] {cfg_name}: no history files in "
              f"results/histories/{env_mode}/", file=sys.stderr)
        return None

    histories = [_load_history(p) for p in files]
    seeds = [_seed_from_filename(p) for p in files]
    eval_grid, eval_mean, eval_std, ep_returns, weight_norms = _stack_seeds(histories)

    n_seeds = len(histories)
    em = np.nanmean(eval_mean, axis=0)
    es = np.nanstd(eval_mean, axis=0)

    # Per-episode training return: align on the longest episode grid
    max_ep_len = max(len(r) for r in ep_returns)
    er_arr = np.full((n_seeds, max_ep_len), np.nan)
    for i, r in enumerate(ep_returns):
        er_arr[i, :len(r)] = r

    # 20-episode moving average of training return per seed
    ma_window = 20
    er_ma = np.full_like(er_arr, np.nan)
    for i in range(n_seeds):
        valid = ~np.isnan(er_arr[i])
        if valid.sum() >= ma_window:
            kernel = np.ones(ma_window) / ma_window
            conv = np.convolve(er_arr[i][valid], kernel, mode="valid")
            er_ma[i, ma_window - 1: ma_window - 1 + len(conv)] = conv
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        er_ma_mean = np.nanmean(er_ma, axis=0)
        er_ma_std = np.nanstd(er_ma, axis=0)

    # Weight-norm: align on the longest episode grid available
    wn = np.full((n_seeds, max_ep_len), np.nan)
    for i, w in enumerate(weight_norms):
        wn[i, :len(w)] = w
    wn_mean = np.nanmean(wn, axis=0)
    wn_std = np.nanstd(wn, axis=0)

    fig, (ax_train, ax_eval, ax_wn) = plt.subplots(
        3, 1, figsize=(9, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2]},
    )
    fig.suptitle(f"{cfg_name}  (n_seeds={n_seeds})", fontsize=11, y=0.995)

    # ---- Top: per-episode training return + 20-ep moving average ----
    ep_x_full = np.arange(max_ep_len)
    # Faint raw per-episode return per seed
    for i in range(n_seeds):
        ax_train.plot(ep_x_full, er_arr[i], color="grey", alpha=0.18, lw=0.6)
    # Bold moving-average mean across seeds
    ax_train.plot(ep_x_full, er_ma_mean, color="seagreen", lw=2.0,
                   label=f"train return ({ma_window}-ep MA mean)")
    ax_train.fill_between(ep_x_full, er_ma_mean - er_ma_std, er_ma_mean + er_ma_std,
                           color="seagreen", alpha=0.20, label="train MA ± std")
    ax_train.axhline(0.0, color="grey", lw=0.6, ls="--")
    ax_train.set_ylabel("Training return")
    ax_train.grid(alpha=0.3)
    ax_train.legend(loc="best", fontsize=9)

    # ---- Middle: eval return (mean ± std across seeds) ----
    ax_eval.plot(eval_grid, em, color="steelblue", lw=2, label="eval mean")
    ax_eval.fill_between(eval_grid, em - es, em + es,
                          color="steelblue", alpha=0.20, label="eval ± std")
    for i in range(n_seeds):
        ax_eval.plot(eval_grid, eval_mean[i], color="steelblue", alpha=0.20, lw=0.8)
    ax_eval.axhline(0.0, color="grey", lw=0.6, ls="--")
    ax_eval.set_ylabel("Eval return")
    ax_eval.grid(alpha=0.3)
    ax_eval.legend(loc="best", fontsize=9)

    # ---- Bottom: weight norm (mean ± std across seeds) ----
    ax_wn.plot(ep_x_full, wn_mean, color="darkorange", lw=2, label="weight norm mean")
    ax_wn.fill_between(ep_x_full, wn_mean - wn_std, wn_mean + wn_std,
                        color="darkorange", alpha=0.20, label="weight norm ± std")
    for i in range(n_seeds):
        ax_wn.plot(ep_x_full, wn[i], color="darkorange", alpha=0.18, lw=0.8)
    ax_wn.set_xlabel("Episode")
    ax_wn.set_ylabel("Weight norm")
    ax_wn.grid(alpha=0.3)
    ax_wn.legend(loc="best", fontsize=9)

    # Summary (on the training subplot — the headline panel)
    slope = float(np.polyfit(eval_grid.astype(float), em, 1)[0]) if len(eval_grid) >= 2 else float("nan")
    final_eval = float(em[-1]) if len(em) else float("nan")
    initial_eval = float(em[0]) if len(em) else float("nan")
    final_train = float(er_ma_mean[~np.isnan(er_ma_mean)][-1]) if np.any(~np.isnan(er_ma_mean)) else float("nan")
    initial_train = float(er_ma_mean[~np.isnan(er_ma_mean)][0]) if np.any(~np.isnan(er_ma_mean)) else float("nan")
    final_wn = float(wn_mean[-1]) if len(wn_mean) else float("nan")

    ax_train.set_title(
        f"eval slope={slope:+.5f}/ep  eval: {initial_eval:+.4f} → {final_eval:+.4f}  |  "
        f"train MA: {initial_train:+.4f} → {final_train:+.4f}  |  |W| final={final_wn:.3f}",
        fontsize=10, loc="left",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return {
        "config": cfg_name,
        "n_seeds": n_seeds,
        "slope": slope,
        "initial": initial_eval,
        "final": final_eval,
        "train_initial": initial_train,
        "train_final": final_train,
        "weight_norm_final": final_wn,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot training trajectories from results/histories/.",
    )
    p.add_argument("--config", default=None,
                   help="Plot a single config (must match a name in feature_configs).")
    p.add_argument("--all", action="store_true",
                   help="Plot every config in the registry that has history files.")
    p.add_argument("--env-mode", default="stoch_regime",
                   help="Env-mode to read. Either a bare prefix (stoch_regime, "
                        "stoch_noregime, det) which auto-resolves to the latest "
                        "timestamped run, or a full tag like "
                        "'stoch_regime__20260722_094500' for a specific run.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed filter (e.g. '0,1,2'). Currently informational only.")
    p.add_argument("--out", default=None,
                   help="Output PNG path. Defaults to results/plots/<env_mode>/. "
                        "If the path is a directory (or ends with /), one PNG "
                        "per config is written inside it.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if not args.config and not args.all:
        args.all = True

    # Resolve the env-mode: a bare prefix (e.g. 'stoch_regime') is upgraded
    # to the latest timestamped subdir; a full tag is kept as-is.
    resolved_env_mode = _resolve_history_dir(args.env_mode)

    # Default the output dir to results/plots/<resolved_env_mode>/
    out_dir = args.out
    if out_dir is None:
        out_dir = os.path.join("results", "plots", resolved_env_mode)

    if args.all:
        targets = list(CONFIG_NAMES)
        if not out_dir.endswith((".png", ".pdf", ".svg")):
            os.makedirs(out_dir, exist_ok=True)
        # One PNG per config
        summaries = []
        for cfg in targets:
            safe = _safe_name(cfg)
            if out_dir.endswith((".png", ".pdf", ".svg")):
                out_path = out_dir
            else:
                out_path = os.path.join(out_dir, f"{safe}.png")
            s = _plot_one(cfg, out_path, env_mode=resolved_env_mode)
            if s is not None:
                summaries.append(s)
        if summaries:
            print(f"\n{'config':<55s} {'n':>3s} {'slope':>10s} {'eval_init':>10s} {'eval_final':>10s} {'train_init':>11s} {'train_final':>11s} {'|W|':>7s}")
            print("-" * 125)
            for s in summaries:
                print(f"{s['config'][:54]:<55s} {s['n_seeds']:>3d} "
                      f"{s['slope']:>+10.5f} {s['initial']:>+10.4f} "
                      f"{s['final']:>+10.4f} {s.get('train_initial', float('nan')):>+11.4f} "
                      f"{s.get('train_final', float('nan')):>+11.4f} "
                      f"{s['weight_norm_final']:>7.4f}")
        return 0

    if args.config:
        out_path = out_dir
        # If --out is a directory, append the safe config name
        if os.path.isdir(out_path) or out_path.endswith("/"):
            os.makedirs(out_path, exist_ok=True)
            out_path = os.path.join(out_path, f"{_safe_name(args.config)}.png")
        s = _plot_one(args.config, out_path, env_mode=resolved_env_mode)
        if s is None:
            return 1
        if resolved_env_mode != args.env_mode:
            print(f"  env-mode     = {args.env_mode}  (resolved to {resolved_env_mode})")
        else:
            print(f"  env-mode     = {resolved_env_mode}")
        print(f"  eval slope  = {s['slope']:+.5f}/episode")
        print(f"  eval:        {s['initial']:+.4f} -> {s['final']:+.4f}")
        print(f"  train MA:    {s.get('train_initial', float('nan')):+.4f} -> {s.get('train_final', float('nan')):+.4f}")
        print(f"  |W| final   = {s['weight_norm_final']:.4f}")
        print(f"  saved -> {out_path}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
