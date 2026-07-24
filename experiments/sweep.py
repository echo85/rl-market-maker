"""Parallel feature-set sweep over the regime-changing market env.

Run from the project root:

    # Quick screen (1 seed, 300 episodes, final eval only) for triage:
    python -m experiments.sweep --configs all --screen

    # Full sweep on a chosen subset of feature sets:
    python -m experiments.sweep --configs "Set B;Set C;Set 3" --seeds 3

    # Customise workers and data:
    python -m experiments.sweep --workers 8 \
        --data-path data/df_normalized.csv

The output CSV and history NPZ files are auto-named from the (env mode,
timestamp) tuple, so each invocation writes to its own files and nothing
clobbers previous runs:

    results/feature_sweep__{stoch_regime|stoch_noregime|det}__YYYYMMDD_HHMMSS.csv
    results/histories/{stoch_regime|stoch_noregime|det}__YYYYMMDD_HHMMSS/<cfg>__seed<N>.npz

The deterministic env has no hidden regime, so --no-regime is silently
ignored when --deterministic is set.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow `python -m experiments.sweep` from the project root
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.linear_sarsa import LinearSARSAAgent
from utils.training import train

from experiments.feature_configs import (
    ENV_SEED,
    FEATURE_CONFIGS,
    CONFIG_NAMES,
    EVAL_EPISODES,
    EVAL_EVERY,
    EVAL_MAX_STEPS,
    MAX_STEPS_PER_EP,
    N_EPISODES,
    _safe_name,
    load_data,
    make_env,
    make_feature_extractor,
)


CSV_COLUMNS = [
    "timestamp", "config", "seed", "alpha", "n_features",
    "mode", "train_return", "eval_return",
    "eval_return_initial", "eval_return_slope", "train_return_slope",
    "weight_norm", "elapsed_s",
]

HISTORY_DIR = os.path.join("results", "histories")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_tag(stochastic: bool, regime_changing: bool, ts: str) -> str:
    """Stable tag for the (stochastic, regime_changing, timestamp) combination.

    The deterministic env ignores the regime flag, so det has no regime
    suffix (avoids redundant det_regime / det_noregime combos).

    The timestamp makes the tag unique per sweep invocation, so successive
    runs never clobber each other on disk.
    """
    if not stochastic:
        return f"det__{ts}"
    regime = "regime" if regime_changing else "noregime"
    return f"stoch_{regime}__{ts}"


def _default_out_csv(stochastic: bool, regime_changing: bool, ts: str) -> str:
    return os.path.join(
        "results", f"feature_sweep__{_env_tag(stochastic, regime_changing, ts)}.csv",
    )


def _history_dir(stochastic: bool, regime_changing: bool, ts: str) -> str:
    return os.path.join(HISTORY_DIR, _env_tag(stochastic, regime_changing, ts))


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope; returns NaN if there are fewer than 2 points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _history_path(cfg_name: str, seed: int,
                  stochastic: bool, regime_changing: bool, ts: str) -> str:
    return os.path.join(
        _history_dir(stochastic, regime_changing, ts),
        f"{_safe_name(cfg_name)}__seed{seed}.npz",
    )


def _save_history(path: str, history, agent=None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    save_dict = {
        "episode_returns": np.asarray(history.episode_returns, dtype=float),
        "eval_episodes": np.asarray(history.eval_episodes, dtype=int),
        "eval_mean_returns": np.asarray(history.eval_mean_returns, dtype=float),
        "eval_std_returns": np.asarray(history.eval_std_returns, dtype=float),
        "weight_norms": np.asarray(history.weight_norms, dtype=float),
    }
    
    if agent is not None:
        save_dict["weights"] = agent.W
        
    np.savez(path, **save_dict)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one(args: Tuple[str, int, int, int, float, str, bool, bool, str, int, str]) -> Dict[str, Any]:
    """Run a single (config, seed) training. Executed in a worker process.

    Only primitives are passed in (no env / agent objects) so pickling is
    cheap and each worker is fully self-contained.

    The env is seeded with ``env_seed`` (default ``MASTER_SEED``) so the
    train and eval envs follow the *same* stochastic path. The loop
    ``seed`` only varies the tile-coder offset jitter, matching the
    notebook's evaluation protocol. The ``ts`` is the same timestamp
    used by the main process to name the CSV/NPZ files.
    """
    (cfg_name, seed, episodes, eval_every, alpha,
     data_path, stochastic, regime_changing, mode, env_seed, ts) = args

    df = load_data(data_path)
    env_train = make_env(df, stochastic=stochastic, regime_changing=regime_changing)
    env_eval = make_env(df, stochastic=stochastic, regime_changing=regime_changing)

    from experiments.feature_configs import get_config
    cfg = get_config(cfg_name)
    feat = make_feature_extractor(cfg, seed)

    agent = LinearSARSAAgent(
        feature_extractor=feat,
        n_actions=env_train.action_space.n,
        alpha=alpha,
        epsilon_start=1.0,
        epsilon_decay=0.997,
        epsilon_min=0.05,
        seed=seed,
    )

    t0 = time.time()
    history = train(
        agent, env_train, n_episodes=episodes,
        max_steps_per_episode=MAX_STEPS_PER_EP,
        eval_every=eval_every,
        eval_max_steps=EVAL_MAX_STEPS,
        eval_episodes=EVAL_EPISODES,
        eval_env=env_eval,
        seed=env_seed,
        progress=False,
    )
    elapsed = time.time() - t0

    # Persist the full trajectory for later plotting
    try:
        _save_history(
            _history_path(cfg_name, seed, stochastic, regime_changing, ts), history, agent
        )
    except Exception as e:
        print(f"  [warn] could not save history for {cfg_name} seed={seed}: {e}",
              file=sys.stderr)

    # Diagnostics
    eval_eps = np.asarray(history.eval_episodes, dtype=float)
    eval_means = np.asarray(history.eval_mean_returns, dtype=float)
    eval_return_initial = float(eval_means[0]) if len(eval_means) else float("nan")
    eval_return_slope = _linear_slope(eval_eps, eval_means)

    ep_returns = np.asarray(history.episode_returns, dtype=float)
    if len(ep_returns) >= 5:
        tail = max(5, len(ep_returns) // 5)  # last 20% (or 5 episodes minimum)
        window = ep_returns[-tail:]
        # Smooth a little with a small moving average so the slope is not pure noise
        k = min(5, len(window))
        if k > 1:
            kernel = np.ones(k) / k
            smoothed = np.convolve(window, kernel, mode="valid")
            x = np.arange(len(smoothed))
            train_return_slope = _linear_slope(x, smoothed)
        else:
            train_return_slope = 0.0
    else:
        train_return_slope = float("nan")

    return {
        "config": cfg_name,
        "seed": seed,
        "alpha": alpha,
        "n_features": int(feat.n_features),
        "mode": mode,
        "train_return": float(np.mean(ep_returns[-20:])) if len(ep_returns) else 0.0,
        "eval_return": float(eval_means[-1]) if len(eval_means) else 0.0,
        "eval_return_initial": eval_return_initial,
        "eval_return_slope": eval_return_slope,
        "train_return_slope": train_return_slope,
        "weight_norm": float(np.linalg.norm(agent.W)),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _open_csv(path: str, restart: bool) -> "tuple[any, bool]":
    """Return (file_handle, wrote_header). Appends unless restart=True."""
    write_header = restart or not os.path.exists(path)
    mode = "w" if restart else "a"
    f = open(path, mode, newline="")
    if write_header:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        f.flush()
    return f, write_header


def _append_row(f, row: Dict[str, Any]) -> None:
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    w.writerow(row)
    f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_configs(spec: str) -> List[str]:
    """Parse the --configs argument.

    Accepted forms:
        "all"                -> all registered configs
        "Set A;Set B;Set C"  -> the listed configs (semicolon-separated,
                                because some config names contain commas)
    """
    spec = (spec or "").strip()
    if not spec or spec.lower() == "all":
        return list(CONFIG_NAMES)
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    out, unknown = [], []
    for p in parts:
        if p in CONFIG_NAMES:
            out.append(p)
        else:
            unknown.append(p)
    if unknown:
        raise SystemExit(
            f"Unknown config(s): {unknown}. Available: {CONFIG_NAMES}"
        )
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a parallel sweep over feature-set configurations.",
    )
    p.add_argument("--configs", default="all",
                   help='Semicolon-separated config names, or "all" '
                        '(default). Use ";" because some names contain commas.')
    p.add_argument("--seeds", type=int, default=3,
                   help="Number of independent seeds per config (default 3).")
    p.add_argument("--episodes", type=int, default=N_EPISODES,
                   help=f"Episodes per training (default {N_EPISODES}).")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY,
                   help=f"Eval interval in episodes (default {EVAL_EVERY}).")
    p.add_argument("--screen", action="store_true",
                   help="Quick-screen mode: defaults to 300 episodes if --episodes "
                        "is not set. The eval schedule is controlled by --eval-every "
                        "(default 50), so 300 episodes with eval-every 50 yields 6 "
                        "evals. Use --eval-every to tune the granularity.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1),
                   help="Number of parallel processes "
                        f"(default {max(1, os.cpu_count() - 1)} = cpu_count-1).")
    p.add_argument("--data-path", default="data/df_normalized.csv",
                   help="Path to a parquet/csv of the normalised price frame.")
    p.add_argument("--out", default=None,
                   help="CSV output path. If omitted, auto-derives "
                        "results/feature_sweep__{stoch_regime|stoch_noregime|det}__YYYYMMDD_HHMMSS.csv "
                        "from the env flags + a per-run timestamp so each "
                        "invocation writes to its own file.")
    p.add_argument("--restart", action="store_true",
                   help="Truncate the output CSV instead of appending. Also "
                        "deletes the matching history NPZ files for this sweep.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use the deterministic (historical-replay) env. The "
                        "hidden regime is not available in this mode; --no-regime "
                        "is silently ignored.")
    p.add_argument("--no-regime", action="store_true",
                   help="Disable the hidden regime. Only meaningful with the "
                        "stochastic env; ignored when --deterministic is set.")
    p.add_argument("--env-seed", type=int, default=ENV_SEED,
                   help=f"Seed for the env's stochastic path (default {ENV_SEED}). "
                        "Fixed across all --seeds runs to match the notebook's "
                        "evaluation protocol. Pass a different value to test "
                        "generalisation to unseen paths.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.screen:
        episodes = args.episodes if args.episodes != N_EPISODES else 300
        eval_every = args.eval_every
        mode = "screen"
        print(f"[screen] {episodes} episodes, eval every {eval_every} "
              f"({episodes // eval_every + 1} evals)")
    else:
        episodes = args.episodes
        eval_every = args.eval_every
        mode = "full"

    configs = parse_configs(args.configs)
    seeds = list(range(args.seeds))
    stochastic = not args.deterministic
    # The deterministic env ignores the regime flag, so force it off there.
    # (--no-regime is silently ignored in --deterministic mode.)
    regime_changing = (not args.no_regime) and stochastic
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Configs  : {configs}")
    print(f"Seeds    : {seeds}")
    print(f"Episodes : {episodes}  eval_every: {eval_every}  mode: {mode}")
    print(f"Workers  : {args.workers}")
    print(f"Data     : {args.data_path}")
    print(f"Timestamp: {ts}")

    # Auto-derive the output CSV path from the env flags + timestamp so each
    # invocation writes to its own file. --out still overrides.
    out_csv = args.out or _default_out_csv(stochastic, regime_changing, ts)
    print(f"Output   : {out_csv}")
    print(f"Env      : stochastic={stochastic}, regime_changing={regime_changing}, "
          f"env_seed={args.env_seed}")

    # Materialise tasks
    tasks = []
    for cfg_name in configs:
        cfg = next(c for c in FEATURE_CONFIGS if c["name"] == cfg_name)
        for seed in seeds:
            tasks.append((
                cfg_name, seed, episodes, eval_every, cfg["alpha"],
                args.data_path, stochastic, regime_changing, mode, args.env_seed, ts,
            ))
    print(f"Total tasks: {len(tasks)}")

    # When --restart is set, also wipe the history NPZ files for this sweep's
    # (config, seed) pairs (in the env-mode subdir) so plot_history doesn't
    # mix stale + new trajectories.
    if args.restart:
        removed = 0
        for cfg_name in configs:
            for seed in seeds:
                path = _history_path(cfg_name, seed, stochastic, regime_changing, ts)
                if os.path.exists(path):
                    os.remove(path)
                    removed += 1
        if removed:
            print(f"[restart] removed {removed} stale history file(s)")

    # Smoke-check the data path before launching workers
    try:
        load_data(args.data_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    csv_file, _ = _open_csv(out_csv, restart=args.restart)

    completed = 0
    t_start = time.time()
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_run_one, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception as e:
                    cfg, seed, *_ = futures[fut]
                    print(f"  [FAIL] {cfg} seed={seed}: {e}", file=sys.stderr)
                    continue
                _append_row(csv_file, row)
                completed += 1
                print(f"  [{completed}/{len(tasks)}] {row['config']:40s} "
                      f"seed={row['seed']}  eval={row['eval_return']:+.4f}  "
                      f"n_feat={row['n_features']:>6d}  "
                      f"({row['elapsed_s']:.1f}s)")
    finally:
        csv_file.close()

    print(f"\nDone. {completed}/{len(tasks)} runs in {time.time()-t_start:.1f}s")
    print(f"Results appended to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
