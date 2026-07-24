# RL Market Maker — Feature Engineering for Linear Function Approximation

A university project focusing on Reinforcement Learning (RL) applied to market making.

## Main Focus
**Linear Function Approximation** and the role of feature design in value-based RL.

## Scientific Objective
This project focuses on the interplay between feature engineering and learning performance in the context of Linear Function Approximation (FA). 

## Problem Description
The agent manages a small portfolio of assets over a finite time horizon. At each step, it observes a set of market signals and decides how to allocate its budget across a discrete set of actions (e.g., buy, hold, sell for each asset). Prices evolve according to a hidden stochastic process with occasional regime changes, requiring the agent to adapt its strategy over time. The reward at each step reflects the portfolio return.

## Project Structure & Implementation
The repository contains both an interactive notebook for step-by-step exploration and a suite of scripts for running parallel experiments on feature sets.

* **`rl-market-maker.ipynb`**: The main Jupyter Notebook. It handles environment exploration, downloading historical market data (via `yfinance`), normalizing prices, and provides an evaluation protocol for the agent.
* **`agents/`**: Contains the reinforcement learning agents, such as `LinearSARSAAgent`.
* **`experiments/`**: A module dedicated to running large-scale feature experiments.
  * **`sweep.py`**: A script to run parallel feature-set sweeps over the market environment. It trains agents across different configurations and seeds, saving the history and results.
  * **`feature_configs.py`**: A registry of various feature-set configurations. It defines the observation ranges and different feature extractors like Tile Coding, Radial Basis Functions (RBF), and Polynomial Features.
  * **`plot_history.py`**: A CLI tool to plot training trajectories (training returns, evaluation returns, and weight norms). This is particularly useful for spotting overfitting when weight norms grow but evaluation returns plateau.

## Running Experiments

You can run the feature sweeps and plot the results directly from the command line:

**1. Quick Screen (Triage)**
Run a quick screen with 1 seed and 300 episodes on all configurations:
```bash
python -m experiments.sweep --configs all --screen
```

**2. Full Sweep**
Run a full sweep on a chosen subset of feature sets with multiple seeds:
```bash
python -m experiments.sweep --configs "Set A - Raw (TileCoder);Set B" --seeds 3
```

**3. Plotting Results**
Plot the training trajectories for all configurations and save them to a directory:
```bash
python -m experiments.plot_history --all --out results/plots/
```
