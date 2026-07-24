import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

class MarketEnv(gym.Env):
    """
    Unified Market Environment.
    If stochastic=True, uses historical data to calibrate drift and covariance, 
    and simulates random paths.
    If stochastic=False, replays the exact historical data timeline.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df_prices: pd.DataFrame,
        stochastic: bool = False,
        initial_cash: float = 10000.0,
        rsi_window: int = 14,
        corr_window: int = 20,
        vol_window: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        regime_changing: bool = False,
        crisis_mu_scale: float = -2.0,
        crisis_cov_scale: float = 3.0,
        p_stay_normal: float = 0.97,
        p_stay_crisis: float = 0.85,
    ):
        super().__init__()

        self.df_prices = df_prices
        self.stochastic = stochastic
        self.initial_cash = initial_cash
        self.rsi_window = rsi_window
        self.corr_window = corr_window
        self.vol_window = vol_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

        self.regime_changing = regime_changing
        self.crisis_mu_scale = float(crisis_mu_scale)
        self.crisis_cov_scale = float(crisis_cov_scale)
        self.p_stay_normal = float(p_stay_normal)
        self.p_stay_crisis = float(p_stay_crisis)

        # MACD needs slow + signal points of history to define both EMAs.
        self.warmup_steps = max(
            rsi_window, corr_window, vol_window,
            macd_slow + macd_signal,
        )
        self.num_risky_assets = len(df_prices.columns)
        self.asset_names = list(df_prices.columns)

        if self.stochastic:
            returns = df_prices.pct_change().dropna()
            self.mu_normal = returns.mean().values
            self.cov_normal = returns.cov().values
            if self.regime_changing:
                # Crisis regime: negative drift shock + inflated covariance.
                self.mu_crisis = self.mu_normal + self.crisis_mu_scale * np.abs(self.mu_normal)
                self.cov_crisis = self.cov_normal * self.crisis_cov_scale
            else:
                self.mu_crisis = self.mu_normal
                self.cov_crisis = self.cov_normal
        else:
            # For historical replay, we can only run as many steps as we have data for
            self.max_steps = len(df_prices) - self.warmup_steps - 1

        # Action space: Discrete(3^num_risky_assets), one op per asset in fixed-width
        # little-endian encoding. Each op_i in {0: sell 10% of position, 1: hold,
        # 2: buy 10% of available cash}. action = sum_i op_i * 3**i.
        self.action_space = spaces.Discrete(3 ** self.num_risky_assets)

        # 11-D state: prices(3) + alloc_assets(3) + cash_ratio(1) + rsi(1) + corr(1) + port_std(1) + macd(1)
        #   0..2  prices
        #   3..5  alloc_assets
        #   6     cash_ratio
        #   7     RSI of portfolio value
        #   8     avg_correlation
        #   9     port_std
        #   10    MACD histogram of portfolio value, normalized by port_val
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Regime helpers (hidden from the agent; exposed via info for plots)
    # ------------------------------------------------------------------ #

    def _regime_params(self):
        if self.regime == 1:
            return self.mu_crisis, self.cov_crisis
        return self.mu_normal, self.cov_normal

    def _step_regime(self):
        if not self.regime_changing:
            self.regime = 0
            return
        if self.regime == 0:
            self.regime = 1 if self.np_random.random() > self.p_stay_normal else 0
        else:
            self.regime = 1 if self.np_random.random() < self.p_stay_crisis else 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cash = self.initial_cash
        self.shares = np.zeros(self.num_risky_assets)
        self.avg_cost = np.zeros(self.num_risky_assets)

        self.price_history = []
        self.portfolio_value_history = []
        self.allocation_history = []
        self.regime_history = []

        if self.stochastic:
            self.current_step = 0
            # Start in the normal regime.
            self.regime = 0
            # For stochastic, initial prices are the last known prices of the historical dataset
            current_prices = self.df_prices.iloc[-1].values.copy()

            for _ in range(self.warmup_steps):
                self.price_history.append(current_prices.copy())
                self.portfolio_value_history.append(self.initial_cash)
                self.regime_history.append(self.regime)
                current_prices = self._simulate_next_prices(current_prices)

            self.price_history.append(current_prices.copy())
            self.portfolio_value_history.append(self.initial_cash)
            self.regime_history.append(self.regime)
            self.current_prices = current_prices
        else:
            self.current_step = self.warmup_steps
            self.regime = 0

            # Extract historical warmup prices directly
            for i in range(self.warmup_steps + 1):
                self.price_history.append(self.df_prices.iloc[i].values.copy())
                self.portfolio_value_history.append(self.initial_cash)
                self.regime_history.append(self.regime)

            self.current_prices = self.df_prices.iloc[self.current_step].values.copy()

        obs = self._get_obs()
        self.allocation_history.append(obs[3:7])
        return obs, {"portfolio_value": self.initial_cash, "cash": self.cash}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        action = int(np.asarray(action).item())

        # 1. Execute Trades (joint action: one op per asset, sequential in asset order)
        SELL_PENALTY_RATE = 0.1
        n = self.num_risky_assets
        for i in range(n):
            op = (action // (3 ** i)) % 3
            if op == 0: # sell
                sell_shares = self.shares[i] * 0.10
                if sell_shares > 0:
                    revenue = float(sell_shares * self.current_prices[i])
                    penalty = revenue * SELL_PENALTY_RATE
                    self.shares[i] -= sell_shares
                    self.cash += (revenue - penalty)
            elif op == 2: # buy
                buy_amount = self.cash * 0.10
                if buy_amount > 0:
                    buy_shares = buy_amount / self.current_prices[i]
                    total_cost = (self.shares[i] * self.avg_cost[i]) + buy_amount
                    self.shares[i] += buy_shares
                    self.avg_cost[i] = total_cost / self.shares[i]
                    self.cash -= buy_amount
            # op == 1 → hold, nothing happens

        # 2. Advance Market
        prev_portfolio_value = self._get_portfolio_value()
        self.current_step += 1

        if self.stochastic:
            self._step_regime()
            self.current_prices = self._simulate_next_prices(self.current_prices)
        else:
            self.current_prices = self.df_prices.iloc[self.current_step].values.copy()

        self.price_history.append(self.current_prices.copy())
        self.regime_history.append(self.regime)

        # 3. Calculate Reward
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_value_history.append(current_portfolio_value)

        if prev_portfolio_value > 1e-4:
            reward = float((current_portfolio_value - prev_portfolio_value) / prev_portfolio_value)
        else:
            reward = 0.0

        # 4. Check Termination
        terminated = False
        truncated = False
        if not self.stochastic:
            if self.current_step >= len(self.df_prices) - 1:
                truncated = True

        if current_portfolio_value <= 0:
            terminated = True

        obs = self._get_obs()
        n = self.num_risky_assets
        self.allocation_history.append(obs[n : 2*n + 1])
        info = {
                "portfolio_value": current_portfolio_value,
                "cash": self.cash,
                "prices": obs[0:n],
                "alloc_assets": obs[n:2*n],
                "alloc_cash": obs[2*n],
                "rsi": obs[2*n + 1],
                "correlation": obs[2*n + 2],
                "port_std": obs[2*n + 3],
                "macd": obs[2*n + 4],
                "regime": int(self.regime),
            }

        return obs, float(reward), terminated, truncated, info

    def _simulate_next_prices(self, prices):
        mu, cov = self._regime_params()
        returns = self.np_random.multivariate_normal(mu, cov)
        new_prices = prices * np.exp(returns)
        return new_prices

    def _get_portfolio_value(self):
        return self.cash + np.sum(self.shares * self.current_prices)

    def _get_obs(self):
        prices = self.current_prices

        port_val = self._get_portfolio_value()
        if port_val > 0:
            alloc_assets = (self.shares * prices) / port_val
            alloc_cash = self.cash / port_val
        else:
            alloc_assets = np.zeros(self.num_risky_assets)
            alloc_cash = 1.0

        hist_prices = np.array(self.price_history)

        port_vals = np.array(self.portfolio_value_history)

        rsi = self._calc_rsi(port_vals, self.rsi_window)

        macd_hist = self._calc_macd(
            port_vals, self.macd_fast, self.macd_slow, self.macd_signal,
        )
        macd_hist_norm = macd_hist / port_val if port_val > 0 else 0.0

        corr = self._calc_avg_correlation(hist_prices, self.corr_window)

        port_vals = np.array(self.portfolio_value_history[-self.vol_window - 1:])
        if len(port_vals) > 1:
            port_returns = np.diff(port_vals) / port_vals[:-1]
            port_std = np.std(port_returns)
        else:
            port_std = 0.0

        obs = np.concatenate([
            prices,
            alloc_assets,
            [alloc_cash],
            [rsi],
            [corr],
            [port_std],
            [macd_hist_norm],
        ])

        return obs.astype(np.float32)

    def _calc_rsi(self, prices, window):
        if len(prices) < window + 1:
            return 50.0
        deltas = np.diff(prices[-window-1:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0 and avg_gain == 0:
            return 50.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _calc_macd(self, series, fast: int, slow: int, signal_period: int) -> float:
        """Return the MACD histogram (MACD line - signal line) for ``series``.

        Uses the recursive EMA definition
            ema_t = alpha * x_t + (1 - alpha) * ema_{t-1}
        with ``alpha = 2 / (span + 1)``. ``series[0]`` seeds both EMAs.

        A single pass maintains ``ema_fast``, ``ema_slow`` and
        ``signal_ema`` (EMA of the running MACD line). Returns 0.0 when the
        series is shorter than ``slow + signal_period + 1`` — i.e. before
        the EMA chain has had enough history to be meaningful.
        """
        min_len = slow + signal_period + 1
        if len(series) < min_len:
            return 0.0
        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)
        alpha_signal = 2.0 / (signal_period + 1)
        ema_fast = float(series[0])
        ema_slow = float(series[0])
        signal_ema = 0.0
        for i in range(1, len(series)):
            x = float(series[i])
            ema_fast = alpha_fast * x + (1.0 - alpha_fast) * ema_fast
            ema_slow = alpha_slow * x + (1.0 - alpha_slow) * ema_slow
            macd = ema_fast - ema_slow
            if i == 1:
                signal_ema = macd
            else:
                signal_ema = alpha_signal * macd + (1.0 - alpha_signal) * signal_ema
        return (ema_fast - ema_slow) - signal_ema

    def _calc_avg_correlation(self, prices, window):
        if len(prices) < window + 1:
            return 0.0
        recent_prices = prices[-window-1:]
        returns = np.diff(recent_prices, axis=0) / recent_prices[:-1]
        corr_matrix = np.corrcoef(returns, rowvar=False)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        if len(upper_tri) == 0 or np.isnan(upper_tri).all():
            return 0.0
        return np.nanmean(upper_tri)

    def render(self, benchmark_asset: str = None):
        
        if len(self.portfolio_value_history) < 2:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot Portfolio Value
        port_vals = self.portfolio_value_history[self.warmup_steps:]
        ax1.plot(port_vals, label="Total Portfolio Value", color="black", lw=3)
        
        # Plot Individual Assets
        prices = np.array(self.price_history[self.warmup_steps:])
        initial_val = port_vals[0]
        normalized_prices = (prices / prices[0]) * initial_val
        
        for i in range(self.num_risky_assets):
            if benchmark_asset and benchmark_asset not in self.asset_names[i]:
                continue
            ax1.plot(normalized_prices[:, i], label=f"{self.asset_names[i]} (Buy & Hold)", linestyle="--", alpha=0.7)
            
        if benchmark_asset:
            ax1.set_title(f"Portfolio vs {benchmark_asset} (Normalized) - {'Stochastic' if self.stochastic else 'Historical'}")
        else:
            ax1.set_title(f"Portfolio vs Individual Assets (Normalized) - {'Stochastic' if self.stochastic else 'Historical'}")
        ax1.set_ylabel("Value ($)")
        ax1.grid(True)
        ax1.legend(loc="upper left")
        
        # Plot Allocations
        allocs = np.array(self.allocation_history) 
        steps = np.arange(len(allocs))
        
        labels = self.asset_names + ["Cash"]
        
        ax2.stackplot(steps, *[allocs[:, i] for i in range(self.num_risky_assets + 1)],
                      labels=labels, alpha=0.8)
        ax2.set_title("Asset Allocation % over Time")
        ax2.set_ylabel("Allocation Ratio")
        ax2.set_xlabel("Steps")
        ax2.set_ylim(0, 1.0)
        ax2.grid(True)
        ax2.legend(loc="upper left")
        
        plt.tight_layout()
        
        try:
            clear_output(wait=True)
            display(fig)
            plt.close(fig)
        except Exception:
            plt.show()