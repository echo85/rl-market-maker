import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MarketCharts:
    @staticmethod
    def aggregate_npz(data_list, key):
        arr = np.array([d[key] for d in data_list])
        return arr.mean(axis=0), arr.std(axis=0)

    @staticmethod
    def plot_normalized_prices(df_normalized):
        df_normalized.plot(title='Normalized Historical Prices (Log Scale)', figsize=(10, 5))
        plt.yscale('log')
        plt.ylabel('Normalized Value (Log)')
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.show()

    @staticmethod
    def plot_market_env_history(history_portfolio, history_prices, history_rsi, history_corr, history_port_std, history_macd):
        fig, axs = plt.subplots(3, 2, figsize=(12, 5), sharex=True)
        axs = axs.flatten()
        
        axs[0].plot(history_portfolio, color='black')
        axs[0].set_title('Portfolio Value')
        axs[0].grid(True)
        
        vwce_prices = [p[2] for p in history_prices]
        axs[1].plot(vwce_prices, color='orange')
        axs[1].set_title('VWCE Price')
        axs[1].grid(True)
        
        axs[2].plot(history_rsi, color='blue')
        axs[2].set_title('RSI (14 periods)')
        axs[2].grid(True)
        
        axs[3].plot(history_corr, color='purple')
        axs[3].set_title('Avg Pairwise Correlation')
        axs[3].grid(True)
        
        axs[4].plot(history_port_std, color='red')
        axs[4].set_title('Portfolio Volatility (Std)')
        axs[4].grid(True)
        
        axs[5].plot(history_macd, color='green')
        axs[5].set_title('MACD')
        axs[5].grid(True)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_training_trajectories(feature_sets_experiments, result_dir, colors=None):
        import glob
        from experiments.feature_configs import _safe_name
        
        final_evals = {}
        is_dict = isinstance(feature_sets_experiments, dict)
        iterable = feature_sets_experiments.items() if is_dict else feature_sets_experiments

        for item in iterable:
            if is_dict:
                label, config_info = item
                color = config_info.get("color", "tab:gray")
            else:
                config_info = item
                label = config_info[0]
                color = colors.get(label, "tab:gray") if colors else "tab:gray"
            
            safe_label = _safe_name(label)
            files = sorted(glob.glob(f"results/histories/{result_dir}/{safe_label}__seed*.npz"))
            
            if not files:
                print(f"No history files found for {label} in {result_dir}!")
                continue
                
            runs_data = [np.load(f) for f in files]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"{label}", fontsize=14, y=1.05)
            
            m, s = MarketCharts.aggregate_npz(runs_data, "episode_returns")
            x = np.arange(1, len(m) + 1)
            
            axes[0].plot(x, m, color="silver", linewidth=1, label="per-episode", zorder=1)
            w = 20
            if len(m) >= w:
                moving_avg = np.convolve(m, np.ones(w)/w, mode="valid")
                x_ma = x[w-1:]
                axes[0].plot(x_ma, moving_avg, color=color, linewidth=2.5, 
                             label=f"moving avg (w={w})", zorder=2)
            else:
                axes[0].plot(x, m, color=color, linewidth=2.5, label="Mean", zorder=2)
                
            axes[0].set(title="Training return", xlabel="Episode", ylabel="Return")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            
            if "eval_mean_returns" in runs_data[0]:
                m_eval, s_eval = MarketCharts.aggregate_npz(runs_data, "eval_mean_returns")
                eval_x = runs_data[0]["eval_episodes"]
                axes[1].fill_between(eval_x, m_eval - s_eval, m_eval + s_eval, alpha=0.2, color=color, label="±1 std", zorder=1)
                axes[1].plot(eval_x, m_eval, marker="o", label="eval mean", color=color, zorder=2)
                
                final_evals[label] = np.array([d["eval_mean_returns"][-1] for d in runs_data])
            else:
                axes[1].text(0.5, 0.5, "No eval data", ha="center", va="center")
                final_evals[label] = np.array([np.nan for d in runs_data])
                
            axes[1].set(title="Greedy evaluation return", xlabel="episode", ylabel="Return")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            
            m_wn, s_wn = MarketCharts.aggregate_npz(runs_data, "weight_norms")
            axes[2].fill_between(x, m_wn - s_wn, m_wn + s_wn, alpha=0.2, color=color, zorder=1)
            axes[2].plot(x, m_wn, label="Mean", color=color, zorder=2)
            axes[2].set(title="Weight norm ‖W‖", xlabel="Episode", ylabel="‖W‖")
            axes[2].grid(alpha=0.3)
            axes[2].legend()
            
            plt.tight_layout()
            plt.show()
            
        return final_evals

    @staticmethod
    def plot_final_performance(final_evals, colors=None):
        labels = list(final_evals.keys())
        if labels:
            means  = [np.nanmean(final_evals[l]) for l in labels]
            stds   = [np.nanstd(final_evals[l])  for l in labels]
            
            plt.figure(figsize=(12, 4))
            bar_colors = []
            for l in labels:
                if isinstance(colors, dict):
                    val = colors.get(l)
                    if isinstance(val, dict):
                        bar_colors.append(val.get('color', 'tab:gray'))
                    else:
                        bar_colors.append(colors.get(l, "tab:gray"))
                else:
                    bar_colors.append("tab:gray")
                    
            plt.bar(labels, means, yerr=stds, capsize=8,
                    color=bar_colors, alpha=0.8)
            plt.ylabel("Final greedy eval return")
            plt.title("Final performance across seeds (Stochastic Regime)")
            plt.grid(alpha=0.3, axis="y")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_value_function_heatmaps(agents, base_state, MACD_IDX, PORT_STD_IDX, hold_action=(1, 1, 1)):
        macd_vals = np.linspace(-0.015, 0.015, 100)
        port_std_vals = np.linspace(0, 0.03, 100)
        MACD, PORT_STD = np.meshgrid(macd_vals, port_std_vals)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Value Function Structure: Q(s, Hold) across MACD and port_std", fontsize=16)
        
        for ax, (name, agent) in zip(axes, agents.items()):
            Q_surface = np.zeros_like(MACD)
            for i in range(MACD.shape[0]):
                for j in range(MACD.shape[1]):
                    s = base_state.copy()
                    s[MACD_IDX] = MACD[i, j]
                    s[PORT_STD_IDX] = PORT_STD[i, j]
                    x = agent.feature_extractor(s)
                    Q_surface[i, j] = np.dot(agent.W[hold_action], x)
                    
            contour = ax.contourf(MACD, PORT_STD, Q_surface, levels=30, cmap='viridis')
            fig.colorbar(contour, ax=ax)
            ax.set_title(name)
            ax.set_xlabel("MACD")
            ax.set_ylabel("Portfolio Std")
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_weight_sparsity(agents, hold_action=(1, 1, 1)):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(np.abs(agents["Tile Coding"].W[hold_action]), bins=20, log=True)
        plt.title("Tile Coding Weights (|w|)")
        
        plt.subplot(1, 3, 2)
        plt.hist(np.abs(agents["RBF Features"].W[hold_action]), bins=20, log=True)
        plt.title("RBF Weights (|w|)")
        
        plt.subplot(1, 3, 3)
        plt.bar(range(len(agents["Polynomial (Deg 2)"].W[hold_action])), np.abs(agents["Polynomial (Deg 2)"].W[hold_action]))
        plt.title("Polynomial Weights (|w|)")
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def decode_action(a):
        syms = ["Sell", "Hold", "Buy"]
        if isinstance(a, (int, np.integer)):
            op0 = a % 3; a1 = a // 3; op1 = a1 % 3; op2 = a1 // 3
        else:
            op0, op1, op2 = a
        return f"{syms[op0]} / {syms[op1]} / {syms[op2]}"

    @staticmethod
    def plot_state_visitation_and_actions(feature_sets_experiments, result_dir, 
                                          make_env_stochastic, LinearSARSAAgent, 
                                          TileCoder, RBFFeatures, PolynomialFeatures,
                                          TC_N_TILINGS, RBF_SIGMA, MASTER_SEED):
        import glob
        from experiments.feature_configs import _safe_name
        
        RSI_IDX = 7
        PORT_STD_IDX = 9
        HOLD_ACTION = (1, 1, 1) 
        N_EPISODES_ROLLOUT = 10
        MAX_STEPS_R = 250
        
        is_dict = isinstance(feature_sets_experiments, dict)
        iterable = feature_sets_experiments.items() if is_dict else feature_sets_experiments
        
        n_experiments = len(feature_sets_experiments)
        cols = 2
        rows = (n_experiments + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes_flat = np.atleast_1d(axes).flatten()
        
        summary_rows = []
        action_freqs_per_agent = {}
        
        for col, item in enumerate(iterable):
            if is_dict:
                label, config_info = item
                feat = config_info['features']
            else:
                config_info = item
                label = config_info[0]
                feat = config_info[1]
                
            safe_label = _safe_name(label)
            files = sorted(glob.glob(f"results/histories/{result_dir}/{safe_label}__seed*.npz"))
            
            if not files:
                print(f"Skipping {label}, no files found.")
                continue
                
            runs_data = [np.load(f) for f in files]
            if "weights" not in runs_data[0]:
                print(f"Skipping {label}, no 'weights' array found in the npz files.")
                continue
                
            seed_idx = 0 
            s = seed_idx
            
            if isinstance(feat, TileCoder):
                f = TileCoder(low=feat.low, high=feat.high, n_tiles=feat.n_tiles, n_tilings=TC_N_TILINGS, feature_indices=feat.feature_indices, seed=s)
            elif isinstance(feat, RBFFeatures):
                f = RBFFeatures(low=feat.low, high=feat.high, n_centers=feat.n_centers, sigma=RBF_SIGMA, normalize=True, feature_indices=feat.feature_indices)
            elif isinstance(feat, PolynomialFeatures):
                f = PolynomialFeatures(state_dim=11, degree=2, feature_indices=feat.feature_indices, low=feat.low, high=feat.high)
            else:
                from utils.features import RawRepresentation
                f = RawRepresentation(state_dim=11, feature_indices=feat.feature_indices, low=feat.low, high=feat.high)
                
            env = make_env_stochastic()
            agent = LinearSARSAAgent(action_space=env.action_space, feature_extractor=f, alpha=0.01, seed=s)
            agent.W = runs_data[seed_idx]["weights"]
            agent.epsilon = 0.0
            
            states_visited, actions_taken = [], []
            for ep in range(N_EPISODES_ROLLOUT):
                obs, _ = env.reset(seed=MASTER_SEED + ep)
                for _ in range(MAX_STEPS_R):
                    a = agent.select_action(obs, greedy=True)
                    states_visited.append(obs.copy())
                    actions_taken.append(tuple(a) if isinstance(a, (list, np.ndarray)) else a)
                    obs, _, term, trunc, _ = env.step(a)
                    if term or trunc: break
                    
            states_visited = np.asarray(states_visited)
            actions_taken = np.asarray(actions_taken)
            
            rsi_vals = states_visited[:, RSI_IDX]
            pstd_vals = states_visited[:, PORT_STD_IDX]
            pstd_top = max(float(pstd_vals.max()), 1e-3)
            
            ax = axes_flat[col]
            hb = ax.hexbin(rsi_vals, pstd_vals, gridsize=25, cmap='Blues', bins='log', mincnt=1)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, pstd_top)
            ax.set_title(f'{label}\n(Seed {seed_idx})', fontsize=12)
            fig.colorbar(hb, ax=ax, label='log(visits)')
            
            unique_actions, action_counts = np.unique(actions_taken, axis=0, return_counts=True)
            sorted_idx = np.argsort(-action_counts)
            action_freqs_per_agent[label] = [(tuple(unique_actions[i]) if isinstance(unique_actions[i], np.ndarray) else unique_actions[i], action_counts[i]) for i in sorted_idx]
            n_hold = int(np.sum([np.array_equal(a, HOLD_ACTION) for a in actions_taken]))
            summary_rows.append((label, len(states_visited), n_hold, 100.0 * n_hold / len(actions_taken)))
            
        for i in range(len(feature_sets_experiments), len(axes_flat)):
            fig.delaxes(axes_flat[i])
            
        plt.tight_layout()
        plt.show()
        
        print("\n=== Top 5 Most Used Actions per Agent ===")
        for name, freqs in action_freqs_per_agent.items():
            print(f"\nAgent: {name}")
            print(f"{'Action ID':<12}{'Description (Asset 0/1/2)':<30}{'Count':<10}{'% of Total':<10}")
            print("-" * 65)
            total_actions = sum(count for _, count in freqs)
            for a, count in freqs[:5]:
                desc = MarketCharts.decode_action(a)
                pct = 100.0 * count / total_actions
                if isinstance(a, tuple):
                    a_str = "(" + ", ".join(str(int(x)) for x in a) + ")"
                else:
                    a_str = str(a)
                print(f"{a_str:<12}{desc:<30}{count:<10}{pct:>5.1f}%")

    @staticmethod
    def plot_feature_sweep_results(df):
        from experiments.feature_configs import summarize
        from IPython.display import display
        
        print(f"Loaded {len(df)} runs across {df['config'].nunique()} configs")
        display(summarize(df, mode="screen") if (df["mode"] == "screen").any() else summarize(df))
        
        if not df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            summary = (df.groupby("config")["eval_return"]
                         .agg(["mean", "std", "count"])
                         .sort_values("mean", ascending=False))
            axes[0].barh(summary.index, summary["mean"], xerr=summary["std"].fillna(0), color="steelblue")
            axes[0].set_xlabel("Mean eval return")
            axes[0].set_title("Feature sets ranked by eval return")
            axes[0].invert_yaxis()
            axes[0].grid(alpha=0.3, axis="x")
            
            nf = (df.drop_duplicates("config").set_index("config")["n_features"].reindex(summary.index))
            axes[1].scatter(nf, summary["mean"], s=80, color="darkorange")
            for name, x, y in zip(summary.index, nf, summary["mean"]):
                axes[1].annotate(name, (x, y), fontsize=7, xytext=(5, 5), textcoords="offset points")
            axes[1].set_xscale("log")
            axes[1].set_xlabel("n_features (log)")
            axes[1].set_ylabel("Mean eval return")
            axes[1].set_title("Expressiveness vs performance")
            axes[1].grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
