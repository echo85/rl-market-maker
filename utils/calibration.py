import numpy as np

class FeatureCalibration:
    @staticmethod
    def calibrate_feature_ranges(make_env_fn, episodes=5, max_steps=500, seed=0, feature_indices=(6, 7, 9, 10)):
        _cal_rng = np.random.default_rng(seed)
        _cal_obs = []
        for _ in range(episodes):
            _env = make_env_fn()
            _o, _ = _env.reset(seed=int(_cal_rng.integers(1 << 31)))
            for _ in range(max_steps):
                _a = int(_cal_rng.integers(_env.action_space.n))
                _o, _, _term, _trunc, _ = _env.step(_a)
                _cal_obs.append(_o.copy())
                if _term or _trunc:
                    break
        _cal_arr = np.asarray(_cal_obs)
        _cal_names = ['p0', 'p1', 'p2', 'a0', 'a1', 'a2', 'cash', 'rsi', 
                      'corr',  'vol', 'macd']
        print(f'{"idx":>3}  {"name":<8}  {"min":>10}  {"max":>10}  {"p1":>10}  {"p99":>10}')
        for _i in feature_indices:
            _s = _cal_arr[:, _i]
            print(f'{_i:>3}  {_cal_names[_i]:<8}  {_s.min():>10.4f}  '
                  f'{_s.max():>10.4f}  {np.percentile(_s,1):>10.4f}  '
                  f'{np.percentile(_s,99):>10.4f}')
