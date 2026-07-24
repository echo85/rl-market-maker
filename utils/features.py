"""Feature extractors for linear value function approximation.

Two families of feature maps for continuous N-dimensional state spaces:

- ``TileCoder``: one or more offset rectangular grids partition the state
  space into tiles; the feature vector is a binary indicator of the tile
  the state falls into, concatenated across tilings.
- ``RBFFeatures``: a regular grid of Gaussian centres; the feature vector
  is the Gaussian-evaluated similarity of the state to each centre.
- ``PolynomialFeatures``: polynomial combinations of the state components.

Both classes implement the ``FeatureExtractor`` protocol below: they expose
a public ``n_features`` attribute and a ``__call__`` method that maps a
state vector to a feature vector.
"""

from __future__ import annotations

from typing import Protocol, Optional, List, Sequence
import itertools

import numpy as np


# ---------------------------------------------------------------------------
# Structural interface
# ---------------------------------------------------------------------------

class FeatureExtractor(Protocol):
    """Structural interface for feature maps.

    Any object exposing an integer ``n_features`` and a callable that maps
    a state vector ``s`` of shape ``(state_dim,)`` to a feature vector of
    shape ``(n_features,)`` satisfies this protocol.
    """

    n_features: int

    def __call__(self, state: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Tile coding
# ---------------------------------------------------------------------------

class TileCoder:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_tiles_x: Optional[int] = None,
        n_tiles_y: Optional[int] = None,
        n_tilings: int = 4,
        seed: int = 0,
        n_tiles: Optional[List[int]] = None,
        feature_indices: Optional[List[int]] = None,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float64).copy()
        self.high = np.asarray(high, dtype=np.float64).copy()
        self.feature_indices = feature_indices
        
        if np.any(self.high <= self.low):
            raise ValueError("high must be strictly greater than low component-wise.")
        
        if n_tiles is not None:
            self.n_tiles = np.array(n_tiles, dtype=int)
        elif n_tiles_x is not None and n_tiles_y is not None:
            self.n_tiles = np.array([n_tiles_x, n_tiles_y], dtype=int)
        else:
            raise ValueError("Must provide either n_tiles or both n_tiles_x and n_tiles_y.")
            
        if np.any(self.n_tiles < 1):
            raise ValueError("All n_tiles dimensions must be positive.")
        if n_tilings < 1:
            raise ValueError("n_tilings must be positive.")

        self.n_tilings = int(n_tilings)
        self.n_dims = len(self.n_tiles)

        # Tile sizes (uniform across tilings).
        self.tile_sizes = (self.high - self.low) / self.n_tiles

        rng = np.random.default_rng(seed)
        if self.n_tilings == 1:
            self._offsets = np.zeros((1, self.n_dims), dtype=np.float64)
        else:
            base = np.linspace(0.0, 1.0, self.n_tilings, endpoint=False)
            self._offsets = np.zeros((self.n_tilings, self.n_dims), dtype=np.float64)
            for d in range(self.n_dims):
                jitter = rng.uniform(-0.05, 0.05, size=self.n_tilings)
                frac = (base + jitter) % 1.0
                # Shuffle the fractional offsets differently for each dimension
                # to ensure non-degenerate overlapping grids.
                if d > 0:
                    rng.shuffle(frac)
                self._offsets[:, d] = frac * self.tile_sizes[d]

        self.n_features_per_tiling = np.prod(self.n_tiles)
        self.n_features = self.n_tilings * self.n_features_per_tiling

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        if self.feature_indices is not None:
            s = s[self.feature_indices]
            
        feat = np.zeros(self.n_features, dtype=np.float64)
        for t in range(self.n_tilings):
            # Shift the state by the tiling offset, then bin.
            shifted_s = (s - self.low + self._offsets[t]) / self.tile_sizes
            coords = np.clip(np.floor(shifted_s), 0, self.n_tiles - 1).astype(int)
            
            # Map N-D coordinate to 1D flat index inside this tiling
            tile_index = np.ravel_multi_index(coords, self.n_tiles)
            
            # Index of this tiling's block in the overall feature vector.
            block_start = t * self.n_features_per_tiling
            feat[block_start + tile_index] = 1.0
        return feat


# ---------------------------------------------------------------------------
# Radial basis functions
# ---------------------------------------------------------------------------

class RBFFeatures:
    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        n_centers_x: Optional[int] = None,
        n_centers_y: Optional[int] = None,
        sigma: float = 1.0,
        normalize: bool = False,
        n_centers: Optional[List[int]] = None,
        feature_indices: Optional[List[int]] = None,
    ) -> None:
        self.low = np.asarray(low, dtype=np.float64).copy()
        self.high = np.asarray(high, dtype=np.float64).copy()
        self.feature_indices = feature_indices

        if np.any(self.high <= self.low):
            raise ValueError("high must be strictly greater than low component-wise.")
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        if n_centers is not None:
            self.n_centers = np.array(n_centers, dtype=int)
        elif n_centers_x is not None and n_centers_y is not None:
            self.n_centers = np.array([n_centers_x, n_centers_y], dtype=int)
        else:
            raise ValueError("Must provide either n_centers or both n_centers_x and n_centers_y.")
            
        if np.any(self.n_centers < 1):
            raise ValueError("All n_centers dimensions must be positive.")

        self.sigma = float(sigma)
        self.normalize = bool(normalize)
        self.n_dims = len(self.n_centers)

        # Place centres at the cell centres of a regular grid covering [low, high].
        axes = []
        for d in range(self.n_dims):
            axis_centers = self.low[d] + (np.arange(self.n_centers[d]) + 0.5) * \
                           (self.high[d] - self.low[d]) / self.n_centers[d]
            axes.append(axis_centers)
            
        grids = np.meshgrid(*axes, indexing="ij")
        
        # Stack and flatten the grids into a (n_features, n_dims) array of centers
        self.centers = np.stack([g.ravel() for g in grids], axis=1)
        self.n_features = np.prod(self.n_centers)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        if self.feature_indices is not None:
            s = s[self.feature_indices]
            
        s = s.reshape(self.n_dims)
        # Squared distances from state to each centre.
        feature_ranges = self.high - self.low
        d2 = np.sum(((self.centers - s[None, :]) / feature_ranges) ** 2, axis=1)
        #d2 = np.sum((self.centers - s[None, :]) ** 2, axis=1)
        feat = np.exp(-d2 / (2.0 * self.sigma ** 2))
        if self.normalize:
            total = feat.sum()
            if total > 0:
                feat = feat / total
        return feat

# ---------------------------------------------------------------------------
# Polynomial Features
# ---------------------------------------------------------------------------

class PolynomialFeatures:
    """Polynomial features up to a given degree for continuous state spaces.
    
    Includes a bias term (1.0) and all combinations of features up to the
    specified maximum degree. For instance, with 2 features (x1, x2) and degree 2:
    [1, x1, x2, x1^2, x1*x2, x2^2]
    
    Parameters
    ----------
    state_dim : int
        Dimension of the input state space (before applying feature_indices if used).
    degree : int
        Maximum polynomial degree. Default 2.
    feature_indices : list of int, optional
        Indices of the input state vector to use.
    """
    def __init__(
        self,
        state_dim: int,
        degree: int = 2,
        feature_indices: Optional[List[int]] = None,
        low: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
    ) -> None:
        self.degree = int(degree)
        self.feature_indices = feature_indices
        self.feature_dim = len(feature_indices) if feature_indices is not None else state_dim
        
        self.low = np.asarray(low, dtype=np.float64) if low is not None else None
        self.high = np.asarray(high, dtype=np.float64) if high is not None else None
        
        if self.degree < 1:
            raise ValueError("degree must be positive.")
            
        # Precompute the index combinations for the specified degree
        self.combinations = []
        for d in range(self.degree + 1):
            self.combinations.extend(itertools.combinations_with_replacement(range(self.feature_dim), d))
            
        self.n_features = len(self.combinations)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float64)
        if self.feature_indices is not None:
            s = s[self.feature_indices]
            
        s = s.reshape(self.feature_dim)
        
        if self.low is not None and self.high is not None:
            # Scale to [-1, 1] to prevent polynomial explosion
            # Add a small epsilon to denominator to prevent division by zero
            range_diff = np.maximum(self.high - self.low, 1e-8)
            s = 2.0 * (s - self.low) / range_diff - 1.0
            
        feat = np.ones(self.n_features, dtype=np.float64)
        
        for i, comb in enumerate(self.combinations):
            if len(comb) > 0:
                feat[i] = np.prod(s[list(comb)])
                
        return feat
