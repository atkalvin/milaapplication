import numpy as np
import pandas as pd


class SVDRecommender:
    """
    Matrix Factorization model using SVD.
    More powerful than simple KNN baseline.
    """

    def __init__(self, n_components: int = 20):
        self.n_components = n_components

        # These will be set in fit()
        self.user_factors = None           # type: np.ndarray | None
        self.item_factors = None           # type: np.ndarray | None
        self.user_mapping: dict[int, int] = {}
        self.item_mapping: dict[int, int] = {}
        self.user_item_matrix: pd.DataFrame | None = None

    def fit(self, ratings: pd.DataFrame) -> "SVDRecommender":
        """
        Fit the SVD model on ratings DataFrame
        with columns: user_id, movie_id, rating.
        """

        # 1. Create user-item matrix
        self.user_item_matrix = ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        ).fillna(0.0)

        # 2. Create index → original ID mapping
        self.user_mapping = {
            i: int(uid) for i, uid in enumerate(self.user_item_matrix.index)
        }
        self.item_mapping = {
            i: int(mid) for i, mid in enumerate(self.user_item_matrix.columns)
        }

        # 3. Convert to dense matrix
        R = self.user_item_matrix.values

        # 4. Compute SVD
        U, sigma, Vt = np.linalg.svd(R, full_matrices=False)

        # 5. Reduce rank
        k = min(self.n_components, len(sigma))
        U_reduced = U[:, :k]
        sigma_reduced = np.diag(sigma[:k])
        Vt_reduced = Vt[:k, :]

        # 6. Latent factors
        self.user_factors = U_reduced @ sigma_reduced   # shape: (n_users, k)
        self.item_factors = Vt_reduced                  # shape: (k, n_items)

        return self

    def recommend(self, user_id: int, top_k: int = 5) -> list[int]:
        """
        Recommend top-k movie IDs for the given user.
        """

        if (
            self.user_item_matrix is None
            or self.user_factors is None
            or self.item_factors is None
        ):
            raise RuntimeError("Model not fitted. Call fit() first.")

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"Unknown user_id: {user_id}")

        # Internal index for user
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Predict scores: user_factors[user] dot item_factors
        user_vec = self.user_factors[user_idx]          # (k,)
        scores = user_vec @ self.item_factors           # (n_items,)

        user_ratings = self.user_item_matrix.iloc[user_idx]

        # Force conversion to numpy array (fixes Pylance warnings)
        ratings_np: np.ndarray = np.asarray(user_ratings.to_numpy(dtype=float), dtype=float)

        seen_mask = ratings_np > 0
        scores[seen_mask] = -np.inf

        # Top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Map back to movie IDs
        movie_ids = [self.item_mapping[i] for i in top_indices]

        return movie_ids