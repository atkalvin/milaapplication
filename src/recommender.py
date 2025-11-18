import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class KNNRecommender:
    """
    Simple user-based collaborative filtering recommender
    using cosine similarity between users.
    """

    def __init__(self):
        self.user_item_matrix: pd.DataFrame | None = None
        self.similarity_matrix: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame) -> "KNNRecommender":
        """
        Fit the recommender on a ratings DataFrame.

        Parameters
        ----------
        ratings : pd.DataFrame
            DataFrame with columns: user_id, movie_id, rating.

        Returns
        -------
        self : KNNRecommender
        """
        # Build user-item matrix (rows = users, columns = movies)
        self.user_item_matrix = ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        ).fillna(0.0)

        # Compute cosine similarity between users
        self.similarity_matrix = cosine_similarity(
            self.user_item_matrix.values
        )

        return self

    def recommend(self, user_id: int, top_k: int = 5) -> list[int]:
        """
        Recommend top_k items for a given user.

        Parameters
        ----------
        user_id : int
            ID of the user to recommend for.
        top_k : int, default=5
            Number of recommendations.

        Returns
        -------
        list[int]
            List of movie IDs recommended.
        """
        if self.user_item_matrix is None or self.similarity_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if user_id not in self.user_item_matrix.index:
            raise ValueError(f"Unknown user_id: {user_id}")

        # Get the index of the user in the matrix
        user_idx = self.user_item_matrix.index.get_loc(user_id)

        # Similarity scores between this user and all others
        sim_scores = self.similarity_matrix[user_idx]

        # Matrix of ratings (numpy)
        ratings_matrix = self.user_item_matrix.values

        # Weighted sum of ratings across all users
        weighted_ratings = sim_scores @ ratings_matrix

        # Movies already seen by the user
        user_ratings = self.user_item_matrix.iloc[user_idx]
        already_rated = user_ratings > 0

        # Prevent recommending already watched movies
        mask = np.array(already_rated.values, dtype=bool)
        weighted_ratings[mask] = -np.inf

        # Get indices of the top recommendations
        top_indices = np.argsort(weighted_ratings)[::-1][:top_k]

        # Map indices back to movie IDs
        movie_ids = self.user_item_matrix.columns[top_indices]

        return movie_ids.astype(int).tolist()