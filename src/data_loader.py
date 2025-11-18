import pandas as pd
from pathlib import Path

def load_movielens_ratings(path: str | Path) -> pd.DataFrame:
    """
    Load the MovieLens 100k ratings dataset from a tab-separated file.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python"
    )

    df = df[["user_id", "movie_id", "rating"]]

    df["user_id"] = df["user_id"].astype(int)
    df["movie_id"] = df["movie_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    return df