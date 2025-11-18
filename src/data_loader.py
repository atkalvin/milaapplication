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

def load_movie_titles(path: str | Path) -> pd.DataFrame:
    """
    Load MovieLens movie metadata from the u.item file.
    Only keep movie_id and title.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=[
            "movie_id", "title", "release_date", "video_release_date",
            "imdb_url", "genre_unknown", "genre_action", "genre_adventure",
            "genre_animation", "genre_children", "genre_comedy", "genre_crime",
            "genre_documentary", "genre_drama", "genre_fantasy", "genre_film_noir",
            "genre_horror", "genre_musical", "genre_mystery", "genre_romance",
            "genre_sci_fi", "genre_thriller", "genre_war", "genre_western"
        ],
        encoding="latin-1"
    )

    return df[["movie_id", "title"]]