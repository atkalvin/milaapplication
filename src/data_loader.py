import pandas as pd

def load_movielens_ratings(path: str) -> pd.DataFrame:
    """
    Load the MovieLens 100k ratings dataset.
    Expected format: user_id, movie_id, rating, timestamp
    """
    df = pd.read_csv(path, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return df