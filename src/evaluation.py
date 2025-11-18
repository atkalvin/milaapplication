from typing import List


def precision_at_k(recommended: List[int], relevant: List[int], k: int = 5) -> float:
    """
    Compute Precision@k for a set of recommendations.

    Parameters
    ----------
    recommended : list[int]
        The list of recommended item IDs (ordered by relevance).
    relevant : list[int]
        The list of relevant (liked) item IDs for the user.
    k : int
        The number of top items to consider.

    Returns
    -------
    float
        Precision@k score between 0 and 1.
    """
    if k == 0:
        return 0.0

    recommended_at_k = recommended[:k]
    hits = sum(1 for r in recommended_at_k if r in relevant)

    return hits / k

# Global precision @ K

def global_precision_at_k(model, df, k=2000, num_users=5000):

    import random

    scores = []
    all_users = df["user_id"].unique().tolist()
    random.shuffle(all_users)

    sampled_users = all_users[:num_users]

    for uid in sampled_users:
        user_ratings = df[df["user_id"] == uid]
        liked = user_ratings[user_ratings["rating"] >= 4]["movie_id"].tolist()

        if len(liked) < 2:
            continue

        test_movie = random.choice(liked)

        recs = model.recommend(uid, top_k=k)

        score = precision_at_k(recommended=recs, relevant=[test_movie], k=k)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0