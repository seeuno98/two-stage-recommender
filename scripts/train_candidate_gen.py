"""Entrypoint for candidate generation model training."""

from src.candidate_gen.popularity import PopularityRecommender
from src.data.load_data import load_interactions


def main() -> None:
    """Train a simple popularity baseline from interaction data."""
    interactions = load_interactions("data/raw/interactions.csv")
    model = PopularityRecommender()
    model.fit(interactions)
    print(f"Trained popularity model on {len(interactions)} interactions.")


if __name__ == "__main__":
    main()
