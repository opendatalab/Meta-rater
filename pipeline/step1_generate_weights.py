import argparse
import json
import numpy as np


def generate_dirichlet_weights(num_raters: int,
                               num_samples: int,
                               strength: float,
                               seed: int = 42) -> np.ndarray:
    """
    Generate weight vectors using a symmetric Dirichlet prior.

    - num_raters: number of features (dimensions)
    - num_samples: number of weight vectors to generate
    - strength: Dirichlet concentration (alpha); larger -> more uniform
    - seed: RNG seed
    """
    rng = np.random.default_rng(seed)
    alphas = np.full(shape=(num_raters,), fill_value=strength, dtype=np.float64)
    samples = rng.dirichlet(alphas, size=num_samples)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate N weight vectors via Dirichlet sampling")
    parser.add_argument("--num_raters", type=int, default=25, help="Number of raters (for Meta-rater, we used 25 raters but this can be further extended.)")
    parser.add_argument("--num_samples", type=int, default=256, help="Number of weight vectors to generate")
    parser.add_argument("--strength", type=float, default=1.0, help="Dirichlet concentration alpha (symmetric)")
    parser.add_argument("--seed", type=int, default=1048, help="Random seed")
    parser.add_argument("--output", type=str, default="weights.json", help="Output path for generated weights (JSON)")

    args = parser.parse_args()

    weights = generate_dirichlet_weights(
        num_raters=args.num_raters,
        num_samples=args.num_samples,
        strength=args.strength,
        seed=args.seed,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"weights": weights.tolist()}, f)

    print(f"Saved {args.num_samples} weight vectors to {args.output}")


if __name__ == "__main__":
    main()