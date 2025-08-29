import argparse
import json
import numpy as np
from joblib import load

def sample_search_space(num_raters: int,
                         num_samples: int,
                         alpha: float,
                         seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alphas = np.full(num_raters, alpha, dtype=np.float64)
    return rng.dirichlet(alphas, size=num_samples)


def main():
    parser = argparse.ArgumentParser(description="Step 4: Search a large space for optimal weights using trained regressor")
    parser.add_argument("--model", type=str, default="regressor_avg_loss.joblib", help="Path to trained avg_loss regressor")
    parser.add_argument("--num_raters", type=int, default=25, help="Feature dimension (must match training)")
    parser.add_argument("--search_samples", type=int, default=1000_000, help="How many candidate weights to sample")
    parser.add_argument("--alpha", type=float, default=1.0, help="Dirichlet concentration for sampling")
    parser.add_argument("--topk", type=int, default=128, help="Average top-k best predictions to form final mixture")
    parser.add_argument("--output", type=str, default="optimal_weights.json", help="Where to save optimal weights JSON")
    args = parser.parse_args()

    reg = load(args.model)

    samples = sample_search_space(
        num_raters=args.num_raters,
        num_samples=args.search_samples,
        alpha=args.alpha,
        seed=42,
    )

    preds = reg.predict(samples)
    # lower is better (loss)
    best_indices = np.argsort(preds)[: args.topk]
    best_samples = samples[best_indices]
    optimal = best_samples.mean(axis=0)

    obj = {
        "optimal_weights": optimal.tolist(),
        "min_pred": float(np.min(preds)),
        "avg_topk_pred": float(np.mean(preds[best_indices])),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    print(f"Saved optimal weights to {args.output}")
    print(f"Min predicted loss: {obj['min_pred']:.6f}")
    print(f"Avg top-{args.topk} predicted loss: {obj['avg_topk_pred']:.6f}")


if __name__ == "__main__":
    main()