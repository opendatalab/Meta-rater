import argparse
import json
import os
from typing import Iterable, List
import numpy as np

def iter_jsonl(paths: Iterable[str]):
    for p in paths:
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)

def write_jsonl(path: str, items: Iterable[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_default_feature_order() -> List[str]:
    here = os.path.dirname(__file__)
    feature_order_path = os.path.join(here, "feature_order.json")
    if not os.path.exists(feature_order_path):
        raise FileNotFoundError(
            f"feature_order.json not found at {feature_order_path}. Please create a JSON array of feature names.")
    with open(feature_order_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("feature_order.json must be a JSON array of strings")
    return [str(x) for x in data]

def compute_feature_statistics(items: List[dict], feature_order: List[str]) -> dict:
    modernbert_features = ['modernbert_professionalism', 'modernbert_reasoning',
                           'modernbert_readability', 'modernbert_cleanliness']
    binary_features = ['ad_en', 'fluency_en']
    stats: dict = {}
    for feat in feature_order:
        if feat in modernbert_features or feat in binary_features:
            # These are categorical/argmax-based; we do not compute min/max normalization stats
            continue
        values: List[float] = []
        for it in items:
            if feat in it:
                try:
                    val = it[feat]
                    if isinstance(val, (list, tuple)):
                        # Default behavior: mean for list-like numeric features
                        numeric = [float(x) for x in val]
                        if len(numeric) == 0:
                            continue
                        values.append(float(np.mean(numeric)))
                    else:
                        values.append(float(val))
                except Exception:
                    # Skip non-numeric values silently
                    continue
        if len(values) > 0:
            min_v = float(np.min(values))
            max_v = float(np.max(values))
            stats[feat] = {"min": min_v, "max": max_v}
    return stats


def compute_weighted_score(item: dict, feature_order: List[str], statistics: dict, weights_vec: np.ndarray) -> float:
    modernbert_features = ['modernbert_professionalism', 'modernbert_reasoning',
                           'modernbert_readability', 'modernbert_cleanliness']
    binary_features = ['ad_en', 'fluency_en']
    score = 0.0
    for idx, feat in enumerate(feature_order):
        if feat in modernbert_features or feat in binary_features:
            preds = item.get(feat, [])
            try:
                k = len(preds) if isinstance(preds, (list, tuple)) else 0
                max_idx = int(np.argmax(preds)) if k > 0 else 0
            except Exception:
                k = 0
                max_idx = 0
            # Scale categorical index to [0, 5]
            if k > 1:
                scaled = (max_idx / (k - 1)) * 5.0
            else:
                # If malformed, default to 0
                scaled = 0.0
            score += weights_vec[idx] * float(scaled)
        else:
            # Robustly extract numeric value
            val = item.get(feat, None)
            if isinstance(val, (list, tuple)):
                # Aggregate list-like numeric features by mean for scoring
                try:
                    nums = [float(x) for x in val]
                    raw = float(np.mean(nums)) if len(nums) > 0 else 0.0
                except Exception:
                    raw = 0.0
            else:
                try:
                    raw = float(val)
                except Exception:
                    raw = 0.0

            # Normalize with pre-computed statistics if available
            if feat in statistics:
                min_val = statistics[feat]['min']
                max_val = statistics[feat]['max']
            else:
                # If no stats (e.g., all values non-numeric), treat as zero-centered
                min_val = 0.0
                max_val = 0.0
            if max_val == min_val:
                norm = 0.0
            else:
                norm = (raw - min_val) / (max_val - min_val) * 5
            score += weights_vec[idx] * norm
    return float(score)


def build_weights_vector(weights_spec: str, feature_order: List[str]) -> np.ndarray:
    """Build weight vector aligned with feature_order from a user specification string.

    Supported formats:
    - Numeric list: "0.1,0.2,0.3" (length must equal len(feature_order))
    - Named weights: "featA=1.0,featB=0.5" or "featA:1.0,featB:0.5"
    - Named list (default weight 1.0): "featA,featB" (others default to 0.0)
    """
    # Try numeric list first
    try:
        numeric_values = [float(x) for x in weights_spec.split(',') if x.strip() != '']
        if len(numeric_values) == len(feature_order):
            return np.array(numeric_values, dtype=np.float64)
        # If it parses but the length mismatches, prefer an explicit error
        # to avoid silently misaligning.
        if len(numeric_values) > 0:
            raise ValueError(
                f"Numeric weights_vec has length {len(numeric_values)}, but feature_order has length {len(feature_order)}"
            )
    except ValueError:
        # Fall through to named formats
        pass

    # Parse named specs
    weights_map = {}
    tokens = [t.strip() for t in weights_spec.split(',') if t.strip() != '']
    for tok in tokens:
        if '=' in tok:
            name, val_str = tok.split('=', 1)
            name = name.strip()
            val = float(val_str.strip())
        elif ':' in tok:
            name, val_str = tok.split(':', 1)
            name = name.strip()
            val = float(val_str.strip())
        else:
            name = tok
            val = 1.0
        if name in feature_order:
            weights_map[name] = val
        elif name == 'qurater':
            # If user specifies generic 'qurater' and the order is expanded, distribute evenly
            parts = [f'qurater_{i}' for i in range(4)]
            present = [p for p in parts if p in feature_order]
            if present:
                each = val / len(present)
                for p in present:
                    weights_map[p] = each
        # Unknown features are ignored silently

    vector = np.array([weights_map.get(feat, 0.0) for feat in feature_order], dtype=np.float64)
    return vector


def expand_feature_order(base_order: List[str]) -> List[str]:
    expanded: List[str] = []
    for name in base_order:
        if name == 'qurater':
            expanded.extend([f'qurater_{i}' for i in range(4)])
        else:
            expanded.append(name)
    return expanded


def expand_item_composites(item: dict) -> None:
    # Expand 'qurater' list into four scalar fields if present
    if 'qurater' in item:
        try:
            seq = item['qurater']
            for i in range(min(4, len(seq))):
                item[f'qurater_{i}'] = seq[i]
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Step 2: Read JSONL(s), compute Top15% threshold, filter and output JSONL for selected data")
    parser.add_argument("--inputs", type=str, nargs='+', required=True, help="One or more JSONL files to read")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path for retained items")
    parser.add_argument("--percentile", type=float, default=85.0, help="Percentile for threshold (e.g., 85 for Top15%)")
    # Mode A: use a direct numeric key present in JSONL
    parser.add_argument("--score_key", type=str, default=None, help="Existing numeric field used as score, if you have processed all scores beforehand(e.g., final_score)")
    # Mode B: compute weighted score from feature keys and weights
    parser.add_argument("--weights_vec", type=str, default=None, help="Comma-separated weights to combine feature scores")
    parser.add_argument("--feature_order", type=str, nargs='*', default=None, help="Feature names order matching weights_vec; default uses project order")
    # Mode C: load weights from a JSON file containing {"weights": [[...], [...], ...]} and select by index
    parser.add_argument("--weights_json", type=str, default=None, help="Path to weights.json containing a top-level 'weights' array of arrays")
    parser.add_argument("--weights_index", type=int, default=None, help="Index into weights list to select a specific vector")
    args = parser.parse_args()

    # Load items
    items = list(iter_jsonl(args.inputs))
    if len(items) == 0:
        print("No items found in inputs.")
        write_jsonl(args.output, [])
        return

    # Load DEFAULT_FEATURE_ORDER from fixed file
    try:
        DEFAULT_FEATURE_ORDER = load_default_feature_order()
    except Exception as e:
        raise

    # Expand feature order (e.g., split 'qurater' into 4 parts) for alignment with weights
    EXPANDED_FEATURE_ORDER = expand_feature_order(DEFAULT_FEATURE_ORDER)

    # Compute statistics by traversing items and save to statistics.json
    # Expand composite fields on items before computing statistics
    for it in items:
        expand_item_composites(it)
    statistics = compute_feature_statistics(items, EXPANDED_FEATURE_ORDER)
    here = os.path.dirname(__file__)
    statistics_path = os.path.join(here, 'statistics.json')
    with open(statistics_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)

    # Determine scoring
    scores = []
    if args.score_key is not None and all(args.score_key in it for it in items):
        scores = [float(it[args.score_key]) for it in items]
    else:
        # Decide weights source: JSON+index overrides weights_vec if provided
        feature_order = expand_feature_order(args.feature_order) if (args.feature_order is not None and len(args.feature_order) > 0) else EXPANDED_FEATURE_ORDER

        weights_vec: np.ndarray
        if args.weights_json is not None and args.weights_index is not None:
            with open(args.weights_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'weights' not in data or not isinstance(data['weights'], list):
                raise ValueError("weights_json must contain a top-level 'weights' list")
            if args.weights_index < 0 or args.weights_index >= len(data['weights']):
                raise IndexError(f"weights_index {args.weights_index} out of range (len={len(data['weights'])})")
            raw_vec = data['weights'][args.weights_index]
            if not isinstance(raw_vec, list):
                raise ValueError("Selected weights entry must be a list of numbers")
            weights_vec = np.array([float(x) for x in raw_vec], dtype=np.float64)
            if len(weights_vec) != len(feature_order):
                raise ValueError("weights_json vector length must equal feature_order length")
        elif args.weights_vec is not None:
            weights_vec = build_weights_vector(args.weights_vec, feature_order)
        else:
            raise ValueError("Cannot determine score: provide --score_key or (--weights_json and --weights_index) or --weights_vec")

        for it in items:
            scores.append(compute_weighted_score(it, feature_order, statistics, weights_vec))

    # Compute threshold and filter
    thresh = float(np.percentile(np.array(scores, dtype=np.float64), args.percentile))
    kept = [it for it, s in zip(items, scores) if s >= thresh]

    # Write output JSONL
    write_jsonl(args.output, kept)

    # Also emit a small sidecar JSON for records
    side = {
        "total": len(items),
        "kept": len(kept),
        "percentile": args.percentile,
        "threshold": thresh,
    }
    side_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(side_path, 'w', encoding='utf-8') as f:
        json.dump(side, f, ensure_ascii=False, indent=2)

    print(f"Kept {len(kept)}/{len(items)} items (threshold={thresh:.6f}) -> {args.output}")


if __name__ == "__main__":
    main()