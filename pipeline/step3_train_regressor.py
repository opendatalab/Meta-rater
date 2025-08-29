import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# You may also try other parts of validation set
LOSS_COLUMNS = [
    'avg_loss',
    'cc_loss',
    'c4_loss'
]


def read_weights_from_json(weights_path: str, keep_last: int | None = None) -> list[list[float]]:
    """Read weights from weights.json file"""
    with open(weights_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    weights_lists = data.get('weights', [])
    if keep_last is not None:
        weights_lists = weights_lists[-keep_last:]
    
    return weights_lists


def read_losses_from_json(losses_path: str) -> pd.DataFrame:
    """Read losses from losses.json file"""
    with open(losses_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    losses_data = data.get('losses', [])
    df = pd.DataFrame(losses_data)
    
    # Ensure all required columns exist
    for col in LOSS_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0  # Default value if column missing
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Step 3: Train regression model to predict val loss from weights")
    parser.add_argument("--weights_file", type=str, default="pipeline_out/weights.json", help="Path to weights.json file")
    parser.add_argument("--losses_file", type=str, default="pipeline_out/losses.json", help="Path to losses.json file")
    parser.add_argument("--keep_last", type=int, default=256, help="Keep the last N weight functions parsed from file")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--model_out", type=str, default="regressor_avg_loss.joblib", help="Output model path for avg_loss")
    parser.add_argument("--metrics_out", type=str, default="regressor_metrics.json", help="Where to store evaluation metrics")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
        from joblib import dump
    except Exception as e:
        raise RuntimeError("LightGBM and joblib are required. Please install: pip install lightgbm joblib") from e

    # Read weights and losses from JSON files
    weights_lists = read_weights_from_json(args.weights_file, keep_last=args.keep_last)
    df = read_losses_from_json(args.losses_file)
    
    # Filter to only include required loss columns
    df = df[[c for c in df.columns if c in LOSS_COLUMNS]]
    
    X = np.array(weights_lists, dtype=np.float64)

    # Align sizes (remove failures already in JSON generation)
    n = min(len(X), len(df))
    X = X[:n]
    Y = df.iloc[:n].values

    # train/test split
    train_n = int(n * args.train_ratio)
    X_train, X_test = X[:train_n], X[train_n:]
    Y_train, Y_test = Y[:train_n], Y[train_n:]

    hyper_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'num_iterations': 1000,
        'seed': 42,
        'learning_rate': 1e-2,
        'verbosity': -1,
    }

    # Train a regressor per metric
    predictors = []
    corrs = {}
    for i, col in enumerate(LOSS_COLUMNS):
        try:
            gbm = lgb.LGBMRegressor(**hyper_params)
            gbm.fit(
                X_train, Y_train[:, i],
                eval_set=[(X_test, Y_test[:, i])],
                eval_metric='l2',
                callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)],
            )
            
            # Check if test data has variance
            y_pred = gbm.predict(X_test)
            if np.var(Y_test[:, i]) > 1e-10 and np.var(y_pred) > 1e-10:
                r, _ = spearmanr(y_pred, Y_test[:, i])
                corrs[col] = float(0.0 if np.isnan(r) else r)
            else:
                print(f"Warning: {col} has no variance in test data, setting correlation to 0")
                corrs[col] = 0.0
                
            predictors.append(gbm)
        except Exception as e:
            print(f"Warning: Failed to train regressor for {col}: {e}")
            corrs[col] = 0.0
            predictors.append(None)

    # Save only avg_loss model by default
    if predictors[0] is not None:
        from joblib import dump
        dump(predictors[0], args.model_out)
        print(f"Saved avg_loss regressor to {args.model_out}")
    else:
        print("Warning: avg_loss regressor is None, skipping model save")

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump({"spearman": corrs}, f, indent=2)

    print("Spearman correlations:")
    for k, v in corrs.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()