import os, time, argparse
import numpy as np
import pandas as pd
import torch
from util import load_dataset, StandardScaler, metric
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def prepare_xy_normalized(data_dir):
    """
    Load data and apply Z-score normalization like most traffic forecasting papers
    """
    # loads x_train,y_train,x_val,y_val,x_test,y_test
    data = load_dataset(data_dir, 1, 1, 1)  # batch sizes=1
    
    # grab the raw arrays
    X_train = data['x_train'][..., 0]  # [samples,12,nodes]
    Y_train = data['y_train'][..., 0]  # [samples,12,nodes]
    X_val = data['x_val'][..., 0]
    Y_val = data['y_val'][..., 0]
    X_test = data['x_test'][..., 0]
    Y_test = data['y_test'][..., 0]
    
    print(f"Raw data ranges - X_train: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"Raw data ranges - Y_train: [{Y_train.min():.2f}, {Y_train.max():.2f}]")
    
    # Z-score normalization using training data statistics
    # Compute mean and std across all training samples and time steps
    train_data = np.concatenate([X_train.reshape(-1, X_train.shape[-1]), 
                                Y_train.reshape(-1, Y_train.shape[-1])], axis=0)
    
    # Per-node normalization (common in traffic forecasting)
    mean_per_node = train_data.mean(axis=0, keepdims=True)  # [1, nodes]
    std_per_node = train_data.std(axis=0, keepdims=True)    # [1, nodes]
    
    # Avoid division by zero
    std_per_node = np.where(std_per_node < 1e-8, 1.0, std_per_node)
    
    print(f"Normalization stats - Mean: [{mean_per_node.min():.2f}, {mean_per_node.max():.2f}]")
    print(f"Normalization stats - Std: [{std_per_node.min():.2f}, {std_per_node.max():.2f}]")
    
    # Apply normalization
    def normalize(data):
        return (data - mean_per_node) / std_per_node
    
    def denormalize(data):
        return data * std_per_node + mean_per_node
    
    X_train_norm = normalize(X_train)
    Y_train_norm = normalize(Y_train)
    X_val_norm = normalize(X_val)
    Y_val_norm = normalize(Y_val)
    X_test_norm = normalize(X_test)
    Y_test_norm = normalize(Y_test)
    
    print(f"Normalized data ranges - X_train: [{X_train_norm.min():.2f}, {X_train_norm.max():.2f}]")
    print(f"Normalized data ranges - Y_train: [{Y_train_norm.min():.2f}, {Y_train_norm.max():.2f}]")
    
    # Return normalized data and normalization functions
    return (X_train_norm, Y_train_norm, X_val_norm, Y_val_norm, X_test_norm, Y_test_norm,
            normalize, denormalize, mean_per_node, std_per_node)

def run_ha(X, Y):
    """
    Historical Average - Traffic Forecasting Style:
    Use the average of recent time steps (not all 12)
    """
    # Use last 3 time steps average (common in traffic forecasting)
    recent_avg = X[:, -3:, :].mean(axis=1, keepdims=True)  # [S, 1, N]
    
    # Repeat for all prediction horizons
    preds = np.repeat(recent_avg, Y.shape[1], axis=1)  # [S, 12, N]
    
    return preds

def run_persistence(X, Y):
    """
    Simple Persistence: Use the last observed value
    Very common baseline in traffic forecasting
    """
    last_val = X[:, -1:, :]  # [S, 1, N]
    preds = np.repeat(last_val, Y.shape[1], axis=1)  # [S, 12, N]
    return preds

def run_linear_trend(X, Y):
    """
    Linear Trend: Fit simple linear trend to last few points
    Better than VAR for traffic data
    """
    S, T_in, N = X.shape
    T_out = Y.shape[1]
    preds = np.zeros((S, T_out, N))
    
    # Use last 6 points to estimate trend
    window = min(6, T_in)
    
    for s in range(S):
        for n in range(N):
            # Get recent values
            recent = X[s, -window:, n]  # [window]
            time_steps = np.arange(window)
            
            # Fit linear trend: y = a*t + b
            if window > 1:
                A = np.vstack([time_steps, np.ones(len(time_steps))]).T
                try:
                    coeffs = np.linalg.lstsq(A, recent, rcond=None)[0]
                    slope, intercept = coeffs
                    
                    # Project forward
                    future_steps = np.arange(window, window + T_out)
                    trend_pred = slope * future_steps + intercept
                    
                    # Apply damping to prevent unrealistic trends
                    damping = np.exp(-0.1 * np.arange(T_out))
                    base_val = recent[-1]
                    preds[s, :, n] = base_val + (trend_pred - base_val) * damping
                    
                except:
                    # Fallback: use last value
                    preds[s, :, n] = recent[-1]
            else:
                preds[s, :, n] = recent[-1]
    
    return preds

def run_svr_simple(X_train, Y_train, X_predict):
    """
    Simplified SVR for traffic forecasting:
    - Use only recent values as features
    - Ridge regression instead of full SVR for stability
    - Per-node models
    """
    S_train, T_in, N = X_train.shape
    S_pred = X_predict.shape[0]
    T_out = Y_train.shape[1]
    preds = np.zeros((S_pred, T_out, N))
    
    print(f"Training Ridge regression models for {N} nodes...")
    
    for node in range(N):
        print(f"  Node {node+1}/{N}", end='\r')
        
        for horizon in range(T_out):
            try:
                # Features: use last 4 values + simple trend
                X_features_train = []
                X_features_pred = []
                
                for s in range(S_train):
                    seq = X_train[s, :, node]
                    # Features: last 4 values + trend + moving average
                    features = [
                        seq[-1], seq[-2], seq[-3], seq[-4],  # last 4 values
                        seq[-1] - seq[-4],  # short trend
                        seq[-3:].mean(),    # recent average
                    ]
                    X_features_train.append(features)
                
                for s in range(S_pred):
                    seq = X_predict[s, :, node]
                    features = [
                        seq[-1], seq[-2], seq[-3], seq[-4],
                        seq[-1] - seq[-4],
                        seq[-3:].mean(),
                    ]
                    X_features_pred.append(features)
                
                X_features_train = np.array(X_features_train)
                X_features_pred = np.array(X_features_pred)
                y_train = Y_train[:, horizon, node]
                
                # Use Ridge regression (more stable than SVR)
                model = Ridge(alpha=1.0)
                model.fit(X_features_train, y_train)
                
                preds[:, horizon, node] = model.predict(X_features_pred)
                
            except Exception as e:
                print(f"\nRidge failed for node {node}, horizon {horizon}: {e}")
                # Fallback: use last value
                preds[:, horizon, node] = X_predict[:, -1, node]
    
    print()  # new line after progress
    return preds

def evaluate_and_save(prefix, Y_true, Y_pred, out_dir, denormalize_fn=None):
    """
    Evaluate predictions and save results
    If denormalize_fn is provided, denormalize before evaluation
    """
    # Denormalize if needed
    if denormalize_fn is not None:
        Y_true_eval = denormalize_fn(Y_true)
        Y_pred_eval = denormalize_fn(Y_pred)
        print(f"  Denormalized for evaluation - True: [{Y_true_eval.min():.2f}, {Y_true_eval.max():.2f}], Pred: [{Y_pred_eval.min():.2f}, {Y_pred_eval.max():.2f}]")
    else:
        Y_true_eval = Y_true
        Y_pred_eval = Y_pred
    
    horizons = Y_true_eval.shape[1]
    rows = []
    
    for h in range(horizons):
        # Convert to torch tensors
        pred_t = torch.from_numpy(Y_pred_eval[:, h, :]).float()
        real_t = torch.from_numpy(Y_true_eval[:, h, :]).float()

        mae, mape, rmse, wmape = metric(pred_t, real_t)
        print(f"  {prefix} horizon {h+1:02d} ▶ MAE {mae:.4f}, RMSE {rmse:.4f}, MAPE {mape:.4f}, WMAPE {wmape:.4f}")
        
        rows.append({
            "horizon": h+1,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "WMAPE": wmape
        })
    
    # Calculate averages
    avg = {
        "horizon": "avg",
        "MAE": np.mean([r["MAE"] for r in rows]),
        "RMSE": np.mean([r["RMSE"] for r in rows]),
        "MAPE": np.mean([r["MAPE"] for r in rows]),
        "WMAPE": np.mean([r["WMAPE"] for r in rows])
    }
    rows.append(avg)
    print(f"  {prefix} AVERAGE ▶ MAE {avg['MAE']:.4f}, RMSE {avg['RMSE']:.4f}, MAPE {avg['MAPE']:.4f}, WMAPE {avg['WMAPE']:.4f}")

    # Save results
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/PEMS08", help="dir of npz files")
    parser.add_argument("--out", default="logs/baselines", help="where to write .csv")
    parser.add_argument("--no_norm", action="store_true", help="skip normalization")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    
    # Load and normalize data
    print("Loading and normalizing data...")
    if args.no_norm:
        print("Skipping normalization (--no_norm flag)")
        # Original loading without normalization
        data = load_dataset(args.data, 1, 1, 1)
        Xtr, Ytr = data['x_train'][..., 0], data['y_train'][..., 0]
        Xv, Yv = data['x_val'][..., 0], data['y_val'][..., 0]
        Xt, Yt = data['x_test'][..., 0], data['y_test'][..., 0]
        normalize_fn, denormalize_fn = None, None
    else:
        (Xtr, Ytr, Xv, Yv, Xt, Yt, 
         normalize_fn, denormalize_fn, mean_stats, std_stats) = prepare_xy_normalized(args.data)

    print(f"\nData shapes:")
    print(f"  Train: X{Xtr.shape} Y{Ytr.shape}")
    print(f"  Val:   X{Xv.shape} Y{Yv.shape}")
    print(f"  Test:  X{Xt.shape} Y{Yt.shape}")

    # Store results for comparison
    results_summary = []

    print("\n" + "="*60)
    print("=== HISTORICAL AVERAGE (Last 3 Steps) ===")
    print("="*60)
    ha_preds_tr = run_ha(Xtr, Ytr)
    res = evaluate_and_save("HA_train", Ytr, ha_preds_tr, args.out, denormalize_fn)
    results_summary.append(("HA_train", res))
    
    ha_preds_v = run_ha(Xv, Yv)
    res = evaluate_and_save("HA_val", Yv, ha_preds_v, args.out, denormalize_fn)
    results_summary.append(("HA_val", res))
    
    ha_preds_t = run_ha(Xt, Yt)
    res = evaluate_and_save("HA_test", Yt, ha_preds_t, args.out, denormalize_fn)
    results_summary.append(("HA_test", res))

    print("\n" + "="*60)
    print("=== PERSISTENCE (Last Value) ===")
    print("="*60)
    pers_preds_tr = run_persistence(Xtr, Ytr)
    res = evaluate_and_save("PERS_train", Ytr, pers_preds_tr, args.out, denormalize_fn)
    results_summary.append(("PERS_train", res))
    
    pers_preds_v = run_persistence(Xv, Yv)
    res = evaluate_and_save("PERS_val", Yv, pers_preds_v, args.out, denormalize_fn)
    results_summary.append(("PERS_val", res))
    
    pers_preds_t = run_persistence(Xt, Yt)
    res = evaluate_and_save("PERS_test", Yt, pers_preds_t, args.out, denormalize_fn)
    results_summary.append(("PERS_test", res))

    print("\n" + "="*60)
    print("=== LINEAR TREND ===")
    print("="*60)
    lt_preds_tr = run_linear_trend(Xtr, Ytr)
    res = evaluate_and_save("LT_train", Ytr, lt_preds_tr, args.out, denormalize_fn)
    results_summary.append(("LT_train", res))
    
    lt_preds_v = run_linear_trend(Xv, Yv)
    res = evaluate_and_save("LT_val", Yv, lt_preds_v, args.out, denormalize_fn)
    results_summary.append(("LT_val", res))
    
    lt_preds_t = run_linear_trend(Xt, Yt)
    res = evaluate_and_save("LT_test", Yt, lt_preds_t, args.out, denormalize_fn)
    results_summary.append(("LT_test", res))

    print("\n" + "="*60)
    print("=== RIDGE REGRESSION ===")
    print("="*60)
    ridge_preds_tr = run_svr_simple(Xtr, Ytr, Xtr)
    res = evaluate_and_save("RIDGE_train", Ytr, ridge_preds_tr, args.out, denormalize_fn)
    results_summary.append(("RIDGE_train", res))
    
    ridge_preds_v = run_svr_simple(Xtr, Ytr, Xv)
    res = evaluate_and_save("RIDGE_val", Yv, ridge_preds_v, args.out, denormalize_fn)
    results_summary.append(("RIDGE_val", res))
    
    ridge_preds_t = run_svr_simple(Xtr, Ytr, Xt)
    res = evaluate_and_save("RIDGE_test", Yt, ridge_preds_t, args.out, denormalize_fn)
    results_summary.append(("RIDGE_test", res))

    # Print summary
    print("\n" + "="*80)
    print("=== RESULTS SUMMARY ===")
    print("="*80)
    print(f"{'Method':<15} {'MAE':<8} {'RMSE':<8} {'MAPE':<8} {'WMAPE':<8}")
    print("-" * 60)
    
    for method, res in results_summary:
        if 'test' in method:  # Only show test results in summary
            method_name = method.replace('_test', '')
            print(f"{method_name:<15} {res['MAE']:<8.3f} {res['RMSE']:<8.3f} {res['MAPE']:<8.3f} {res['WMAPE']:<8.3f}")
    
    print("\n" + "="*80)
    print(f"All results saved to: {args.out}")
    print("Expected ranking (best to worst): RIDGE < LT < HA < PERS")
    print("Your advanced model should beat all of these!")
    print("="*80)