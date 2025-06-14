"""
MSMT Testing and Evaluation Script

This script loads a trained MSMT model and evaluates its performance on test data.
It provides comprehensive evaluation metrics, visualization capabilities, and saves
predictions for further analysis.

Usage:
    python test.py --data PEMS04 --checkpoint path/to/best_model.pth

Author: [Your Name]
"""

import util
import argparse
import torch
from models.msmt import MSMT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

parser = argparse.ArgumentParser(description='MSMT Model Testing and Evaluation')
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for testing")
parser.add_argument("--data", type=str, default="PEMS04", help="Dataset name")
parser.add_argument("--input_dim", type=int, default=3, help="Input feature dimension")
parser.add_argument("--channels", type=int, default=128, help="Number of hidden channels")
parser.add_argument("--num_nodes", type=int, default=170, help="Number of nodes in the graph")
parser.add_argument("--input_len", type=int, default=12, help="Input sequence length")
parser.add_argument("--output_len", type=int, default=12, help="Output sequence length")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (for reference)")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay rate (for reference)")
parser.add_argument("--checkpoint", type=str, required=True, 
                   help="Path to the trained model checkpoint")
parser.add_argument("--memory_size", type=int, default=4, help="Memory bank size for MSMT")
parser.add_argument("--save_predictions", type=bool, default=True, 
                   help="Whether to save predictions to disk")
parser.add_argument("--plot_results", type=bool, default=True, 
                   help="Whether to generate evaluation plots")
parser.add_argument("--output_dir", type=str, default="./test_results", 
                   help="Directory to save test results")

args = parser.parse_args()


def configure_dataset_params(args) -> argparse.Namespace:
    """
    Configure dataset-specific parameters
    
    Args:
        args: Command line arguments
        
    Returns:
        Updated arguments with dataset-specific configurations
    """
    dataset_configs = {
        "PEMS08": {"path": "data//PEMS08", "num_nodes": 170},
        "PEMS08_36": {"path": "data//PEMS08_36", "num_nodes": 170, "input_len": 36, "output_len": 36},
        "PEMS08_48": {"path": "data//PEMS08_48", "num_nodes": 170, "input_len": 48, "output_len": 48},
        "PEMS08_60": {"path": "data//PEMS08_60", "num_nodes": 170, "input_len": 60, "output_len": 60},
        "PEMS03": {"path": "data//PEMS03", "num_nodes": 358},
        "PEMS04": {"path": "data//PEMS04", "num_nodes": 307},
        "PEMS04_36": {"path": "data//PEMS04_36", "num_nodes": 307, "input_len": 36, "output_len": 36},
        "PEMS04_48": {"path": "data//PEMS04_48", "num_nodes": 307, "input_len": 48, "output_len": 48},
        "PEMS04_60": {"path": "data//PEMS04_60", "num_nodes": 307, "input_len": 60, "output_len": 60},
        "PEMS07": {"path": "data//PEMS07", "num_nodes": 883},
        "bike_drop": {"path": "data//bike_drop", "num_nodes": 250},
        "bike_pick": {"path": "data//bike_pick", "num_nodes": 250},
        "taxi_drop": {"path": "data//taxi_drop", "num_nodes": 266},
        "taxi_pick": {"path": "data//taxi_pick", "num_nodes": 266},
        "gba_his_2019": {"path": "data//gba_his_2019", "num_nodes": 2352},
        "gla_his_2019": {"path": "data//gla_his_2019", "num_nodes": 3834},
        "ca_his_2019": {"path": "data//ca_his_2019", "num_nodes": 8600},
    }
    
    if args.data in dataset_configs:
        config = dataset_configs[args.data]
        args.data = config["path"]
        args.num_nodes = config["num_nodes"]
        
        # Override sequence lengths if specified in config
        if "input_len" in config:
            args.input_len = config["input_len"]
        if "output_len" in config:
            args.output_len = config["output_len"]
    else:
        print(f"⚠️  Unknown dataset: {args.data}, using default parameters")
    
    return args


def load_model(args, device: torch.device) -> MSMT:
    """
    Load trained MSMT model from checkpoint
    
    Args:
        args: Command line arguments
        device: Device to load model on
        
    Returns:
        Loaded MSMT model
    """
    print("🔧 Initializing MSMT model...")
    
    model = MSMT(
        device=device,
        input_dim=args.input_dim,
        channels=args.channels,
        num_nodes=args.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        dropout=args.dropout,
        memory_size=args.memory_size
    )
    
    model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {model.param_num():,}")
    
    return model


def evaluate_model(model: MSMT, dataloader: Dict, scaler, args, device: torch.device) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Evaluate model on test data
    
    Args:
        model: Trained MSMT model
        dataloader: Data loaders dictionary
        scaler: Data scaler for inverse transform
        args: Command line arguments
        device: Device for computation
        
    Returns:
        Tuple of (predictions, ground_truth, metrics_dict)
    """
    print("🔍 Evaluating model on test data...")
    
    outputs = []
    
    # Prepare ground truth data
    realy = torch.Tensor(dataloader['y_test']).to(device)
    print(f"Original test target shape: {realy.shape}")
    
    # Handle different tensor shapes consistently
    if realy.dim() == 4:
        if realy.shape[1] == 1:  # [S, 1, N, T]
            realy = realy.squeeze(1).permute(0, 2, 1)  # → [S, T, N]
        else:  # [S, C, N, T] - take first channel and transpose
            realy = realy[:, 0, :, :].permute(0, 2, 1)  # → [S, T, N]
    elif realy.dim() == 3:
        if realy.shape[1] == args.num_nodes:  # [S, N, T]
            realy = realy.permute(0, 2, 1)  # → [S, T, N]
        # else already [S, T, N]
    
    print(f"Processed test target shape: {realy.shape}")
    
    # Generate predictions
    print("🚀 Generating predictions...")
    with torch.no_grad():
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)  # [B, T, N, C] → [B, C, N, T]
            
            preds = model(testx)
            if isinstance(preds, tuple):
                preds = preds[0]  # Take main prediction if uncertainty returned
                
            if preds.dim() == 4:
                preds = preds[..., -1]  # Take last time step: [B, C, N, T] → [B, C, N]
                preds = preds.transpose(1, 2)  # [B, C, N] → [B, N, C]
            
            outputs.append(preds.squeeze())
    
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  # Match sizes
    
    print(f"Final prediction shape: {yhat.shape}")
    print(f"Final ground truth shape: {realy.shape}")
    
    # Compute horizon-wise metrics
    horizon_metrics = {
        'mae': [], 'mape': [], 'rmse': [], 'wmape': []
    }
    
    print("\n📈 Horizon-wise Performance:")
    print("=" * 70)
    
    actual_horizons = min(args.output_len, yhat.shape[-1], realy.shape[1])
    
    for horizon in range(actual_horizons):
        # Get predictions and ground truth for this horizon
        if yhat.dim() == 3:  # [S, N, H]
            pred = scaler.inverse_transform(yhat[:, :, horizon])
        else:  # [S, H] 
            pred = scaler.inverse_transform(yhat[:, horizon])
            
        real = realy[:, horizon, :]  # [S, N]
        
        # Compute metrics
        mae, mape, rmse, wmape = util.metric(pred, real)
        
        print(f"Horizon {horizon+1:2d}: MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f} | WMAPE {wmape:.4f}")
        
        horizon_metrics['mae'].append(mae)
        horizon_metrics['mape'].append(mape)
        horizon_metrics['rmse'].append(rmse)
        horizon_metrics['wmape'].append(wmape)
    
    # Average performance
    avg_metrics = {k: np.mean(v) for k, v in horizon_metrics.items()}
    
    print("=" * 70)
    print(f"📊 Average Performance:")
    print(f"   MAE: {avg_metrics['mae']:.4f}")
    print(f"   RMSE: {avg_metrics['rmse']:.4f}")
    print(f"   MAPE: {avg_metrics['mape']:.4f}")
    print(f"   WMAPE: {avg_metrics['wmape']:.4f}")
    print("=" * 70)
    
    # Convert to CPU and original scale for saving
    realy_cpu = realy.cpu()
    yhat_cpu = scaler.inverse_transform(yhat).cpu()
    
    return yhat_cpu, realy_cpu, {'horizon': horizon_metrics, 'average': avg_metrics}


def save_results(predictions: torch.Tensor, ground_truth: torch.Tensor, 
                metrics: Dict, args, output_dir: str) -> None:
    """
    Save test results to disk
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth values
        metrics: Evaluation metrics
        args: Command line arguments
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = args.data.split('/')[-1]
    
    # Save predictions and ground truth
    if args.save_predictions:
        pred_file = os.path.join(output_dir, f"{dataset_name}_predictions.pt")
        real_file = os.path.join(output_dir, f"{dataset_name}_ground_truth.pt")
        
        torch.save(predictions, pred_file)
        torch.save(ground_truth, real_file)
        
        print(f"💾 Predictions saved to: {pred_file}")
        print(f"💾 Ground truth saved to: {real_file}")
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame(metrics['horizon'])
    metrics_df['horizon'] = range(1, len(metrics_df) + 1)
    metrics_file = os.path.join(output_dir, f"{dataset_name}_horizon_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    # Save average metrics
    avg_metrics_df = pd.DataFrame([metrics['average']])
    avg_file = os.path.join(output_dir, f"{dataset_name}_average_metrics.csv")
    avg_metrics_df.to_csv(avg_file, index=False)
    
    print(f"📋 Metrics saved to: {metrics_file}")
    print(f"📋 Average metrics saved to: {avg_file}")


def plot_results(metrics: Dict, args, output_dir: str) -> None:
    """
    Generate evaluation plots
    
    Args:
        metrics: Evaluation metrics
        args: Command line arguments
        output_dir: Output directory
    """
    if not args.plot_results:
        return
        
    dataset_name = args.data.split('/')[-1]
    
    # Create horizon-wise performance plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    horizons = range(1, len(metrics['horizon']['mae']) + 1)
    
    # MAE plot
    ax1.plot(horizons, metrics['horizon']['mae'], 'o-', linewidth=2, markersize=6)
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Forecasting Horizon')
    ax1.set_ylabel('MAE')
    ax1.grid(True, alpha=0.3)
    
    # RMSE plot
    ax2.plot(horizons, metrics['horizon']['rmse'], 'o-', linewidth=2, markersize=6, color='orange')
    ax2.set_title('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Forecasting Horizon')
    ax2.set_ylabel('RMSE')
    ax2.grid(True, alpha=0.3)
    
    # MAPE plot
    ax3.plot(horizons, metrics['horizon']['mape'], 'o-', linewidth=2, markersize=6, color='green')
    ax3.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Forecasting Horizon')
    ax3.set_ylabel('MAPE')
    ax3.grid(True, alpha=0.3)
    
    # WMAPE plot
    ax4.plot(horizons, metrics['horizon']['wmape'], 'o-', linewidth=2, markersize=6, color='red')
    ax4.set_title('Weighted Mean Absolute Percentage Error (WMAPE)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Forecasting Horizon')
    ax4.set_ylabel('WMAPE')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'MSMT Performance on {dataset_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{dataset_name}_performance_plots.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Performance plots saved to: {plot_file}")


def main():
    """Main testing function"""
    print("🎯 MSMT Model Testing Started")
    print("=" * 50)
    
    # Configure parameters
    args = configure_dataset_params(args)
    device = torch.device(args.device)
    
    print(f"📋 Test Configuration:")
    print(f"   Dataset: {args.data}")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    print(f"   Nodes: {args.num_nodes}")
    print(f"   Input Length: {args.input_len}")
    print(f"   Output Length: {args.output_len}")
    
    # Load data
    print(f"\n📂 Loading dataset...")
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    
    # Load model
    model = load_model(args, device)
    
    # Evaluate model
    predictions, ground_truth, metrics = evaluate_model(model, dataloader, scaler, args, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    save_results(predictions, ground_truth, metrics, args, args.output_dir)
    
    # Generate plots
    plot_results(metrics, args, args.output_dir)
    
    print(f"\n🎉 Testing completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()