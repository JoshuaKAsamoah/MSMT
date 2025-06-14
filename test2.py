import util
import argparse
import torch
from model import EnhancedSTAMT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS04", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="weight decay rate")
parser.add_argument('--checkpoint', type=str,
                    default='/home/lay/lay/code/Work2_a/MSMT_qkv/logs/2023-10-31-11:48:57-PEMS04/best_model.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')

# Research paper specific arguments
parser.add_argument('--research_plots', action='store_true', help='Generate research paper quality plots')
parser.add_argument('--target_node', type=int, default=5, help='Target node for detailed analysis')
parser.add_argument('--viz_nodes', type=int, nargs='+', default=[0, 5, 10, 15, 20], help='Nodes to visualize')
parser.add_argument('--comparison_models', type=str, nargs='+', default=['Ground Truth', 'MSMT'], help='Models to compare')
parser.add_argument('--save_research_plots', action='store_true', help='Save research quality plots')
parser.add_argument('--daily_analysis', action='store_true', help='Perform 24-hour daily pattern analysis')
parser.add_argument('--horizon_analysis', action='store_true', help='Detailed horizon analysis')

args = parser.parse_args()

# JUST ADDED THE 4 URBAN DATASETS TO YOUR WORKING CONFIG
DATASET_CONFIG = {
    'PEMS03': {'nodes': 358, 'granularity_min': 5, 'start_date': '2018-09-01', 'end_date': '2018-11-30', 'samples': 26208},
    'PEMS04': {'nodes': 307, 'granularity_min': 5, 'start_date': '2018-01-01', 'end_date': '2018-02-28', 'samples': 16992},
    'PEMS07': {'nodes': 883, 'granularity_min': 5, 'start_date': '2017-05-01', 'end_date': '2017-08-31', 'samples': 28224},
    'PEMS08': {'nodes': 170, 'granularity_min': 5, 'start_date': '2016-07-01', 'end_date': '2016-08-31', 'samples': 17856},
    'BIKE_DROP': {'nodes': 250, 'granularity_min': 30, 'start_date': '2016-04-01', 'end_date': '2016-06-30', 'samples': 4368},
    'BIKE_PICK': {'nodes': 250, 'granularity_min': 30, 'start_date': '2016-04-01', 'end_date': '2016-06-30', 'samples': 4368},
    'TAXI_DROP': {'nodes': 266, 'granularity_min': 30, 'start_date': '2016-04-01', 'end_date': '2016-06-30', 'samples': 4368},
    'TAXI_PICK': {'nodes': 266, 'granularity_min': 30, 'start_date': '2016-04-01', 'end_date': '2016-06-30', 'samples': 4368}
}

def get_datetime_index(dataset_name, num_samples):
    """Generate datetime index for the dataset"""
    # SLIGHTLY MODIFIED YOUR LOGIC TO HANDLE URBAN DATASETS
    dataset_key = dataset_name.replace('data//', '').upper()
    
    # Handle exact matches first
    if dataset_key in DATASET_CONFIG:
        config = DATASET_CONFIG[dataset_key]
    else:
        # For PEMS datasets with suffixes like PEMS04_36
        base_name = dataset_key.split('_')[0]
        config = DATASET_CONFIG.get(base_name, DATASET_CONFIG['PEMS04'])
    
    start_date = pd.to_datetime(config['start_date'])
    granularity = timedelta(minutes=config['granularity_min'])
    
    datetime_index = [start_date + i * granularity for i in range(num_samples)]
    return datetime_index

def identify_weekends(datetime_index):
    """Identify weekend indices in the datetime index"""
    weekend_mask = []
    for dt in datetime_index:
        weekend_mask.append(dt.weekday() >= 5)  # Saturday = 5, Sunday = 6
    return np.array(weekend_mask)

def plot_daily_forecasting_curve(real, pred, node_id, datetime_index, dataset_name, 
                                horizon=0, day_type='weekday', save_dir=None, show_plots=False):
    """
    Create research paper quality daily forecasting curve (like the example you showed)
    """
    if node_id >= real.shape[2]:  # nodes are at index 2
        print(f"Node {node_id} exceeds available nodes {real.shape[2]}")
        return
    
    # Identify weekdays or weekends
    weekend_mask = identify_weekends(datetime_index[:real.shape[0]])
    if day_type == 'weekday':
        day_mask = ~weekend_mask
        title_suffix = "Weekday"
    else:
        day_mask = weekend_mask
        title_suffix = "Weekend"
    
    # Group data by hour of day
    hourly_real = []
    hourly_pred = []
    hours = []
    
    for hour in range(24):
        hour_mask = day_mask & np.array([dt.hour == hour for dt in datetime_index[:real.shape[0]]])
        
        if hour_mask.sum() > 0:
            # Format: [time, horizons, nodes] -> use [:, horizon, node_id]
            hour_real = real[hour_mask, horizon, node_id].mean().item()
            hour_pred = pred[hour_mask, horizon, node_id].mean().item()
            hourly_real.append(hour_real)
            hourly_pred.append(hour_pred)
            hours.append(hour)
    
    # Create the plot (matching your example style)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot lines with publication quality styling
    time_labels = [f"{h:02d}:00" for h in hours]
    x_pos = np.arange(len(hours))
    
    # Plot with different line styles to match research papers
    ax.plot(x_pos, hourly_real, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
    ax.plot(x_pos, hourly_pred, 'r-', linewidth=2.5, label='MSMT', marker='s', markersize=4, alpha=0.8)
    
    # Set professional styling
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Traffic Flow', fontsize=14, fontweight='bold')
    ax.set_title(f'{title_suffix} forecasting curve for node {node_id} on {dataset_name.upper()}.', 
                fontsize=14, fontweight='bold')
    
    print(f"🎯 Creating {title_suffix} plot for node {node_id}")
    
    # Set x-axis ticks and labels
    tick_positions = np.arange(0, len(hours), 6)  # Show every 6 hours: 00:00, 06:00, 12:00, 18:00
    if len(hours) > 0:
        # Add 24:00 (which is same as 00:00 next day) to complete the cycle
        tick_positions = np.append(tick_positions, len(hours)-1)
        tick_labels = [time_labels[i] if i < len(time_labels) else "00:00" for i in tick_positions]
        # Fix the last label to be 00:00 (next day)
        if len(tick_labels) > 0:
            tick_labels[-1] = "00:00"
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Professional grid and legend
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Set axis limits with some padding
    y_min = min(min(hourly_real), min(hourly_pred))
    y_max = max(max(hourly_real), max(hourly_pred))
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_xlim(-0.5, len(hours) - 0.5)
    
    # Calculate and display metrics
    mae = np.abs(np.array(hourly_real) - np.array(hourly_pred)).mean()
    rmse = np.sqrt(np.mean((np.array(hourly_real) - np.array(hourly_pred))**2))
    
    # Add metrics text box
    metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
           facecolor='lightblue', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    
    if save_dir:
        filename = f'{day_type}_forecasting_curve_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Also save as PDF for publications
        filename_pdf = f'{day_type}_forecasting_curve_node_{node_id}_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    
    if show_plots:
        plt.show()
    plt.close()

def plot_horizon_comparison_research(real, pred, node_id, dataset_name, horizons=[0, 2, 5, 11], 
                                   save_dir=None, show_plots=False):
    """
    Research paper quality horizon comparison plot
    """
    if node_id >= real.shape[2]:  # nodes are at index 2
        print(f"Node {node_id} exceeds available nodes {real.shape[2]}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    horizon_names = ['1-step', '3-step', '6-step', '12-step']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (ax, horizon) in enumerate(zip(axes, horizons)):
        if horizon >= real.shape[1]:  # horizons are at index 1
            continue
            
        # Sample representative days
        sample_days = min(7, real.shape[0] // 288)  # 288 = 24*60/5 (samples per day for 5-min intervals)
        samples_per_day = 288
        
        for day in range(sample_days):
            start_idx = day * samples_per_day
            end_idx = min((day + 1) * samples_per_day, real.shape[0])
            
            if end_idx - start_idx < samples_per_day:
                continue
                
            # Format: [time, horizons, nodes] -> use [:, horizon, node_id]
            day_real = real[start_idx:end_idx, horizon, node_id].numpy()
            day_pred = pred[start_idx:end_idx, horizon, node_id].numpy()
            
            # Convert to hourly averages for cleaner visualization
            hours = np.arange(24)
            hourly_real = []
            hourly_pred = []
            
            for hour in range(24):
                hour_start = hour * 12  # 12 samples per hour (5-min intervals)
                hour_end = (hour + 1) * 12
                if hour_end <= len(day_real):
                    hourly_real.append(day_real[hour_start:hour_end].mean())
                    hourly_pred.append(day_pred[hour_start:hour_end].mean())
            
            if len(hourly_real) == 24:
                alpha = 0.3 if day > 0 else 0.8
                linewidth = 1 if day > 0 else 2.5
                
                if day == 0:  # Only label the first day
                    ax.plot(hours, hourly_real, 'b-', linewidth=linewidth, alpha=alpha, 
                           label='Ground Truth', marker='o', markersize=3)
                    ax.plot(hours, hourly_pred, 'r-', linewidth=linewidth, alpha=alpha, 
                           label='MSMT', marker='s', markersize=3)
                else:
                    ax.plot(hours, hourly_real, 'b-', linewidth=linewidth, alpha=alpha)
                    ax.plot(hours, hourly_pred, 'r-', linewidth=linewidth, alpha=alpha)
        
        ax.set_title(f'{horizon_names[i]} ahead prediction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('Traffic Flow', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    
    plt.suptitle(f'Multi-horizon Prediction Analysis - Node {node_id} ({dataset_name.upper()})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        filename = f'horizon_comparison_research_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        
        filename_pdf = f'horizon_comparison_research_node_{node_id}_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    plt.close()

def plot_node_performance_comparison(real, pred, node_ids, dataset_name, 
                                   save_dir=None, show_plots=False):
    """
    Compare performance across multiple nodes - research paper style
    """
    print(f"🔍 DEBUG: node_ids = {node_ids}, real.shape = {real.shape}")
    
    if not node_ids:
        print("❌ ERROR: No valid node_ids provided!")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. MAE comparison across nodes
    ax1 = axes[0, 0]
    node_maes = []
    for node_id in node_ids:
        if node_id < real.shape[2]:  # nodes are at index 2
            # Format: [time, horizons, nodes] -> use [:, :, node_id]
            mae = torch.abs(real[:, :, node_id] - pred[:, :, node_id]).mean().item()
            node_maes.append(mae)
            print(f"📊 Node {node_id}: MAE = {mae:.4f}")
        else:
            print(f"⚠️ Skipping node {node_id} (>= {real.shape[2]})")
    
    if not node_maes:
        print("❌ ERROR: No valid MAE values calculated!")
        return
        
    bars = ax1.bar(range(len(node_maes)), node_maes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Node ID', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax1.set_title('Performance Comparison Across Nodes', fontweight='bold')
    ax1.set_xticks(range(len(node_maes)))
    ax1.set_xticklabels([str(node_ids[i]) for i in range(len(node_maes))])
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars, node_maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Horizon-wise performance
    ax2 = axes[0, 1]
    horizons = range(1, min(13, real.shape[1] + 1))  # horizons are at index 1
    horizon_maes = []
    horizon_stds = []
    
    valid_node_ids = [nid for nid in node_ids if nid < real.shape[2]]  # nodes are at index 2
    
    for h in range(len(horizons)):
        if h < real.shape[1]:  # horizons are at index 1
            mae_per_node = []
            for node_id in valid_node_ids:
                # Format: [time, horizons, nodes] -> use [:, h, node_id]
                mae = torch.abs(real[:, h, node_id] - pred[:, h, node_id]).mean().item()
                mae_per_node.append(mae)
            if mae_per_node:
                horizon_maes.append(np.mean(mae_per_node))
                horizon_stds.append(np.std(mae_per_node))
    
    if horizon_maes:
        ax2.errorbar(horizons[:len(horizon_maes)], horizon_maes, yerr=horizon_stds, 
                    marker='o', markersize=6, linewidth=2, capsize=5, capthick=2)
    ax2.set_xlabel('Prediction Horizon', fontweight='bold')
    ax2.set_ylabel('MAE (Mean ± Std)', fontweight='bold')
    ax2.set_title('Performance vs Prediction Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution
    ax3 = axes[1, 0]
    all_errors = []
    for node_id in valid_node_ids:
        # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
        errors = (real[:, 0, node_id] - pred[:, 0, node_id]).numpy()
        all_errors.extend(errors)
    
    if all_errors:
        ax3.hist(all_errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Error Distribution (1-step ahead)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation matrix
    ax4 = axes[1, 1]
    correlations = []
    for node_id in valid_node_ids:
        # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
        real_vals = real[:, 0, node_id].numpy()
        pred_vals = pred[:, 0, node_id].numpy()
        corr = np.corrcoef(real_vals, pred_vals)[0, 1]
        correlations.append(corr)
    
    if correlations:
        bars = ax4.bar(range(len(correlations)), correlations, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(len(correlations)))
        ax4.set_xticklabels([str(valid_node_ids[i]) for i in range(len(correlations))])
        
        # Add value labels
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Node ID', fontweight='bold')
    ax4.set_ylabel('Correlation Coefficient', fontweight='bold')
    ax4.set_title('Prediction Correlation by Node', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.suptitle(f'Comprehensive Performance Analysis ({dataset_name.upper()})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        filename = f'node_performance_comparison_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        
        filename_pdf = f'node_performance_comparison_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    plt.close()

def plot_research_summary_figure(real, pred, node_id, dataset_name, datetime_index,
                                save_dir=None, show_plots=False):
    """
    Create a comprehensive research paper summary figure
    """
    if node_id >= real.shape[2]:  # nodes are at index 2
        print(f"Node {node_id} exceeds available nodes {real.shape[2]}")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x2 subplot layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # 1. Daily pattern (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    weekend_mask = identify_weekends(datetime_index[:real.shape[0]])
    weekday_mask = ~weekend_mask
    
    # Calculate hourly averages for weekdays
    hourly_real_wd = []
    hourly_pred_wd = []
    for hour in range(24):
        hour_mask = weekday_mask & np.array([dt.hour == hour for dt in datetime_index[:real.shape[0]]])
        if hour_mask.sum() > 0:
            # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
            hourly_real_wd.append(real[hour_mask, 0, node_id].mean().item())
            hourly_pred_wd.append(pred[hour_mask, 0, node_id].mean().item())
        else:
            hourly_real_wd.append(0)
            hourly_pred_wd.append(0)
    
    hours = range(24)
    ax1.plot(hours, hourly_real_wd, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4)
    ax1.plot(hours, hourly_pred_wd, 'r-', linewidth=2.5, label='MSMT', marker='s', markersize=4)
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Traffic Flow', fontweight='bold')
    ax1.set_title('(a) Daily Traffic Pattern', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 25, 6))
    ax1.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    
    # 2. Multi-horizon performance (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    horizons = range(1, min(13, real.shape[1] + 1))  # horizons are at index 1
    mae_by_horizon = []
    for h in range(len(horizons)):
        if h < real.shape[1]:  # horizons are at index 1
            # Format: [time, horizons, nodes] -> use [:, h, node_id]
            mae = torch.abs(real[:, h, node_id] - pred[:, h, node_id]).mean().item()
            mae_by_horizon.append(mae)
    
    ax2.plot(horizons[:len(mae_by_horizon)], mae_by_horizon, 'o-', linewidth=2.5, markersize=6, color='green')
    ax2.set_xlabel('Prediction Horizon', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax2.set_title('(b) Performance vs Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sample time series (middle, spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])
    sample_length = min(288*3, real.shape[0])  # 3 days
    # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
    sample_real = real[:sample_length, 0, node_id].numpy()
    sample_pred = pred[:sample_length, 0, node_id].numpy()
    sample_times = datetime_index[:sample_length]
    
    ax3.plot(sample_times, sample_real, 'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
    ax3.plot(sample_times, sample_pred, 'r-', linewidth=1.5, label='MSMT', alpha=0.8)
    
    # Highlight weekends
    weekend_sample_mask = identify_weekends(sample_times)
    for i, is_weekend in enumerate(weekend_sample_mask):
        if is_weekend and i < len(sample_times):
            ax3.axvspan(sample_times[i], sample_times[min(i+1, len(sample_times)-1)], 
                       alpha=0.1, color='yellow')
    
    ax3.set_xlabel('Date/Time', fontweight='bold')
    ax3.set_ylabel('Traffic Flow', fontweight='bold')
    ax3.set_title('(c) Sample Time Series (3 days)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    # 4. Scatter plot (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    sample_indices = np.random.choice(real.shape[0], min(1000, real.shape[0]), replace=False)
    # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
    real_sample = real[sample_indices, 0, node_id].numpy()
    pred_sample = pred[sample_indices, 0, node_id].numpy()
    
    ax4.scatter(real_sample, pred_sample, alpha=0.6, s=20, color='purple')
    min_val, max_val = min(real_sample.min(), pred_sample.min()), max(real_sample.max(), pred_sample.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    ax4.set_xlabel('Ground Truth', fontweight='bold')
    ax4.set_ylabel('Prediction', fontweight='bold')
    ax4.set_title('(d) Prediction Accuracy', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add R² value
    r2 = np.corrcoef(real_sample, pred_sample)[0, 1]**2
    ax4.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax4.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. Error analysis (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])
    # Format: [time, horizons, nodes] -> use [:, 0, node_id] for first horizon
    errors = (real[:, 0, node_id] - pred[:, 0, node_id]).numpy()
    ax5.hist(errors, bins=40, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
    ax5.set_xlabel('Prediction Error', fontweight='bold')
    ax5.set_ylabel('Density', fontweight='bold')
    ax5.set_title('(e) Error Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Analysis: Node {node_id} on {dataset_name.upper()}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        filename = f'research_summary_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        
        filename_pdf = f'research_summary_node_{node_id}_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    plt.close()

def main():
    print(f"🚨 DEBUG: args.target_node = {args.target_node}")
    print(f"🚨 DEBUG: args.data = {args.data}")
    
    # ADDED THE 4 URBAN DATASETS TO YOUR WORKING DATASET CONFIGURATION
    if args.data == "PEMS08":
        args.data = "data//"+args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"
    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
    elif args.data == "PEMS03":
        args.data = "data//"+args.data
        args.num_nodes = 358
        args.adjdata = "data/adj/adj_PEMS03_gs.npy"
    elif args.data == "PEMS07":
        args.data = "data//"+args.data
        args.num_nodes = 883
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"
    # ADDED THESE 4 URBAN DATASETS
    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        args.num_nodes = 250
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"
    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        args.num_nodes = 250
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"
    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        args.num_nodes = 266
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"
    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        args.num_nodes = 266
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"

    device = torch.device(args.device)

    model = EnhancedSTAMT(
            device, args.input_dim, args.channels, args.num_nodes, args.input_len, args.output_len, args.dropout
        )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('Model loaded successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    
    realy = torch.Tensor(dataloader['y_test']).to(device)
    
    print(f"🔍 Original tensor shape: {realy.shape}")
    print(f"🔍 Expected nodes from args: {args.num_nodes}")
    
    # Handle tensor shapes - ENSURE FORMAT IS [time, horizons, nodes]
    if realy.dim() == 4:
        if realy.shape[1] == 1:
            realy = realy.squeeze(1).permute(0, 2, 1)  # [time, horizons, nodes]
        elif realy.shape[-1] == 1:
            # Original: [time, horizons, nodes, 1] -> [time, horizons, nodes]
            realy = realy.squeeze(-1)
        else:
            realy = realy[:, 0, :, :].permute(0, 2, 1)
    elif realy.dim() == 3:
        # Check if it's [time, nodes, horizons] and swap to [time, horizons, nodes]
        if realy.shape[1] == args.num_nodes:
            # Shape is [time, nodes, horizons] - swap to [time, horizons, nodes]
            realy = realy.permute(0, 2, 1)
        elif realy.shape[2] == args.num_nodes:
            # Shape is [time, horizons, nodes] - already correct!
            pass
        else:
            # Try to detect based on which dimension matches num_nodes
            if realy.shape[1] == args.num_nodes:
                realy = realy.permute(0, 2, 1)
            # Otherwise assume it's already in correct format
    
    print(f"🔍 Final tensor shape: {realy.shape}")
    print(f"🔍 Format should be [time, horizons, nodes]")
    print(f"🔍 Using {realy.shape[2]} nodes for analysis")

    # Model inference (same as before)
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx)
            if isinstance(preds, tuple):
                preds = preds[0]
            if preds.dim() == 4:
                preds = preds[..., -1]
        outputs.append(preds)
    
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    
    print(f"🔍 yhat shape after concatenation: {yhat.shape}")
    
    # Make sure yhat and realy have the same shape format [time, horizons, nodes]
    if yhat.shape != realy.shape:
        print(f"⚠️ Shape mismatch! yhat: {yhat.shape}, realy: {realy.shape}")
        # Ensure both are [time, horizons, nodes]
        if yhat.dim() == 3:
            if yhat.shape[1] == args.num_nodes:
                # yhat is [time, nodes, horizons] -> swap to [time, horizons, nodes]
                yhat = yhat.permute(0, 2, 1)
                print(f"🔧 Reshaped yhat to: {yhat.shape}")
            elif yhat.shape[2] == args.num_nodes:
                # yhat is already [time, horizons, nodes]
                print(f"✅ yhat already in correct format")
    
    print(f"🔍 Final shapes - yhat: {yhat.shape}, realy: {realy.shape}")
    print(f"🔍 Both tensors should be [time, horizons, nodes]")

    # Calculate metrics (same as before)
    amae = []
    amape = []
    awmape = []
    armse = []
    
    actual_horizons = min(args.output_len, yhat.shape[1], realy.shape[1])
    
    for i in range(actual_horizons):
        pred = scaler.inverse_transform(yhat[:, i, :])
        real = realy[:, i, :]
        
        metrics = util.metric(pred, real)
        print(f'Horizon {i+1}: MAE={metrics[0]:.4f}, MAPE={metrics[1]:.4f}, RMSE={metrics[2]:.4f}')
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    print(f'Average: MAE={np.mean(amae):.4f}, MAPE={np.mean(amape):.4f}, RMSE={np.mean(armse):.4f}')

    # Apply inverse transform
    realy = realy.to("cpu")
    yhat_inverse = torch.zeros_like(yhat)
    for i in range(yhat.shape[1]):
        yhat_inverse[:, i, :] = scaler.inverse_transform(yhat[:, i, :])
    yhat1 = yhat_inverse.to("cpu")

    # Research paper quality visualizations
    if args.research_plots:
        print("Creating research paper quality visualizations...")
        
        dataset_name = args.data.split('/')[-1] if '/' in args.data else args.data
        datetime_index = get_datetime_index(dataset_name, realy.shape[0])
        
        save_dir = None
        if args.save_research_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"plots_{dataset_name}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            print(f"Research plots will be saved to: {save_dir}")
        
        print(f"🔧 Args target_node: {args.target_node}")
        print(f"📊 Dataset shape after reshaping: {realy.shape}")
        print(f"📊 Number of nodes available: {realy.shape[2]}")  # nodes are at index 2
        
        # Ensure target node is valid - use the correct dimension!
        target_node = min(args.target_node, realy.shape[2] - 1)  # shape[2] is nodes
        
        # Fix viz_nodes - ensure we have valid nodes to visualize
        viz_nodes = [node for node in args.viz_nodes if node < realy.shape[2]]  # shape[2] is nodes
        
        # If no valid viz_nodes, create some automatically
        if not viz_nodes:
            max_node = realy.shape[2] - 1  # shape[2] is nodes
            if max_node >= 20:
                viz_nodes = [0, 5, 10, 15, 20]
            elif max_node >= 10:
                viz_nodes = [0, max_node//4, max_node//2, 3*max_node//4, max_node]
            else:
                viz_nodes = list(range(min(5, max_node + 1)))
        
        print(f"✅ Final target node: {target_node}")
        print(f"✅ Final viz_nodes: {viz_nodes}")
        
        # 1. Daily forecasting curve (like your example)
        if args.daily_analysis:
            print("Creating daily forecasting curves...")
            plot_daily_forecasting_curve(realy, yhat1, target_node, datetime_index, dataset_name,
                                        horizon=0, day_type='weekday', save_dir=save_dir, show_plots=False)
            plot_daily_forecasting_curve(realy, yhat1, target_node, datetime_index, dataset_name,
                                        horizon=0, day_type='weekend', save_dir=save_dir, show_plots=False)
        
        # 2. Multi-horizon analysis
        if args.horizon_analysis:
            print("Creating horizon comparison...")
            plot_horizon_comparison_research(realy, yhat1, target_node, dataset_name,
                                           save_dir=save_dir, show_plots=False)
        
        # 3. Node performance comparison
        print("Creating node performance comparison...")
        plot_node_performance_comparison(realy, yhat1, viz_nodes, dataset_name,
                                       save_dir=save_dir, show_plots=False)
        
        # 4. Comprehensive summary figure
        print("Creating comprehensive summary figure...")
        plot_research_summary_figure(realy, yhat1, target_node, dataset_name, datetime_index,
                                    save_dir=save_dir, show_plots=False)
        
        print("Research paper quality visualizations complete!")
        
        if save_dir:
            # Create research summary document
            with open(os.path.join(save_dir, 'Summary.txt'), 'w') as f:
                f.write(f"Research Paper Analysis Summary\n")
                f.write(f"={'='*50}\n\n")
                f.write(f"Dataset: {dataset_name.upper()}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target Node: {target_node}\n")
                f.write(f"Comparison Nodes: {viz_nodes}\n")
                f.write(f"Data Shape: {realy.shape}\n")
                f.write(f"Date Range: {datetime_index[0]} to {datetime_index[-1]}\n\n")
                
                f.write("Performance Metrics:\n")
                f.write(f"Average MAE: {np.mean(amae):.4f}\n")
                f.write(f"Average RMSE: {np.mean(armse):.4f}\n")
                f.write(f"Average MAPE: {np.mean(amape):.4f}%\n")
                f.write(f"Average WMAPE: {np.mean(awmape):.4f}%\n\n")
                
                f.write("Files Generated:\n")
                f.write("- weekday_forecasting_curve_node_X_dataset.png/pdf\n")
                f.write("- weekend_forecasting_curve_node_X_dataset.png/pdf\n")
                f.write("- horizon_comparison_node_X_dataset.png/pdf\n")
                f.write("- node_performance_comparison_dataset.png/pdf\n")
                f.write("- summary_node_X_dataset.png/pdf\n")
                
                f.write("\nRecommended Usage:\n")
                f.write("- Use daily forecasting curves for showing model accuracy\n")
                f.write("- Use horizon comparison for multi-step prediction analysis\n")
                f.write("- Use node comparison for spatial performance analysis\n")
                f.write("- Use comprehensive summary as main figure in paper\n")

if __name__ == "__main__":
    main()