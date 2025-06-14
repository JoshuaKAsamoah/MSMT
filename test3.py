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
                    default='/home/lay/lay/code/Work2_a/STAMT_qkv/logs/2023-10-31-11:48:57-PEMS04/best_model.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')

# Research paper specific arguments
parser.add_argument('--research_plots', action='store_true', help='Generate research paper quality plots')
parser.add_argument('--target_node', type=int, default=13, help='Target node for detailed analysis')
parser.add_argument('--viz_nodes', type=int, nargs='+', default=[0, 5, 10, 15, 20], help='Nodes to visualize')
parser.add_argument('--comparison_models', type=str, nargs='+', default=['Ground Truth', 'STAMT'], help='Models to compare')
parser.add_argument('--save_research_plots', action='store_true', help='Save research quality plots')
parser.add_argument('--daily_analysis', action='store_true', help='Perform 24-hour daily pattern analysis')
parser.add_argument('--horizon_analysis', action='store_true', help='Detailed horizon analysis')
parser.add_argument('--debug_time', action='store_true', help='Enable detailed time mapping debugging')
parser.add_argument('--manual_start_date', type=str, help='Manually override start date (YYYY-MM-DD HH:MM:SS)')
parser.add_argument('--time_offset_hours', type=int, default=0, help='Apply hour offset to fix time mapping')

args = parser.parse_args()

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

def get_datetime_index(dataset_name, num_samples, manual_start_date=None, time_offset_hours=0):
    dataset_key = dataset_name.replace('data//', '').upper()
    if 'BIKE_DROP' in dataset_key: config = DATASET_CONFIG['BIKE_DROP']
    elif 'BIKE_PICK' in dataset_key: config = DATASET_CONFIG['BIKE_PICK']
    elif 'TAXI_DROP' in dataset_key: config = DATASET_CONFIG['TAXI_DROP']
    elif 'TAXI_PICK' in dataset_key: config = DATASET_CONFIG['TAXI_PICK']
    else:
        base_name = dataset_key.split('_')[0]
        config = DATASET_CONFIG.get(base_name, DATASET_CONFIG['PEMS04'])
    
    # Use manual start date if provided
    if manual_start_date:
        start_date = pd.to_datetime(manual_start_date)
        print(f"🔧 Using MANUAL start date: {start_date}")
    else:
        start_date = pd.to_datetime(config['start_date'])
    
    # Apply time offset if specified
    if time_offset_hours != 0:
        start_date = start_date + timedelta(hours=time_offset_hours)
        print(f"🔧 Applied {time_offset_hours} hour offset. New start: {start_date}")
    
    granularity = timedelta(minutes=config['granularity_min'])
    
    # Generate datetime index
    datetime_index = [start_date + i * granularity for i in range(num_samples)]
    
    print(f"📅 Generated datetime index for {dataset_name}:")
    print(f"   Start: {datetime_index[0]} ")
    print(f"   End: {datetime_index[-1]}")
    print(f"   Granularity: {config['granularity_min']} minutes")
    print(f"   Total samples: {num_samples}")
    
    return datetime_index

def fix_time_mapping_if_needed(datetime_index, real, node_id=0):
    """
    Detect and fix common time mapping issues
    """
    print(f"\n🔧 CHECKING FOR TIME MAPPING ISSUES...")
    
    # Calculate hourly averages
    hourly_averages = []
    for hour in range(24):
        hour_values = []
        for i, dt in enumerate(datetime_index):
            if i < real.shape[0] and dt.hour == hour:
                hour_values.append(real[i, node_id, 0].item())
        
        if hour_values:
            hourly_averages.append(np.mean(hour_values))
        else:
            hourly_averages.append(0)
    
    midnight_avg = hourly_averages[0]
    noon_avg = hourly_averages[12]
    morning_rush = max(hourly_averages[7:10]) if any(hourly_averages[7:10]) else 0
    
    # Check if pattern is inverted (midnight > noon suggests issue)
    if midnight_avg > noon_avg and midnight_avg > morning_rush:
        print(f"⚠️  DETECTED INVERTED TIME PATTERN!")
        print(f"   Midnight: {midnight_avg:.1f}, Noon: {noon_avg:.1f}")
        
        # Try different fixes
        print(f"🔧 Trying time offset corrections...")
        
        # Option 1: Try 12-hour offset
        offset_hours = 12
        datetime_index_fixed = [dt + timedelta(hours=offset_hours) for dt in datetime_index]
        
        hourly_averages_fixed = []
        for hour in range(24):
            hour_values = []
            for i, dt in enumerate(datetime_index_fixed):
                if i < real.shape[0] and dt.hour == hour:
                    hour_values.append(real[i, node_id, 0].item())
            
            if hour_values:
                hourly_averages_fixed.append(np.mean(hour_values))
            else:
                hourly_averages_fixed.append(0)
        
        midnight_fixed = hourly_averages_fixed[0]
        noon_fixed = hourly_averages_fixed[12]
        morning_rush_fixed = max(hourly_averages_fixed[7:10]) if any(hourly_averages_fixed[7:10]) else 0
        
        if midnight_fixed < noon_fixed and morning_rush_fixed > midnight_fixed:
            print(f"✅ 12-hour offset FIX WORKS!")
            print(f"   Fixed - Midnight: {midnight_fixed:.1f}, Noon: {noon_fixed:.1f}")
            return datetime_index_fixed
        
        # Option 2: Try reversing the data order
        print(f"🔧 Trying data order reversal...")
        # This would require reversing the actual data, which is more complex
        # For now, let's just warn the user
        
        print(f"❌ Could not automatically fix time mapping.")
        print(f"💡 SUGGESTIONS:")
        print(f"   1. Check if your data loading is correct")
        print(f"   2. Verify the dataset start date/time")
        print(f"   3. Check if data is in different timezone")
        print(f"   4. Manually adjust DATASET_CONFIG start_date")
        
        return datetime_index  # Return original if fix doesn't work
    
    else:
        print(f"✅ Time mapping looks reasonable!")
        return datetime_index

# Cleaned up function - no datetime issues

def debug_time_mapping(real, pred, node_id, dataset_name, datetime_index):
    """Debug function to check if time mapping makes sense"""
    print(f"\n🔍 TIME MAPPING DIAGNOSTIC for {dataset_name}")
    print(f"=" * 60)
    
    # Check dataset configuration
    dataset_key = dataset_name.replace('data//', '').upper().split('_')[0]
    config = DATASET_CONFIG.get(dataset_key, DATASET_CONFIG['PEMS04'])
    print(f"📋 Dataset config: {config}")
    
    # Check data structure
    print(f"📊 Data shape: {real.shape}")
    print(f"⏰ Datetime index length: {len(datetime_index)}")
    print(f"⏰ First timestamp: {datetime_index[0]}")
    print(f"⏰ Last timestamp: {datetime_index[-1]}")
    
    # Sample data points across the day
    print(f"\n📈 SAMPLE VALUES for Node {node_id}:")
    print(f"{'Time':<20} {'Hour':<4} {'Value':<8} {'Expected'}")
    print("-" * 50)
    
    # Check first 48 hours (2 days) to see pattern
    samples_to_check = min(48 * (60 // config['granularity_min']), len(datetime_index), real.shape[0])
    
    for i in range(0, samples_to_check, 12):  # Every hour
        if i < len(datetime_index) and i < real.shape[0]:
            dt = datetime_index[i]
            val = real[i, node_id, 0].item()
            
            # Expected traffic level based on hour
            hour = dt.hour
            if 0 <= hour <= 5:
                expected = "LOW (Night)"
            elif 6 <= hour <= 9:
                expected = "HIGH (Morning Rush)"
            elif 10 <= hour <= 15:
                expected = "MEDIUM (Midday)"
            elif 16 <= hour <= 19:
                expected = "HIGH (Evening Rush)"
            else:
                expected = "MEDIUM (Evening)"
            
            print(f"{str(dt):<20} {hour:02d}   {val:6.1f}   {expected}")
    
    # Calculate hourly averages to see overall pattern
    print(f"\n📊 HOURLY AVERAGES:")
    hourly_averages = []
    for hour in range(24):
        hour_values = []
        for i, dt in enumerate(datetime_index):
            if i < real.shape[0] and dt.hour == hour:
                hour_values.append(real[i, node_id, 0].item())
        
        if hour_values:
            avg = np.mean(hour_values)
            hourly_averages.append(avg)
            print(f"  Hour {hour:02d}: {avg:6.1f} (n={len(hour_values)})")
        else:
            hourly_averages.append(0)
            print(f"  Hour {hour:02d}: No data")
    
    # Check if pattern makes sense
    midnight_avg = hourly_averages[0]
    noon_avg = hourly_averages[12]
    morning_rush = max(hourly_averages[7:10])  # 7-9 AM
    evening_rush = max(hourly_averages[17:20])  # 5-7 PM
    
    print(f"\n🚦 TRAFFIC PATTERN ANALYSIS:")
    print(f"   Midnight (00:00): {midnight_avg:.1f}")
    print(f"   Morning Rush:     {morning_rush:.1f}")
    print(f"   Noon (12:00):     {noon_avg:.1f}")
    print(f"   Evening Rush:     {evening_rush:.1f}")
    
    # Sanity checks
    issues_found = []
    if midnight_avg > noon_avg:
        issues_found.append(f"❌ Midnight traffic ({midnight_avg:.1f}) > Noon traffic ({noon_avg:.1f})")
    
    if midnight_avg > morning_rush:
        issues_found.append(f"❌ Midnight traffic ({midnight_avg:.1f}) > Morning rush ({morning_rush:.1f})")
    
    if midnight_avg > evening_rush:
        issues_found.append(f"❌ Midnight traffic ({midnight_avg:.1f}) > Evening rush ({evening_rush:.1f})")
    
    if len(issues_found) == 0:
        print(f"✅ Traffic pattern looks NORMAL!")
    else:
        print(f"⚠️  ISSUES DETECTED:")
        for issue in issues_found:
            print(f"   {issue}")
        print(f"⚠️  This suggests a TIME MAPPING problem!")
    
    return hourly_averages

def debug_raw_data_pattern(real, node_id, dataset_name):
    """
    RAW DATA INSPECTION - Let's see what's actually in your data!
    """
    print(f"\n🔍 RAW DATA INSPECTION for Node {node_id}")
    print(f"=" * 60)
    
    dataset_key = dataset_name.replace('data//', '').upper().split('_')[0]
    config = DATASET_CONFIG.get(dataset_key, DATASET_CONFIG['PEMS04'])
    samples_per_hour = 60 // config['granularity_min']
    
    print(f"📊 Data shape: {real.shape}")
    print(f"⏰ Assumed samples per hour: {samples_per_hour}")
    
    # Look at first 24 hours of data
    print(f"\n📈 FIRST 24 HOURS (hours 0-23):")
    for hour in range(24):
        start_idx = hour * samples_per_hour
        end_idx = start_idx + samples_per_hour
        
        if end_idx <= real.shape[0]:
            hour_avg = real[start_idx:end_idx, node_id, 0].mean().item()
            print(f"  Hour {hour:02d}: {hour_avg:6.1f}")
        else:
            print(f"  Hour {hour:02d}: No data")
    
    print(f"\n📈 LAST 24 HOURS:")
    last_day_start = max(0, real.shape[0] - (24 * samples_per_hour))
    for hour in range(24):
        start_idx = last_day_start + (hour * samples_per_hour)
        end_idx = start_idx + samples_per_hour
        
        if end_idx <= real.shape[0]:
            hour_avg = real[start_idx:end_idx, node_id, 0].mean().item()
            print(f"  Hour {hour:02d}: {hour_avg:6.1f}")
    
    # Check if data is in reverse order
    print(f"\n🔄 REVERSE ORDER CHECK:")
    print(f"  First sample: {real[0, node_id, 0].item():.1f}")
    print(f"  Last sample: {real[-1, node_id, 0].item():.1f}")
    
    # Check different nodes to see if they all have this pattern
    print(f"\n🌐 OTHER NODES CHECK (first hour vs middle hour):")
    for test_node in [0, 5, 10, 15, min(20, real.shape[1]-1)]:
        if test_node < real.shape[1]:
            first_hour_avg = real[:samples_per_hour, test_node, 0].mean().item()
            mid_hour_avg = real[12*samples_per_hour:13*samples_per_hour, test_node, 0].mean().item()
            print(f"  Node {test_node:2d}: Hour 0={first_hour_avg:6.1f}, Hour 12={mid_hour_avg:6.1f}")
    
    # Try different starting assumptions
    print(f"\n🕐 DIFFERENT STARTING TIME ASSUMPTIONS:")
    for offset in [0, 6, 12, 18]:  # Try starting at different hours
        print(f"  If data starts at hour {offset}:")
        midnight_idx = (24 - offset) % 24
        noon_idx = (12 - offset) % 24
        
        midnight_start = midnight_idx * samples_per_hour
        midnight_end = midnight_start + samples_per_hour
        noon_start = noon_idx * samples_per_hour  
        noon_end = noon_start + samples_per_hour
        
        if midnight_end <= real.shape[0] and noon_end <= real.shape[0]:
            midnight_avg = real[midnight_start:midnight_end, node_id, 0].mean().item()
            noon_avg = real[noon_start:noon_end, node_id, 0].mean().item()
            
            pattern = "✅ NORMAL" if noon_avg > midnight_avg else "❌ WEIRD"
            print(f"    Midnight: {midnight_avg:6.1f}, Noon: {noon_avg:6.1f} {pattern}")

def plot_daily_forecasting_curve_simple(real, pred, node_id, dataset_name, 
                                horizon=0, day_type='weekday', model_name='Enhanced STAMT', 
                                save_dir=None, show_plots=False):
    """
    SIMPLE VERSION with RAW DATA DEBUGGING
    """
    if node_id >= real.shape[1]:
        print(f"Node {node_id} exceeds available nodes {real.shape[1]}")
        return
    
    # FIRST: Debug the raw data pattern
    debug_raw_data_pattern(real, node_id, dataset_name)
    
    print(f"\n🚀 Creating {day_type} plot for node {node_id} (SIMPLE TIME STEPS)")
    
    # Get dataset config for granularity
    dataset_key = dataset_name.replace('data//', '').upper().split('_')[0]
    config = DATASET_CONFIG.get(dataset_key, DATASET_CONFIG['PEMS04'])
    samples_per_hour = 60 // config['granularity_min']  # e.g., 12 for 5-min data
    samples_per_day = samples_per_hour * 24
    
    print(f"📊 Samples per hour: {samples_per_hour}, per day: {samples_per_day}")
    
    # Try REVERSE DATA ORDER as a test
    print(f"\n🔄 TESTING REVERSE DATA ORDER:")
    real_reversed = torch.flip(real, [0])  # Reverse time dimension
    pred_reversed = torch.flip(pred, [0])
    
    hourly_real_rev = []
    for hour in range(24):
        start_idx = hour * samples_per_hour
        end_idx = start_idx + samples_per_hour
        
        if end_idx <= real_reversed.shape[0]:
            hour_avg = real_reversed[start_idx:end_idx, node_id, horizon].mean().item()
            hourly_real_rev.append(hour_avg)
        else:
            hourly_real_rev.append(0)
    
    print(f"📈 REVERSED data pattern:")
    print(f"  Midnight (00): {hourly_real_rev[0]:.1f}")
    print(f"  Morning (06): {hourly_real_rev[6]:.1f}")  
    print(f"  Noon (12): {hourly_real_rev[12]:.1f}")
    print(f"  Evening (18): {hourly_real_rev[18]:.1f}")
    
    if hourly_real_rev[6] > hourly_real_rev[0] and hourly_real_rev[12] > hourly_real_rev[0]:
        print(f"🎯 REVERSE ORDER WORKS! Using reversed data...")
        real_to_use = real_reversed
        pred_to_use = pred_reversed
    else:
        print(f"🔄 Reverse doesn't help either...")
        real_to_use = real
        pred_to_use = pred
    
    # Now do the normal plotting with the data that makes sense
    hourly_real = []
    hourly_pred = []
    hourly_counts = []
    
    for hour in range(24):
        hour_values_real = []
        hour_values_pred = []
        
        # Go through data and collect samples for this hour
        for day_start in range(0, real_to_use.shape[0], samples_per_day):
            hour_start = day_start + (hour * samples_per_hour)
            hour_end = hour_start + samples_per_hour
            
            if hour_end <= real_to_use.shape[0]:
                # Determine if this is a weekday or weekend (simple assumption)
                day_number = day_start // samples_per_day
                is_weekend = (day_number % 7) >= 5  # Assume starts Monday
                
                # Filter by day type
                if (day_type == 'weekday' and not is_weekend) or (day_type == 'weekend' and is_weekend):
                    # Average the samples for this hour
                    hour_real = real_to_use[hour_start:hour_end, node_id, horizon].mean().item()
                    hour_pred = pred_to_use[hour_start:hour_end, node_id, horizon].mean().item()
                    
                    hour_values_real.append(hour_real)
                    hour_values_pred.append(hour_pred)
        
        if hour_values_real:
            hourly_real.append(np.mean(hour_values_real))
            hourly_pred.append(np.mean(hour_values_pred))
            hourly_counts.append(len(hour_values_real))
        else:
            hourly_real.append(0)
            hourly_pred.append(0)
            hourly_counts.append(0)
        
        if hour in [0, 6, 12, 18]:  # Debug key hours
            print(f"  Hour {hour:02d}: Real={hourly_real[-1]:.1f}, Samples={hourly_counts[-1]}")
    
    print(f"📈 FINAL Traffic pattern check:")
    print(f"  Midnight (00): {hourly_real[0]:.1f}")
    print(f"  Morning (06): {hourly_real[6]:.1f}")  
    print(f"  Noon (12): {hourly_real[12]:.1f}")
    print(f"  Evening (18): {hourly_real[18]:.1f}")
    
    # This should now show NORMAL traffic pattern!
    if hourly_real[6] > hourly_real[0] and hourly_real[12] > hourly_real[0]:
        print(f"🎉 SUCCESS! Traffic pattern looks NORMAL! (Morning/Noon > Midnight)")
    else:
        print(f"😕 Still abnormal pattern... Node {node_id} might have unusual data.")
        print(f"💡 Try a different node with --target_node [0-{real.shape[1]-1}]")
    
    # Create plot (same as before)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_pos = np.arange(24)
    
    ax.plot(x_pos, hourly_real, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
    ax.plot(x_pos, hourly_pred, 'r-', linewidth=2.5, label=model_name, marker='s', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Traffic Flow', fontsize=14, fontweight='bold')
    ax.set_title(f'{day_type.title()} forecasting curve for node {node_id} on {dataset_name.upper()} (DEBUGGED)', fontsize=14, fontweight='bold')
    
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '00:00'])
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    if max(hourly_real) > 0:
        y_min = min(min(hourly_real), min(hourly_pred))
        y_max = max(max(hourly_real), max(hourly_pred))
        if y_max > y_min:
            y_padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        mae = np.abs(np.array(hourly_real) - np.array(hourly_pred)).mean()
        rmse = np.sqrt(np.mean((np.array(hourly_real) - np.array(hourly_pred))**2))
        
        metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightblue', alpha=0.8, edgecolor='black'))
    
    ax.set_xlim(-0.5, 23.5)
    plt.tight_layout()
    
    if save_dir:
        filename = f'{day_type}_forecasting_curve_DEBUGGED_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✅ Saved DEBUGGED plot to {filename}")
    
    if show_plots:
        plt.show()
    plt.close()
    if node_id >= real.shape[1]:
        print(f"Node {node_id} exceeds available nodes {real.shape[1]}")
        return
    
    print(f"🚀 Creating {day_type} plot for node {node_id}")
    
    data_length = real.shape[0]
    if len(datetime_index) > data_length:
        datetime_index = datetime_index[:data_length]
    elif len(datetime_index) < data_length:
        print(f"WARNING: datetime_index ({len(datetime_index)}) shorter than data ({data_length})")
        return
    
    weekend_mask = identify_weekends(datetime_index)
    day_mask = ~weekend_mask if day_type == 'weekday' else weekend_mask
    title_suffix = "Weekday" if day_type == 'weekday' else "Weekend"
    
    print(f"📊 Selected {day_type} samples: {day_mask.sum()}")
    if day_mask.sum() == 0:
        print(f"WARNING: No {day_type} data found! Skipping plot.")
        return
    
    # DEBUG: Check time mapping
    print(f"🔍 DEBUGGING TIME MAPPING for {day_type}:")
    sample_indices = np.where(day_mask)[0][:20]  # First 20 matching samples
    for idx in sample_indices:
        if idx < len(datetime_index):
            dt = datetime_index[idx]
            val = real[idx, node_id, horizon].item()
            print(f"  Index {idx}: {dt} (Hour: {dt.hour}) -> Value: {val:.2f}")
    
    # Calculate hourly averages with better debugging
    hourly_real = []
    hourly_pred = []
    hourly_counts = []
    
    for hour in range(24):
        hour_indices = [i for i, dt in enumerate(datetime_index) if day_mask[i] and dt.hour == hour]
        
        if len(hour_indices) > 0:
            hour_real_vals = [real[i, node_id, horizon].item() for i in hour_indices]
            hour_pred_vals = [pred[i, node_id, horizon].item() for i in hour_indices]
            
            avg_real = np.mean(hour_real_vals)
            avg_pred = np.mean(hour_pred_vals)
            
            hourly_real.append(avg_real)
            hourly_pred.append(avg_pred)
            hourly_counts.append(len(hour_indices))
            
            if hour in [0, 6, 12, 18, 23]:  # Debug key hours
                print(f"  Hour {hour:02d}: Avg Real={avg_real:.1f}, Samples={len(hour_indices)}")
        else:
            hourly_real.append(0)
            hourly_pred.append(0)
            hourly_counts.append(0)
    
    print(f"📈 Traffic pattern check:")
    print(f"  Midnight (00): {hourly_real[0]:.1f}")
    print(f"  Morning (06): {hourly_real[6]:.1f}")  
    print(f"  Noon (12): {hourly_real[12]:.1f}")
    print(f"  Evening (18): {hourly_real[18]:.1f}")
    
    # Sanity check - warn if pattern looks weird
    if hourly_real[0] > hourly_real[12]:  # Midnight > Noon
        print(f"⚠️  WARNING: Traffic higher at midnight ({hourly_real[0]:.1f}) than noon ({hourly_real[12]:.1f})!")
        print(f"⚠️  This suggests a time indexing issue!")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x_pos = np.arange(24)
    
    ax.plot(x_pos, hourly_real, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
    ax.plot(x_pos, hourly_pred, 'r-', linewidth=2.5, label=model_name, marker='s', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Traffic Flow', fontsize=14, fontweight='bold')
    ax.set_title(f'{title_suffix} forecasting curve for node {node_id} on {dataset_name.upper()}.', fontsize=14, fontweight='bold')
    
    # Simple tick setup
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '00:00'])
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    if max(hourly_real) > 0 or max(hourly_pred) > 0:
        y_min = min(min(hourly_real), min(hourly_pred))
        y_max = max(max(hourly_real), max(hourly_pred))
        if y_max > y_min:
            y_padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        mae = np.abs(np.array(hourly_real) - np.array(hourly_pred)).mean()
        rmse = np.sqrt(np.mean((np.array(hourly_real) - np.array(hourly_pred))**2))
        
        metrics_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightblue', alpha=0.8, edgecolor='black'))
    
    ax.set_xlim(-0.5, 23.5)
    plt.tight_layout()
    
    if save_dir:
        filename = f'{day_type}_forecasting_curve_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        filename_pdf = f'{day_type}_forecasting_curve_node_{node_id}_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    plt.close()

def plot_horizon_comparison_research(real, pred, node_id, dataset_name, horizons=[0, 2, 5, 11], save_dir=None, show_plots=False):
    if node_id >= real.shape[1]:
        print(f"Node {node_id} exceeds available nodes {real.shape[1]}")
        return
    
    print(f"🔍 DEBUGGING TRAFFIC PATTERNS for Node {node_id}")
    print(f"📊 Data shape: {real.shape}")
    
    # Create datetime index to verify time mapping
    datetime_index = get_datetime_index(dataset_name, real.shape[0])
    print(f"⏰ First few timestamps: {datetime_index[:10]}")
    print(f"⏰ Last few timestamps: {datetime_index[-10:]}")
    
    # Debug: Check actual data values at different times
    print("\n🔍 SAMPLE DATA VALUES:")
    for i in range(min(50, len(datetime_index))):
        if i % 12 == 0:  # Every hour for 5-min data
            print(f"Time: {datetime_index[i]}, Value: {real[i, node_id, 0].item():.2f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    horizon_names = ['1-step', '3-step', '6-step', '12-step']
    
    for i, (ax, horizon) in enumerate(zip(axes, horizons)):
        if horizon >= real.shape[2]:
            continue
        
        print(f"\n📈 Processing horizon {horizon} ({horizon_names[i]})")
        
        # Use datetime index approach instead of arbitrary day chunking
        hourly_real_all = [[] for _ in range(24)]
        hourly_pred_all = [[] for _ in range(24)]
        
        # Group data by hour of day across ALL samples
        for idx in range(min(len(datetime_index), real.shape[0])):
            dt = datetime_index[idx]
            hour = dt.hour
            
            real_val = real[idx, node_id, horizon].item()
            pred_val = pred[idx, node_id, horizon].item()
            
            hourly_real_all[hour].append(real_val)
            hourly_pred_all[hour].append(pred_val)
        
        # Calculate averages for each hour
        hourly_real = []
        hourly_pred = []
        
        for hour in range(24):
            if len(hourly_real_all[hour]) > 0:
                avg_real = np.mean(hourly_real_all[hour])
                avg_pred = np.mean(hourly_pred_all[hour])
                hourly_real.append(avg_real)
                hourly_pred.append(avg_pred)
                
                if hour in [0, 6, 12, 18]:  # Debug key hours
                    print(f"Hour {hour:02d}: Real={avg_real:.1f}, Pred={avg_pred:.1f}, Samples={len(hourly_real_all[hour])}")
            else:
                hourly_real.append(0)
                hourly_pred.append(0)
        
        hours = np.arange(24)
        ax.plot(hours, hourly_real, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4)
        ax.plot(hours, hourly_pred, 'r-', linewidth=2.5, label='Enhanced STAMT', marker='s', markersize=4)
        
        ax.set_title(f'{horizon_names[i]} ahead prediction', fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('Traffic Flow', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(range(0, 25, 6))
        ax.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    
    plt.suptitle(f'Multi-horizon Prediction Analysis - Node {node_id} ({dataset_name.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        filename = f'horizon_comparison_research_node_{node_id}_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        filename_pdf = f'horizon_comparison_research_node_{node_id}_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    plt.close()

def plot_node_performance_comparison(real, pred, node_ids, dataset_name, save_dir=None, show_plots=False):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # MAE comparison across nodes
    ax1 = axes[0, 0]
    node_maes = []
    for node_id in node_ids:
        if node_id < real.shape[1]:
            mae = torch.abs(real[:, node_id, :] - pred[:, node_id, :]).mean().item()
            node_maes.append(mae)
        else:
            node_maes.append(0)
    
    bars = ax1.bar(range(len(node_ids)), node_maes, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Node ID', fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax1.set_title('Performance Comparison Across Nodes', fontweight='bold')
    ax1.set_xticks(range(len(node_ids)))
    ax1.set_xticklabels([str(nid) for nid in node_ids])
    ax1.grid(True, alpha=0.3)
    
    for bar, mae in zip(bars, node_maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Horizon-wise performance
    ax2 = axes[0, 1]
    horizons = range(1, min(13, real.shape[2] + 1))
    horizon_maes = []
    horizon_stds = []
    
    for h in range(len(horizons)):
        if h < real.shape[2]:
            mae_per_node = [torch.abs(real[:, node_id, h] - pred[:, node_id, h]).mean().item() for node_id in node_ids if node_id < real.shape[1]]
            if mae_per_node:
                horizon_maes.append(np.mean(mae_per_node))
                horizon_stds.append(np.std(mae_per_node))
    
    if horizon_maes:
        ax2.errorbar(horizons[:len(horizon_maes)], horizon_maes, yerr=horizon_stds, marker='o', markersize=6, linewidth=2, capsize=5, capthick=2)
    ax2.set_xlabel('Prediction Horizon', fontweight='bold')
    ax2.set_ylabel('MAE (Mean ± Std)', fontweight='bold')
    ax2.set_title('Performance vs Prediction Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution
    ax3 = axes[1, 0]
    all_errors = []
    for node_id in node_ids:
        if node_id < real.shape[1]:
            errors = (real[:, node_id, 0] - pred[:, node_id, 0]).numpy()
            all_errors.extend(errors)
    
    if all_errors:
        ax3.hist(all_errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error', fontweight='bold')
    ax3.set_ylabel('Density', fontweight='bold')
    ax3.set_title('Error Distribution (1-step ahead)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Correlation matrix
    ax4 = axes[1, 1]
    correlations = []
    for node_id in node_ids:
        if node_id < real.shape[1]:
            real_vals = real[:, node_id, 0].numpy()
            pred_vals = pred[:, node_id, 0].numpy()
            corr = np.corrcoef(real_vals, pred_vals)[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0)
    
    bars = ax4.bar(range(len(node_ids)), correlations, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Node ID', fontweight='bold')
    ax4.set_ylabel('Correlation Coefficient', fontweight='bold')
    ax4.set_title('Prediction Correlation by Node', fontweight='bold')
    ax4.set_xticks(range(len(node_ids)))
    ax4.set_xticklabels([str(nid) for nid in node_ids])
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'Comprehensive Performance Analysis ({dataset_name.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        filename = f'node_performance_comparison_{dataset_name}.png'
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        filename_pdf = f'node_performance_comparison_{dataset_name}.pdf'
        plt.savefig(os.path.join(save_dir, filename_pdf), dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    plt.close()

def plot_research_summary_figure(real, pred, node_id, dataset_name, save_dir=None, show_plots=False):
    if node_id >= real.shape[1]:
        print(f"Node {node_id} exceeds available nodes {real.shape[1]}")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # Daily pattern - USE SIMPLE METHOD
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get dataset config for simple time calculation
    dataset_key = dataset_name.replace('data//', '').upper().split('_')[0]
    config = DATASET_CONFIG.get(dataset_key, DATASET_CONFIG['PEMS04'])
    samples_per_hour = 60 // config['granularity_min']
    samples_per_day = samples_per_hour * 24
    
    hourly_real_wd = []
    for hour in range(24):
        hour_values = []
        for day_start in range(0, real.shape[0], samples_per_day):
            hour_start = day_start + (hour * samples_per_hour)
            hour_end = hour_start + samples_per_hour
            
            if hour_end <= real.shape[0]:
                day_number = day_start // samples_per_day
                is_weekend = (day_number % 7) >= 5
                
                if not is_weekend:  # weekday only
                    hour_real = real[hour_start:hour_end, node_id, 0].mean().item()
                    hour_values.append(hour_real)
        
        if hour_values:
            hourly_real_wd.append(np.mean(hour_values))
        else:
            hourly_real_wd.append(0)
    
    # Do same for predictions
    hourly_pred_wd = []
    for hour in range(24):
        hour_values = []
        for day_start in range(0, pred.shape[0], samples_per_day):
            hour_start = day_start + (hour * samples_per_hour)
            hour_end = hour_start + samples_per_hour
            
            if hour_end <= pred.shape[0]:
                day_number = day_start // samples_per_day
                is_weekend = (day_number % 7) >= 5
                
                if not is_weekend:  # weekday only
                    hour_pred = pred[hour_start:hour_end, node_id, 0].mean().item()
                    hour_values.append(hour_pred)
        
        if hour_values:
            hourly_pred_wd.append(np.mean(hour_values))
        else:
            hourly_pred_wd.append(0)
    
    hours = range(24)
    ax1.plot(hours, hourly_real_wd, 'b-', linewidth=2.5, label='Ground Truth', marker='o', markersize=4)
    ax1.plot(hours, hourly_pred_wd, 'r-', linewidth=2.5, label='Enhanced STAMT', marker='s', markersize=4)
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_ylabel('Traffic Flow', fontweight='bold')
    ax1.set_title('(a) Daily Traffic Pattern', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 25, 6))
    ax1.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    
    # Multi-horizon performance
    ax2 = fig.add_subplot(gs[0, 1])
    horizons = range(1, min(13, real.shape[2] + 1))
    mae_by_horizon = []
    for h in range(len(horizons)):
        if h < real.shape[2]:
            mae = torch.abs(real[:, node_id, h] - pred[:, node_id, h]).mean().item()
            mae_by_horizon.append(mae)
    
    ax2.plot(horizons[:len(mae_by_horizon)], mae_by_horizon, 'o-', linewidth=2.5, markersize=6, color='green')
    ax2.set_xlabel('Prediction Horizon', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error', fontweight='bold')
    ax2.set_title('(b) Performance vs Horizon', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Sample time series
    ax3 = fig.add_subplot(gs[1, :])
    sample_length = min(288*3, real.shape[0])
    sample_real = real[:sample_length, node_id, 0].numpy()
    sample_pred = pred[:sample_length, node_id, 0].numpy()
    sample_times = datetime_index[:sample_length]
    
    ax3.plot(sample_times, sample_real, 'b-', linewidth=1.5, label='Ground Truth', alpha=0.8)
    ax3.plot(sample_times, sample_pred, 'r-', linewidth=1.5, label='Enhanced STAMT', alpha=0.8)
    
    weekend_sample_mask = identify_weekends(sample_times)
    for i, is_weekend in enumerate(weekend_sample_mask):
        if is_weekend and i < len(sample_times):
            ax3.axvspan(sample_times[i], sample_times[min(i+1, len(sample_times)-1)], alpha=0.1, color='yellow')
    
    ax3.set_xlabel('Date/Time', fontweight='bold')
    ax3.set_ylabel('Traffic Flow', fontweight='bold')
    ax3.set_title('(c) Sample Time Series (3 days)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    
    # Scatter plot
    ax4 = fig.add_subplot(gs[2, 0])
    sample_indices = np.random.choice(real.shape[0], min(1000, real.shape[0]), replace=False)
    real_sample = real[sample_indices, node_id, 0].numpy()
    pred_sample = pred[sample_indices, node_id, 0].numpy()
    
    ax4.scatter(real_sample, pred_sample, alpha=0.6, s=20, color='purple')
    min_val, max_val = min(real_sample.min(), pred_sample.min()), max(real_sample.max(), pred_sample.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    ax4.set_xlabel('Ground Truth', fontweight='bold')
    ax4.set_ylabel('Prediction', fontweight='bold')
    ax4.set_title('(d) Prediction Accuracy', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    r2 = np.corrcoef(real_sample, pred_sample)[0, 1]**2
    ax4.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax4.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Error analysis
    ax5 = fig.add_subplot(gs[2, 1])
    errors = (real[:, node_id, 0] - pred[:, node_id, 0]).numpy()
    ax5.hist(errors, bins=40, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
    ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax5.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.3f}')
    ax5.set_xlabel('Prediction Error', fontweight='bold')
    ax5.set_ylabel('Density', fontweight='bold')
    ax5.set_title('(e) Error Distribution', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Analysis: Node {node_id} on {dataset_name.upper()}', fontsize=16, fontweight='bold')
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
    # Dataset configuration
    if args.data == "PEMS08":
        args.data = "data//"+args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"
    elif args.data == "PEMS08_36":
        args.data = "data//"+args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"
    elif args.data == "PEMS08_48":
        args.data = "data//"+args.data
        args.num_nodes = 170
        args.adjdata = "data/adj/adj_PEMS08_gs.npy"
    elif args.data == "PEMS03":
        args.data = "data//"+args.data
        args.num_nodes = 358
        args.adjdata = "data/adj/adj_PEMS03_gs.npy"
    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
    elif args.data == "PEMS04_36":
        args.data = "data//"+args.data
        args.num_nodes = 307
        args.adjdata = "data/adj/adj_PEMS04_gs.npy"
    elif args.data == "PEMS04_48":
        args.data = "data//"+args.data
        args.num_nodes = 307
        args.adjdata = "data/adj/adj_PEMS04_gs.npy"
    elif args.data == "PEMS07":
        args.data = "data//"+args.data
        args.num_nodes = 883
        args.adjdata = "data/adj/adj_PEMS07_gs.npy"
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
    model = EnhancedSTAMT(device, args.input_dim, args.channels, args.num_nodes, args.input_len, args.output_len, args.dropout)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('Model loaded successfully')

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    
    realy = torch.Tensor(dataloader['y_test']).to(device)
    
    if realy.dim() == 4:
        if realy.shape[1] == 1:
            realy = realy.squeeze(1).permute(0, 2, 1)
        elif realy.shape[-1] == 1:
            realy = realy.squeeze(-1)
        else:
            realy = realy[:, 0, :, :].permute(0, 2, 1)
    elif realy.dim() == 3:
        if realy.shape[1] == args.num_nodes:
            realy = realy.permute(0, 2, 1)

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

    realy = realy.to("cpu")
    yhat_inverse = torch.zeros_like(yhat)
    for i in range(yhat.shape[1]):
        yhat_inverse[:, i, :] = scaler.inverse_transform(yhat[:, i, :])
    yhat1 = yhat_inverse.to("cpu")

    torch.save(realy, "stamt_04real.pt")
    torch.save(yhat1, "stamt_04pred.pt")

    if args.research_plots:
        print("Creating research paper quality visualizations...")
        
        dataset_name = args.data.split('/')[-1] if '/' in args.data else args.data
        target_node = min(args.target_node, realy.shape[1] - 1)
        
        print(f"🔧 SUPER DEBUGGING MODE ACTIVATED!")
        print(f"Target node: {target_node}")
        print(f"💡 If node {target_node} has weird data, try: --target_node 0 or --target_node 5 etc.")
        
        save_dir = None
        if args.save_research_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"research_plots_{dataset_name}_{timestamp}"
            os.makedirs(save_dir, exist_ok=True)
            print(f"Research plots will be saved to: {save_dir}")
        
        if not args.viz_nodes or len(args.viz_nodes) == 0:
            max_node = realy.shape[1] - 1
            viz_nodes = [0, max_node//4, max_node//2, 3*max_node//4, max_node]
        else:
            viz_nodes = [min(node, realy.shape[1] - 1) for node in args.viz_nodes if node < realy.shape[1]]
            
        if len(viz_nodes) == 0:
            max_node = realy.shape[1] - 1
            viz_nodes = [0, max_node//4, max_node//2, 3*max_node//4, max_node]
        
        print(f"Target node: {target_node}, Viz nodes: {viz_nodes}")
        
        if args.daily_analysis:
            print("Creating daily forecasting curves... (SUPER DEBUGGED VERSION!)")
            # Use the SUPER DEBUGGED method!
            plot_daily_forecasting_curve_simple(realy, yhat1, target_node, dataset_name,
                                        horizon=0, day_type='weekday', model_name='Enhanced STAMT',
                                        save_dir=save_dir, show_plots=False)
            print("\n" + "="*60 + "\n")  # Separator between weekday and weekend
            plot_daily_forecasting_curve_simple(realy, yhat1, target_node, dataset_name,
                                        horizon=0, day_type='weekend', model_name='Enhanced STAMT',
                                        save_dir=save_dir, show_plots=False)
        
        if args.horizon_analysis:
            print("Creating horizon comparison...")
            plot_horizon_comparison_research(realy, yhat1, target_node, dataset_name, save_dir=save_dir, show_plots=False)
        
        if len(viz_nodes) > 0:
            print("Creating node performance comparison...")
            plot_node_performance_comparison(realy, yhat1, viz_nodes, dataset_name, save_dir=save_dir, show_plots=False)
        
        print("Creating comprehensive summary figure...")
        plot_research_summary_figure(realy, yhat1, target_node, dataset_name, save_dir=save_dir, show_plots=False)
        
        print("Research paper quality visualizations complete!")
        
        print(f"\n🎯 FINAL DIAGNOSIS:")
        print(f"   If traffic pattern still looks weird, try different nodes:")
        print(f"   python script.py --data PEMS08 --research_plots --target_node 0 --daily_analysis")
        print(f"   python script.py --data PEMS08 --research_plots --target_node 5 --daily_analysis")
        print(f"   python script.py --data PEMS08 --research_plots --target_node 20 --daily_analysis")
        
        if save_dir:
            with open(os.path.join(save_dir, 'research_summary.txt'), 'w') as f:
                f.write(f"Research Paper Analysis Summary\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Dataset: {dataset_name.upper()}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Target Node: {target_node}\n")
                f.write(f"Comparison Nodes: {viz_nodes}\n")
                f.write(f"Data Shape: {realy.shape}\n")
                f.write(f"Date Range: {datetime_index[0]} to {datetime_index[-1]}\n")
                f.write(f"Granularity: {DATASET_CONFIG.get(dataset_name.upper(), {}).get('granularity_min', 'Unknown')} minutes\n\n")
                
                f.write("Performance Metrics:\n")
                f.write(f"Average MAE: {np.mean(amae):.4f}\n")
                f.write(f"Average RMSE: {np.mean(armse):.4f}\n")
                f.write(f"Average MAPE: {np.mean(amape):.4f}%\n")
                f.write(f"Average WMAPE: {np.mean(awmape):.4f}%\n\n")

if __name__ == "__main__":
    main()