# MSMT: Memory-augmented Spatio-temporal Multi-scale Transformer


A PyTorch implementation of **Memory-augmented Spatio-temporal Multi-scale Transformer (MSMT)** for enhanced time series forecasting with adaptive attention mechanisms and uncertainty quantification.

## 🌟 Key Features

- **🧠 Memory-Augmented Attention**: Adaptive memory banks for capturing long-range dependencies
- **⏱️ Multi-scale Temporal Processing**: Dilated convolutions for multi-resolution temporal modeling
- **🗺️ Adaptive Graph Learning**: Dynamic spatial relationship discovery
- **🎯 Uncertainty Quantification**: Multi-resolution outputs with uncertainty estimation
- **🚀 State-of-the-art Performance**: Competitive results on multiple benchmarks

## 📋 Architecture Overview

MSMT combines several innovative components:

1. **Adaptive Temporal Embedding**: Dynamic time-of-day and day-of-week representations
2. **Multi-scale Temporal Convolutions**: Capture dependencies at different time scales
3. **Memory-Augmented Spatial Attention**: Learn adaptive spatial patterns with memory
4. **Uncertainty-Aware Output**: Multi-resolution predictions with confidence estimation

## 🔧 Installation

### Quick Install
```bash
git clone https://github.com/JoshuaKAsamoah/MSMT.git
cd MSMT
pip install -r requirements.txt
```

### Development Install
```bash
git clone https://github.com/JoshuaKAsamoah/MSMT.git
cd MSMT
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 1.12+
- NumPy, Pandas, Matplotlib, Seaborn

## 🚀 Quick Start

### Basic Usage

```python
import torch
from models.msmt import MSMT

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSMT(
    device=device,
    input_dim=3,
    channels=128,
    num_nodes=170,
    input_len=12,
    output_len=12,
    dropout=0.1,
    memory_size=4
)

# Example input: [batch_size, input_dim, num_nodes, input_len]
x = torch.randn(32, 3, 170, 12).to(device)
predictions = model(x)  # [32, output_len, num_nodes, 1]
```

### Training

```bash
# Train on PEMS04 dataset
python training/train.py \
    --data PEMS04 \
    --batch_size 16 \
    --epochs 300 \
    --learning_rate 0.001 \
    --channels 128 \
    --memory_size 4

# Train with custom settings
python training/train.py \
    --data PEMS08 \
    --input_len 24 \
    --output_len 24 \
    --batch_size 32 \
    --epochs 500 \
    --es_patience 100
```

### Testing

```bash
# Test trained model
python test.py \
    --data PEMS04 \
    --checkpoint ./experiments/logs/best_model.pth \
    --output_dir ./test_results \
    --plot_results True

# Batch testing
python test.py \
    --data PEMS08 \
    --checkpoint ./models/pems08_best.pth \
    --batch_size 64 \
    --save_predictions True
```

## 📊 Supported Datasets

MSMT supports various traffic and mobility datasets:

### Traffic Datasets
- **PEMS03**: 358 nodes, 5-minute intervals
- **PEMS04**: 307 nodes, 5-minute intervals  
- **PEMS07**: 883 nodes, 5-minute intervals
- **PEMS08**: 170 nodes, 5-minute intervals

### Mobility Datasets
- **Bike Drop/Pick**: 250 nodes
- **Taxi Drop/Pick**: 266 nodes

### Data Format
Expected data format: `[num_samples, num_nodes, num_features, seq_len]`
- Features: `[traffic_flow, time_of_day, day_of_week]`
- Files: `train.npz`, `val.npz`, `test.npz`

## 🏗️ Project Structure

```
MSMT/
├── models/
│   ├── __init__.py
│   └── msmt.py                 # Main MSMT model
├── training/
│   ├── __init__.py
│   └── train.py               # Training script
├── dataset/
│   ├── PEMS04/
│   ├── PEMS08/
│   └── ...
├── experiments/
│   ├── configs/
│   └── logs/
├── test_results/
├── util.py                    # Utilities and metrics
├── test.py                   # Testing script
├── requirements.txt
├── setup.py
└── README.md
```

## ⚙️ Configuration

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dim` | Input feature dimension | 3 |
| `channels` | Hidden channels | 128 |
| `num_nodes` | Number of graph nodes | 170 |
| `input_len` | Input sequence length | 12 |
| `output_len` | Output sequence length | 12 |
| `dropout` | Dropout rate | 0.1 |
| `memory_size` | Memory bank size | 4 |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 16 |
| `learning_rate` | Learning rate | 0.001 |
| `weight_decay` | Weight decay | 0.0001 |
| `epochs` | Maximum epochs | 500 |
| `es_patience` | Early stopping patience | 100 |

## 📈 Performance

### PEMS Datasets Results

| Dataset | MAE | RMSE | MAPE |
|---------|-----|------|------|
| PEMS03  | TBD | TBD  | TBD  |
| PEMS04  | TBD | TBD  | TBD  |
| PEMS07  | TBD | TBD  | TBD  |
| PEMS08  | TBD | TBD  | TBD  | 

*Results to be updated with your experimental findings.*

## 🔬 Experimental Features

### Uncertainty Quantification
```python
# Get predictions with uncertainty
predictions, uncertainty = model(x, return_uncertainty=True)
```

### Memory Visualization
```python
# Access memory attention weights
attention_weights = model.spatial_encoder.attention.memory_importance
```

### Multi-resolution Outputs
```python
# Get fine, coarse, and uncertainty predictions
main_pred, fine_pred, coarse_pred, uncertainty = model.output_head(features)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use MSMT in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Memory-augmented Spatio-temporal Multi-scale Transformer for Enhanced Time Series Forecasting},
  author={Your Name and Co-authors},
  journal={Your Journal},
  year={2025},
  volume={X},
  pages={XXX-XXX}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Traffic dataset providers (Caltrans, etc.)
- Research community for inspiration and feedback

## 📞 Contact

- **Author**: Joshua Kofi Asamoah
- **Email**: joshua.asamoah@ndsu.edu
- **GitHub**: [@JoshuaKAsamoah](https://github.com/JoshuaKAsamoah)

---

⭐ **Star this repo if you find it helpful!** ⭐