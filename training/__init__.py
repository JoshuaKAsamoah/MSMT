"""
MSMT: Memory-augmented Spatio-temporal Multi-scale Transformer

A PyTorch implementation of MSMT for spatio-temporal forecasting with:
- Adaptive temporal embeddings
- Multi-scale temporal convolutions  
- Memory-augmented spatial attention
- Uncertainty-aware multi-resolution outputs

Author: Joshua Kofi Asamoah
License: MIT
"""

from .models import MSMT
from .training import MSMTTrainer
import util

__version__ = "1.0.0"
__author__ = "Joshua Kofi Asamoah"
__email__ = "joshua.asamoah@ndsu.edu"
__license__ = "MIT"

__all__ = [
    "MSMT",
    "MSMTTrainer",
    "util"
]