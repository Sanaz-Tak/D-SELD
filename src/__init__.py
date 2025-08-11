"""
D-SELD: Dataset-Scalable Exemplar LCA-Decoder

Official implementation of D-SELD, a novel approach for sparse coding using the 
Locally Competitive Algorithm (LCA) with exemplar-based decoding.
"""

__version__ = "1.0.0"
__author__ = "Sanaz Mahmoodi Takaghaj and Jack Sampson"
__email__ = "sxm788@psu.edu"

from .lca.core import LCA
from .models.mnist_lca import MNISTLCA
from .models.cnn_lca import CNNLCA

__all__ = ["LCA", "MNISTLCA", "CNNLCA"] 