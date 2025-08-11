"""
Default configuration for sparse coding with LCA.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DefaultConfig:
    """
    Default configuration for LCA models.
    
    This class provides default parameters for both MNIST and CNN LCA models.
    """
    
    # LCA parameters (from DSELD paper)
    dictionary_num: int = 50000  # M = 50K from paper
    neuron_iter: int = 100       # k = 100 time steps from paper
    lr_neuron: float = 0.1
    lr_dictionary: float = 0.01
    landa: float = 2.0          # λ = 2 threshold from paper
    threshold_type: str = 'soft'
    rectify: bool = True
    tau: float = 100.0          # τ = 100 leakage from paper
    
    # Device
    device: str = 'cpu'
    
    # MNIST specific
    mnist_feature_size: int = 28
    
    # CNN specific
    cnn_feature_layer: str = 'layer4'
    cnn_landa: float = 0.1
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    
    # Data parameters
    data_dir: str = './data'
    download_data: bool = True
    
    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    
    def get_mnist_config(self) -> dict:
        """Get configuration for MNIST LCA model."""
        return {
            'feature_size': self.mnist_feature_size,
            'dictionary_num': self.dictionary_num,
            'neuron_iter': self.neuron_iter,
            'lr_neuron': self.lr_neuron,
            'lr_dictionary': self.lr_dictionary,
            'landa': self.landa,
            'threshold_type': self.threshold_type,
            'rectify': self.rectify,
            'device': self.device
        }
    
    def get_cnn_config(self) -> dict:
        """Get configuration for CNN LCA model."""
        return {
            'dictionary_num': self.dictionary_num,
            'feature_layer': self.cnn_feature_layer,
            'neuron_iter': self.neuron_iter,
            'lr_neuron': self.lr_neuron,
            'lr_dictionary': self.lr_dictionary,
            'landa': self.cnn_landa,
            'threshold_type': self.threshold_type,
            'device': self.device
        } 