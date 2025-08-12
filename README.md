# D-SELD: Dataset-Scalable Exemplar LCA-Decoder

A PyTorch implementation of innovative Exemplar Locally Competitive Algorithm (LCA) across multiple datasets using various CNN architectures.

## Features

- **Multiple CNN Models**: Support for ResNet152, DenseNet121, VGG16, EfficientNet, MobileNet, and Inception
- **Multi-Dataset Support**: CIFAR-10, CIFAR-100, ImageNet, and MNIST
- **LCA Integration**: Locally Competitive Algorithm for sparse feature representation
- **Interactive CLI**: User-friendly command-line interface for model and dataset selection
- **GPU/CPU Compatible**: Automatic device detection with memory optimization

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Torchvision
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sanaz-Tak/D-SELD.git
cd D-SELD
```

2. Install dependencies:
```bash
pip install torch torchvision numpy
```

## Usage

### Quick Start

Run the interactive multi-dataset script:
```bash
python interactive_runner.py
```

This will guide you through:
- Dataset selection (CIFAR-10, CIFAR-100, ImageNet, MNIST)
- Model selection (CNN architectures)
- Automatic execution with optimal parameters

### Direct Execution

Run the main script directly with custom parameters:
```bash
python src/lca_feature_extractor.py --dataset cifar10 --model resnet152
```

### Parameters

- `--dataset`: Dataset name (cifar10, cifar100, imagenet, mnist)
- `--model`: Model architecture (resnet152, densenet121, vgg16, efficientnet, mobilenet, inception, none)
- `--data_path`: Path to dataset directory
- `--batch_size`: Batch size for processing (default: 50)
- `--max_samples`: Maximum samples to process (default: 50)
- `--dictionary_num`: Number of dictionary elements (default: 400)
- `--neuron_iter`: Number of neuron iterations (default: 50)
- `--lr_neuron`: Learning rate for neurons (default: 0.01)
- `--landa`: Sparsity parameter (default: 2)

## Dataset Preparation

### CIFAR-10/100 and MNIST
Data will be automatically downloaded to `./data_cifar10/`, `./data_cifar100/`, and `./data_mnist/`.

### ImageNet
Organize your ImageNet data as follows:
```
data_imagenet/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Architecture

The project consists of two main components:

1. **CNN Feature Extraction**: Various pre-trained CNN models extract hierarchical features from input images
2. **Local Competition Algorithm (LCA)**: Implements sparse coding for feature representation

## Project Structure

- **`src/lca_feature_extractor.py`**: Core D-SELD implementation and main execution pipeline
- **`interactive_runner.py`**: Interactive interface for dataset and model selection
- **`data_*/`**: Dataset directories (automatically created)

### LCA Algorithm

The LCA algorithm provides:
- Sparse feature representation
- Dictionary learning
- Neuron competition mechanism
- Configurable sparsity constraints

## Research Applications

This implementation is suitable for:
- Sparse coding studies
- Computer Vision and CNN feature analysis
- Transfer learning experiments
- Neuromorphic computing research



## Citation

If you use this code in your research, please cite:

```bibtex
@article{takaghaj2024dseld,
  title={D-SELD: Dataset-Scalable Exemplar LCA-Decoder},
  author={Sanaz M. Takaghaj and Jack Sampson},
  journal={Neuromorphic Computing and Engineering Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Issues

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/Sanaz-Tak/D-SELD/issues) page.

## Acknowledgments

- PyTorch team for the deep learning framework
- Torchvision for pre-trained models and datasets
- Neuromorphic computing community for foundational LCA research