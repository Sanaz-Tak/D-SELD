# D-SELD: Dataset-Scalable Exemplar LCA-Decoder

A PyTorch implementation of innovative Exemplar Locally Competitive Algorithm (LCA) across multiple datasets using various CNN architectures.

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
- `--batch_size`: Batch size for processing (default: 16)
- `--max_test_samples`: Maximum samples to test (default: 10000)
- `--dictionary_num`: Number of dictionary elements (default: 50000)
- `--neuron_iter`: Number of neuron iterations (default: 100)
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


## Issues

If you encounter any issues, please report them on the [GitHub Issues](https://github.com/Sanaz-Tak/D-SELD/issues) page.
