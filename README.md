# D-SELD: Dataset-Scalable Exemplar LCA-Decoder

A PyTorch implementation of Locally Competitive Algorithm (LCA) for sparse feature extraction across multiple datasets using various CNN architectures.

## Features

- **Multiple Datasets**: Support for CIFAR-10, CIFAR-100, ImageNet, and MNIST
- **Various CNN Models**: ResNet152, DenseNet121, VGG16, EfficientNet, MobileNet, Inception
- **Sparse Coding**: LCA implementation for learning sparse representations
- **GPU/CPU Compatible**: Automatic device detection with memory management
- **Interactive Interface**: Easy-to-use command-line interface

## Requirements

```bash
pip install torch torchvision numpy
```

## Quick Start

### Interactive Mode

Run the interactive interface to select dataset and model:

```bash
python interactive_runner.py
```

### Command Line Mode

Run directly with specific parameters:

```bash
python src/lca_feature_extractor.py --dataset cifar10 --model resnet152
```

## Usage

### Available Datasets

1. **CIFAR-10**: 10 classes, automatically downloaded
2. **CIFAR-100**: 100 classes, automatically downloaded  
3. **ImageNet**: 1000 classes, requires manual setup
4. **MNIST**: 10 classes, automatically downloaded

### Available Models

- `resnet152`: ResNet-152 (2048 features)
- `densenet121`: DenseNet-121 (50176 features)
- `vgg16`: VGG-16 (512 features)
- `efficientnet`: EfficientNet-B0 (1280 features)
- `mobilenet`: MobileNet-V2 (62720 features)
- `inception`: Inception-V3 (2048 features)
- `none`: Raw image data (MNIST only, 784 features)

### Command Line Arguments

```bash
python src/lca_feature_extractor.py [OPTIONS]

Options:
  --dataset {cifar10,cifar100,imagenet,mnist}  Dataset to use (default: cifar10)
  --model {resnet152,densenet121,vgg16,efficientnet,mobilenet,inception,none}
                                               CNN model to use (default: resnet152)
  --data_path PATH                             Data directory path (default: ./data)
  --dictionary_num INT                         Dictionary size (default: 400)
  --neuron_iter INT                           Neuron iterations (default: 50)
  --lr_neuron FLOAT                           Neuron learning rate (default: 0.01)
  --landa FLOAT                               Sparsity coefficient (default: 2)
  --batch_size INT                            Batch size (default: 50)
  --max_samples INT                           Max test samples (default: 50)
  --num_workers INT                           DataLoader workers (default: 0)
  --pin_memory                                Pin memory for GPU transfer
```

## Dataset Setup

### CIFAR-10/100 and MNIST
These datasets are automatically downloaded when first used.

### ImageNet
ImageNet requires manual setup:

```
data_imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Algorithm Overview

### Locally Competitive Algorithm (LCA)

LCA learns sparse representations by optimizing:

```
minimize ||x - Φa||² + λ||a||₁
```

Where:
- `x`: Input features
- `Φ`: Dictionary matrix
- `a`: Sparse activations
- `λ`: Sparsity coefficient

### Processing Pipeline

1. **Feature Extraction**: CNN models extract features from images
2. **Dictionary Learning**: LCA creates sparse dictionary from training features
3. **Sparse Coding**: Test features are encoded using learned dictionary
4. **Classification**: Neural network classifies based on sparse codes

## Performance

### Accuracy Metrics

The system computes two accuracy measures:
- **Max-based**: Classification using maximum activation
- **Sum-based**: Classification using sum of class-specific activations

### Memory Management

- Automatic GPU memory management with `torch.cuda.empty_cache()`
- Configurable batch sizes for memory efficiency
- CPU/GPU data transfers optimized for performance

## Examples

### CIFAR-10 with ResNet152
```bash
python src/lca_feature_extractor.py --dataset cifar10 --model resnet152 --dictionary_num 1000
```

### MNIST with raw images
```bash
python src/lca_feature_extractor.py --dataset mnist --model none --dictionary_num 400
```

### ImageNet with EfficientNet
```bash
python src/lca_feature_extractor.py --dataset imagenet --model efficientnet --batch_size 32
```

## GPU Compatibility

The code automatically detects and utilizes available GPUs:
- **NVIDIA GPUs**: Full CUDA support
- **AMD GPUs**: ROCm support  
- **CPU Fallback**: Automatic fallback when no GPU available

Memory recommendations:
- **4-6GB GPU**: `--dictionary_num 1000 --max_samples 500`
- **8-12GB GPU**: `--dictionary_num 2000 --max_samples 1000`
- **16GB+ GPU**: `--dictionary_num 5000 --max_samples 2000`

## Output

The system outputs:
- Training and test processing times
- Dictionary creation statistics  
- Training accuracy on neural network classifier
- Max-based and sum-based test accuracies
- Memory usage information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{takaghaj2024dseld,
  title={D-SELD: Dataset-Scalable Exemplar LCA-Decoder},
  author={S. M. Takaghaj and J. Sampson},
  journal={Neuromorphic Computing and Engineering Journal},
  year={2024}
}

@misc{d-seld-implementation,
  title={D-SELD: Dataset-Scalable Exemplar LCA-Decoder Implementation},
  author={Sanaz Mahmoodi Takaghaj},
  year={2024},
  url={https://github.com/Sanaz-Tak/D-SELD}
}
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `--dictionary_num` and `--batch_size`
2. **CUDA Errors**: Ensure PyTorch CUDA version matches your driver
3. **Dataset Not Found**: Check data path and ensure proper dataset structure

### Performance Tips

- Use GPU for better performance
- Adjust batch size based on available memory
- Use fewer dictionary elements for faster processing
- Enable `--pin_memory` for faster GPU transfers

## Acknowledgments

- PyTorch team for the deep learning framework
- Torchvision for pre-trained models and datasets
- Original LCA algorithm developers