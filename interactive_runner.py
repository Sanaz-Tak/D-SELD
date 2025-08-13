#!/usr/bin/env python3
"""
Simple script to run the multi-dataset LCA code with user selection.
"""

import subprocess
import sys
import os

def main():
    print("Multi-Dataset LCA Feature Extraction")
    print("=" * 40)
    print("Available datasets:")
    print("1. CIFAR-10 (10 classes)")
    print("2. CIFAR-100 (100 classes)")
    print("3. ImageNet (1000 classes)")
    print("4. MNIST (10 classes)")
    print()
    
    while True:
        try:
            choice = input("Select dataset (1-4): ").strip()
            if choice == '1':
                dataset = 'cifar10'
                break
            elif choice == '2':
                dataset = 'cifar100'
                break
            elif choice == '3':
                dataset = 'imagenet'
                break
            elif choice == '4':
                dataset = 'mnist'
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    print(f"\nSelected dataset: {dataset}")
    
    # Set default data path based on dataset
    if dataset == 'imagenet':
        data_path = './data_imagenet'
        print("Note: For ImageNet, ensure your data is organized as:")
        print("  data_imagenet/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  └── val/")
        print("      ├── class1/")
        print("      ├── class2/")
        print("      └── ...")
    elif dataset == 'mnist':
        data_path = './data_mnist'
        print("Note: MNIST will be automatically downloaded to ./data_mnist/")
    else:
        data_path = f'./data_{dataset}'
    
    print(f"Data will be stored in: {data_path}")
    
    # Ask for model selection
    print("\nAvailable models:")
    if dataset == 'mnist':
        print("Note: For MNIST, you can choose 'none' to use raw image data (no CNN features)")
        models = ['resnet152', 'densenet121', 'vgg16', 'efficientnet', 'mobilenet', 'inception', 'none']
    else:
        models = ['resnet152', 'densenet121', 'vgg16', 'efficientnet', 'mobilenet', 'inception']
    
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = input(f"\nSelect model (1-{len(models)}): ").strip()
            choice = int(choice)
            if 1 <= choice <= len(models):
                model = models[choice - 1]
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(models)}.")
        except (ValueError, KeyboardInterrupt):
            if KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
            print("Invalid input. Please enter a number.")
    
    print(f"Selected model: {model}")
    
    # Validate model selection for MNIST
    if dataset == 'mnist' and model == 'none':
        print("Using MNIST with raw image data (no CNN feature extraction)")
    elif dataset == 'mnist':
        print("Note: MNIST with CNN model will still work, but raw data is often sufficient")
    
    # Build command
    cmd = [
        'python', 'src/lca_feature_extractor.py',
        '--dataset', dataset,
        '--model', model,
        '--data_path', data_path,
        '--batch_size', '16',
        '--max_test_samples', '10000'
    ]
    
    print(f"\nRunning command:")
    print(' '.join(cmd))
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("\nExecution completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with error code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: Python executable not found. Make sure Python is installed and in your PATH.")
        sys.exit(1)

if __name__ == '__main__':
    main()
