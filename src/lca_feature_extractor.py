import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
import pickle
import os

class LCA:
    def __init__(self, feature_size, dictionary_num, UPDATE_DICT, dictionary_iter, 
                 neuron_iter, lr_dictionary, lr_neuron, landa):
        self.feature_size = feature_size
        self.dict_num = dictionary_num
        self.UPDATE_DICT = UPDATE_DICT
        self.dict_iter = dictionary_iter
        self.neuron_iter = neuron_iter
        self.lr_dict = lr_dictionary
        self.lr_neuron = lr_neuron
        self.landa = landa
        self.dictionary = None
        self.data = None
        self.input = None
        self.a = None
        self.u = None

    def lca_update(self, phi, I):
        device = self.input.device  # Get device from input tensor
        u_list = [torch.zeros([1, self.dict_num]).to(device)]
        a_list = [self.threshold(u_list[0], 'soft', True, self.landa).to(device)]

        # Simplified input reshaping - use the actual input shape
        batch_size = self.input.shape[0]
        input = self.input.reshape(batch_size, -1)

        S = input.T
        b = torch.matmul(S.T, phi)
        for t in range(self.neuron_iter):
            u = self.neuron_update(u_list[t], a_list[t], b, phi, I)
            u_list.append(u)
            a = self.threshold(u, 'soft', True, self.landa)
            a_list.append(a)

        self.a = a_list[-1]
        self.u = u_list[-1]
        del u_list, a_list, input, S, b, phi, I

    def loss(self):
        # Dynamic reshaping based on actual input dimensions
        batch_size = self.input.shape[0]
        feature_size = self.input.shape[1:].numel()  # Total features per sample
        s = self.input.reshape(batch_size, feature_size).T  # [feature_size, batch_size]
        
        phi = self.dictionary.reshape(self.dict_num, -1).T  # [feature_size, dict_num]
        a = self.a  # [batch_size, dict_num]
        
        # Compute residual: s - phi @ a.T
        residual = s - torch.mm(phi, a.T)  # [feature_size, batch_size]
        approximation_loss = .5 * torch.linalg.norm(residual, 'fro')
        sparsity_loss = self.landa * torch.sum(torch.abs(a))
        loss = approximation_loss + sparsity_loss
        print('Loss: {:.2f}'.format(loss.item()), 'approximation loss: {:.2f}'.format(approximation_loss.item()), 'sparsity loss: {:.2f}'.format(sparsity_loss.item()))
        return loss

    def dict_update(self):
        phi = self.dictionary.reshape(self.dict_num, -1).T
        phi = phi.to(self.input.device)  # Use device from input tensor
        S = self.input.reshape(-1, 1)
        d_phi = torch.matmul((S.reshape(-1, 50)-torch.matmul(phi, self.a.T)), self.a)
        d_dict = d_phi.T.reshape([self.dict_num, 1000])
        d_dict = d_dict.cpu()
        self.dictionary = self.dictionary + d_dict * self.lr_dict

    def threshold(self, u, type, rectify, landa):
        u_zeros = torch.zeros_like(u)
        if type == 'soft':
            if rectify:
                a_out = torch.where(torch.greater(u, landa), u - landa, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u - landa,
                                    torch.where(torch.le(u, - landa), u + landa, u_zeros))
        elif type == 'hard':
            if rectify:
                a_out = torch.where(torch.gt(u, landa), u, u_zeros)
            else:
                a_out = torch.where(torch.ge(u, landa), u,
                                    torch.where(torch.le(u, -landa), u, u_zeros))
        else:
            assert False, (f'Parameter thresh_type must be "soft" or "hard", not {type}')
        return a_out

    def neuron_update(self, u_in, a_in, b, phi, I):
        # Lazy G computation: compute (phi.T @ phi - I) @ a_in on-demand
        # This avoids storing the large G matrix in memory
        Ga = torch.mm(phi.T, torch.mm(phi, a_in.T)) - torch.mm(I, a_in.T)
        du = b - Ga.T - u_in
        u_out = u_in + self.lr_neuron * du
        return u_out

def normalize(M):
    sigma = torch.sum(M * M)
    return M / torch.sqrt(sigma)

def get_model_and_transforms(model_name, dataset_name='cifar10'):
    if model_name == 'resnet152':
        model = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1)
        
        # Version 1: Remove only classifier, keep avgpool (flattened features [2048])
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Version 2: Remove both avgpool and classifier (spatial features [2048, 7, 7])
        # model = nn.Sequential(*list(model.children())[:-2])
        
        transforms_func = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
        model = model.features
        
        transforms_func = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        
        # Version 1: Remove only classifier, keep avgpool (flattened features [512])
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Version 2: Use only features layer (spatial features [512, 7, 7])
        # model = model.features
        
        transforms_func = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'efficientnet':
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Version 1: Remove only classifier, keep avgpool (flattened features [1280])
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Version 2: Remove both avgpool and classifier (spatial features [1280, 7, 7])
        # model = nn.Sequential(*list(model.children())[:-2])
        
        transforms_func = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Version 1: Remove only classifier, keep avgpool (flattened features [1280])
        model = nn.Sequential(*list(model.children())[:-1])
        
        # Version 2: Remove both avgpool and classifier (spatial features [1280, 7, 7])
        # model = nn.Sequential(*list(model.children())[:-2])
        
        transforms_func = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_name == 'inception':
        model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Version 1: Remove only classifier, keep avgpool (flattened features [2048])
        model.fc = torch.nn.Identity()
        
        # Version 2: Remove both avgpool and classifier (spatial features [2048, 8, 8])
        # model = nn.Sequential(*list(model.children())[:-2])
        
        transforms_func = transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, transforms_func

def get_mnist_transforms():
    return transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),  # This converts 0-255 to 0-1
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST-specific normalization
    ])

def get_feature_dimensions(model_name):
    """Get the flattened feature dimensions for each model."""
    
    # Version 1: Using [:-1] (keep avgpool, remove classifier)
    feature_dims_v1 = {
        'resnet152': 2048,      # [batch, 2048] - flattened after avgpool
        'densenet121': 1024 * 7 * 7, # [batch, 1024, 7, 7] = 50176
        'vgg16': 512,           # [batch, 512] - flattened after avgpool
        'efficientnet': 1280,   # [batch, 1280] - flattened after avgpool
        'mobilenet': 1280 * 7 * 7,   # [batch, 1280, 7, 7] = 62720
        'inception': 2048       # [batch, 2048] - flattened after avgpool
    }
    
    # Version 2: Using [:-2] for ResNet, model.features for VGG (spatial features)
    feature_dims_v2 = {
        'resnet152': 2048 * 7 * 7,  # [batch, 2048, 7, 7] = 100352
        'densenet121': 1024 * 7 * 7, # [batch, 1024, 7, 7] = 50176
        'vgg16': 512 * 7 * 7,        # [batch, 512, 7, 7] = 25088
        'efficientnet': 1280 * 7 * 7, # [batch, 1280, 7, 7] = 62720
        'mobilenet': 1280 * 7 * 7,   # [batch, 1280, 7, 7] = 62720
        'inception': 2048 * 7 * 7     # [batch, 2048, 7, 7] = 100352
    }
    
    # Currently using Version 1 (flattened features)
    feature_dims = feature_dims_v1
    
    # To switch to Version 2 (spatial features), uncomment this line:
    # feature_dims = feature_dims_v2
    
    if model_name not in feature_dims:
        raise ValueError(f"Unknown model: {model_name}")
    
    return feature_dims[model_name]

def get_dataset_info(dataset_name):
    """Get dataset information including number of classes and data loading function."""
    if dataset_name == 'cifar10':
        num_classes = 10
        def load_dataset(data_path, train, transform):
            return torchvision.datasets.CIFAR10(
                root=data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        num_classes = 100
        def load_dataset(data_path, train, transform):
            return torchvision.datasets.CIFAR100(
                root=data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'mnist':
        num_classes = 10
        def load_dataset(data_path, train, transform):
            return torchvision.datasets.MNIST(
                root=data_path, train=train, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        num_classes = 1000
        def load_dataset(data_path, train, transform):
            # For ImageNet, we need to specify the split
            split = 'train' if train else 'val'
            return torchvision.datasets.ImageFolder(
                root=os.path.join(data_path, split), transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return num_classes, load_dataset

def main():
    parser = argparse.ArgumentParser(description='Multi-Dataset CNN Feature Extraction with LCA (Memory Efficient)')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'cifar100', 'mnist', 'imagenet'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet152', 
                       choices=['resnet152', 'densenet121', 'vgg16', 'efficientnet', 'mobilenet', 'inception', 'none'],
                       help='CNN model to use (use "none" for MNIST)')
    parser.add_argument('--data_path', type=str, default='./data', help='Data path')
    parser.add_argument('--dictionary_num', type=int, default=400, help='Dictionary size')
    parser.add_argument('--neuron_iter', type=int, default=50, help='Neuron iterations')
    parser.add_argument('--lr_neuron', type=float, default=0.01, help='Neuron learning rate')
    parser.add_argument('--landa', type=float, default=2, help='Sparsity coefficient')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for data loading and feature extraction')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum test samples for LCA processing (dictionary uses all training samples)')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 for memory efficiency)')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster GPU transfer (uses more RAM)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected dataset: {args.dataset}")
    
    torch.manual_seed(1234)
    
    # Get dataset information
    num_classes, load_dataset = get_dataset_info(args.dataset)
    print(f"Number of classes: {num_classes}")
    
    # Special handling for MNIST - no CNN features needed
    if args.dataset == 'mnist':
        print("MNIST dataset detected - using raw image data (no CNN feature extraction)")
        transforms_func = get_mnist_transforms()
        
        # Create data directory if it doesn't exist
        os.makedirs(args.data_path, exist_ok=True)
        
        # Load MNIST datasets
        train_dataset = load_dataset(args.data_path, train=True, transform=transforms_func)
        test_dataset = load_dataset(args.data_path, train=False, transform=transforms_func)
        
        # Use configurable batch sizes and workers for memory efficiency
        training_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_memory)
        testing_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        # For MNIST, feature_size is 28*28 = 784
        feature_size = 28 * 28
        print(f"MNIST feature size: {feature_size}")
        
        lca_model = LCA(
            feature_size=feature_size,
            dictionary_num=args.dictionary_num,
            UPDATE_DICT=False,
            dictionary_iter=1,
            neuron_iter=args.neuron_iter,
            lr_dictionary=0.001,
            lr_neuron=args.lr_neuron,
            landa=args.landa
        )
        
        # Process MNIST data directly (no CNN features)
        print("Processing MNIST training data directly...")
        
        # Initialize storage for features and labels
        all_feature_maps = []
        all_labels = []
        
        # Process training data in small chunks for dictionary creation
        sample_count = 0
        for batch_data, batch_labels in training_dataloader:
            print(f"Processing batch {sample_count//args.batch_size + 1}, samples {sample_count}-{sample_count + len(batch_data)}")
            
            # For MNIST, batch_data is already the features (28x28 images)
            # Flatten to [batch, 784]
            batch_features = batch_data.view(batch_data.size(0), -1)
            
            # Store features and labels
            all_feature_maps.append(batch_features.cpu())
            all_labels.append(batch_labels)
            
            sample_count += len(batch_data)
            
            # Break if we've collected enough samples for dictionary
            if sample_count >= args.dictionary_num:
                print(f"Collected {sample_count} samples, stopping data loading")
                break
            
            # Clear memory
            del batch_data, batch_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all features and labels
        all_feature_maps = torch.cat(all_feature_maps, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"Processed {len(all_labels)} MNIST training samples for dictionary")
        print(f"Feature maps shape: {all_feature_maps.shape}")
        
        # Create complete dictionary from all features
        dict_size = min(args.dictionary_num, len(all_feature_maps))
        lca_model.dictionary = torch.zeros(dict_size, lca_model.feature_size)
        
        for i in range(dict_size):
            lca_model.dictionary[i] = normalize(all_feature_maps[i].detach())
        
        # Keep features in memory for LCA processing
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Created MNIST dictionary with {dict_size} elements")
        
        # Continue with LCA processing...
        
    else:
        # Regular CNN-based processing for other datasets
        if args.model == 'none':
            print("No CNN model selected - using raw image data")
            # This should not happen for non-MNIST datasets, but handle gracefully
            if args.dataset == 'mnist':
                print("MNIST with no model - this is correct")
            else:
                print("Warning: No model selected for non-MNIST dataset")
                return
        else:
            model, transforms_func = get_model_and_transforms(args.model)
            model.eval()
        
        # Create data directory if it doesn't exist
        os.makedirs(args.data_path, exist_ok=True)
        
        # Load datasets
        if args.model == 'none':
            # For MNIST with no model, use MNIST transforms
            transforms_func = get_mnist_transforms()
        
        train_dataset = load_dataset(args.data_path, train=True, transform=transforms_func)
        test_dataset = load_dataset(args.data_path, train=False, transform=transforms_func)
        
        # Use configurable batch sizes and workers for memory efficiency
        training_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_memory)
        testing_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=args.pin_memory)
        
        # Get correct feature dimensions for the model
        if args.model == 'none':
            # For MNIST with no model, feature size is 28*28 = 784
            feature_size = 28 * 28
            print(f"MNIST feature size (no model): {feature_size}")
        else:
            feature_size = get_feature_dimensions(args.model)
            print(f"Model {args.model} flattened feature size: {feature_size}")
        
        lca_model = LCA(
            feature_size=feature_size,
            dictionary_num=args.dictionary_num,
            UPDATE_DICT=False,
            dictionary_iter=1,
            neuron_iter=args.neuron_iter,
            lr_dictionary=0.001,
            lr_neuron=args.lr_neuron,
            landa=args.landa
        )
        
        # Process data incrementally without loading everything into memory
        print("Processing training data incrementally...")
        
        # Initialize storage for features and labels
        all_feature_maps = []
        all_labels = []
        
        # Process training data in small chunks for dictionary creation
        sample_count = 0
        for batch_data, batch_labels in training_dataloader:
            print(f"Processing batch {sample_count//args.batch_size + 1}, samples {sample_count}-{sample_count + len(batch_data)}")
            
            # Process this batch
            if args.model == 'none':
                # For MNIST with no model, use raw image data
                batch_features = batch_data.view(batch_data.size(0), -1)
            else:
                # For CNN models, extract features
                with torch.no_grad():
                    batch_features = model(batch_data.to(device))
                    # Check if DenseNet needs reshaping to 1024 dimensions
                    if args.model == 'densenet121' and feature_size == 1024:
                        batch_features = batch_features.reshape(-1, 1024)
                
            # Store features and labels
            all_feature_maps.append(batch_features.cpu())
            all_labels.append(batch_labels)
                
            sample_count += len(batch_data)
            
            # Break if we've collected enough samples for dictionary
            if sample_count >= args.dictionary_num:
                print(f"Collected {sample_count} samples, stopping data loading")
                break
            
            # Clear memory
            del batch_data, batch_features
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Concatenate all features and labels
        all_feature_maps = torch.cat(all_feature_maps, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        print(f"Processed {len(all_labels)} training samples for dictionary")
        print(f"Feature maps shape: {all_feature_maps.shape}")
        
        # Flatten the features if they're not already flattened
        if len(all_feature_maps.shape) > 2:
            all_feature_maps = all_feature_maps.view(all_feature_maps.size(0), -1)
            print(f"Flattened feature maps shape: {all_feature_maps.shape}")
        
        # Create complete dictionary from all features
        dict_size = min(args.dictionary_num, len(all_feature_maps))
        lca_model.dictionary = torch.zeros(dict_size, lca_model.feature_size)
        
        for i in range(dict_size):
            lca_model.dictionary[i] = normalize(all_feature_maps[i].detach())
        
        # Keep features in memory for LCA processing (only delete after LCA is done)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Created dictionary with {dict_size} elements")
    
    # Now process the same data with LCA
    print("Processing training data with LCA for neural network training...")
    start_time = time.time()
    
    # Create phi and I matrices for LCA
    if args.dataset != 'mnist':
        # For CNN-based datasets, move model to device
        model = model.to(device)
    
    dict = lca_model.dictionary.reshape(lca_model.dict_num, -1)
    phi = dict.T.to(device)
    I = torch.eye(lca_model.dict_num).to(device)
    del dict
    print("Using lazy G computation to avoid OOM...")
    
    # Initialize storage for training LCA results
    a_all_train = []
    all_train_labels_for_nn = []
    
    # Process the same data with LCA in batches (capped at dict_size)
    sample_count = 0
    batch_size = min(100, args.batch_size)  # Use smaller batches for LCA
    
    for i in range(0, min(dict_size, len(all_labels)), batch_size):
        if sample_count >= dict_size:
            break
            
        end_idx = min(i + batch_size, dict_size, len(all_labels))
        batch_features = all_feature_maps[i:end_idx].to(device)
        batch_labels = all_labels[i:end_idx]
        
        print(f"Processing LCA batch {i//batch_size + 1} (samples {i}-{end_idx})")
        
        # Run LCA on this batch
        try:
            print(f"Running LCA on batch with shape: {batch_features.shape}")
            lca_model.input = batch_features
            lca_model.lca_update(phi, I)
            a = lca_model.a.clone().detach().type(torch.float)
            print(f"LCA completed, output shape: {a.shape}")
            
            # Store results
            a_all_train.append(a.cpu())
            all_train_labels_for_nn.append(batch_labels)
            sample_count += (end_idx - i)
            
        except Exception as e:
            print(f"Error in LCA processing: {e}")
            import traceback
            traceback.print_exc()
            break
        
        # Clear memory
        del batch_features, a
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    end_time = time.time()
    
    print(f"LCA processing completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {sample_count} training samples for LCA")
    
    # Clean up memory after LCA processing
    del all_feature_maps, all_labels
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process LCA results without concatenation
    if a_all_train:
        print(f"Training LCA processing completed")
        print(f"Number of LCA batches: {len(a_all_train)}")
        print(f"First batch shape: {a_all_train[0].shape}")
        
        # Concatenate labels for accuracy calculations (keep features as batches)
        all_train_labels_concatenated = torch.cat(all_train_labels_for_nn, dim=0)
        print(f"Concatenated training labels shape: {all_train_labels_concatenated.shape}")
    else:
        print("No LCA processing completed - dictionary size may be larger than available training samples")
        return
    
    # Train Neural Network classifier on training LCA features
    print("Training Neural Network classifier on training data...")
    
    # Define neural network with dynamic output size based on dataset
    nn_model = torch.nn.Sequential(
        torch.nn.Linear(lca_model.dict_num, 1000),
        #torch.nn.Dropout(p=0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, num_classes),
        #torch.nn.Softmax(dim=1)
    ).to(device)
    
    # Training parameters
    lr = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    nn_model.zero_grad()
    
    # Training loop - process batches individually
    print("Training NN classifier...")
    for epoch in range(1000):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_a, batch_labels) in enumerate(zip(a_all_train, all_train_labels_for_nn)):
            # Normalize this batch
            batch_a_normalized = batch_a.clone()
            for i in range(len(batch_a_normalized)):
                batch_a_normalized[i] = batch_a_normalized[i] / torch.max(batch_a_normalized[i])
            
            # Prepare labels for this batch
            y0_batch = batch_labels.clone().detach().type(torch.int64).to(device)
            y_hot_batch = torch.nn.functional.one_hot(y0_batch, num_classes=num_classes).float().to(device)
            
            # Forward pass
            y_pred = nn_model(batch_a_normalized.to(device))
            loss = loss_fn(y_pred, y_hot_batch)
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            num_batches += 1
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 100 == 99:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}, average classification loss: {avg_loss:.4f}')
    
    # Calculate training accuracy
    print("Calculating training accuracy...")
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch_idx, (batch_a, batch_labels) in enumerate(zip(a_all_train, all_train_labels_for_nn)):
            # Normalize this batch
            batch_a_normalized = batch_a.clone()
            for i in range(len(batch_a_normalized)):
                batch_a_normalized[i] = batch_a_normalized[i] / torch.max(batch_a_normalized[i])
            
            y0_batch = batch_labels.clone().detach().type(torch.int64).to(device)
            
            # Predict
            y_pred_batch = nn_model(batch_a_normalized.to(device))
            nn_predictions_batch = torch.argmax(y_pred_batch, dim=1)
            
            # Count correct predictions
            correct_predictions += sum(nn_predictions_batch == y0_batch).item()
            total_predictions += len(y0_batch)
    
    accuracy_nn_train = correct_predictions / total_predictions
    print(f'Training accuracy (NN): {accuracy_nn_train:.4f}')
    
    # Second: Process test data and run LCA for testing
    print("Processing test data and running LCA for testing...")
    start_time = time.time()
    
    # Initialize storage for test results
    a_all_test = []
    all_test_labels = []
    
    # Process test data in batches
    sample_count = 0
    for batch_data, batch_labels in testing_dataloader:
        if sample_count >= args.max_samples:  # Limit test samples
            break
            
        print(f"Processing test batch {sample_count//args.batch_size + 1}, samples {sample_count}-{sample_count + len(batch_data)}")
        
        # Process this batch
        with torch.no_grad():
            if args.dataset == 'mnist' or args.model == 'none':
                # For MNIST or no model, batch_data is already the features (28x28 images)
                # Flatten to [batch, 784]
                batch_features = batch_data.view(batch_data.size(0), -1)
            else:
                # For other datasets, extract CNN features
                batch_features = model(batch_data.to(device))
                # Check if DenseNet needs reshaping to 1024 dimensions
                if args.model == 'densenet121' and feature_size == 1024:
                    batch_features = batch_features.reshape(-1, 1024)
                
                # Flatten batch features if needed
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.view(batch_features.size(0), -1)
            
            # Run LCA on this batch
            lca_model.input = batch_features
            lca_model.lca_update(phi, I)
            a = lca_model.a.clone().detach().type(torch.float)
            
            # Store results
            a_all_test.append(a.cpu())
            all_test_labels.append(batch_labels)
            
        sample_count += len(batch_data)
        
        # Clear memory immediately
        del batch_data, batch_features, a
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    a_all_test = torch.cat(a_all_test, 0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    end_time = time.time()
    
    print(f"Test LCA processing completed in {end_time - start_time:.2f} seconds")
    print(f"a_all_test shape: {a_all_test.shape}")
    print(f"Test labels shape: {all_test_labels.shape}")
    
    # Calculate Max-based accuracy on test data
    indices = torch.argmax(a_all_test, dim=1).to('cpu')
    # Use training labels for dictionary indices, compare with test labels
    accuracy_max = sum(all_train_labels_concatenated[indices] == all_test_labels) / len(all_test_labels)
    print(f'Test accuracy (max): {accuracy_max:.4f}')
    
    # Calculate Sum-based accuracy
    print("Calculating Sum-based accuracy...")
    
    # Create dictionary of indices for each class
    indices_dict = {}
    for digit in range(num_classes):
        indices = torch.nonzero(all_train_labels_concatenated[0:lca_model.dict_num] == digit).squeeze()
        # Handle empty tensors and 0-d tensors properly
        if indices.numel() == 0:
            indices_dict[digit] = torch.tensor([])
        elif indices.dim() == 0:  # 0-d tensor (scalar)
            indices_dict[digit] = torch.tensor([indices.item()])
        else:
            indices_dict[digit] = indices
    
    # Calculate sum-based predictions
    max_indices = []
    for i in range(len(all_test_labels)):
        # Sum coefficients for each class, handling empty tensors
        data = []
        for digit in range(num_classes):
            if indices_dict[digit].numel() > 0:
                # Ensure we're working with a 1-d tensor
                digit_indices = indices_dict[digit]
                if digit_indices.dim() == 0:
                    digit_indices = digit_indices.unsqueeze(0)
                data.append(torch.sum(a_all_test[i, digit_indices]).item())
            else:
                data.append(0.0)  # No samples for this class
        # Find class with maximum sum
        max_index = max(range(len(data)), key=lambda x: data[x])
        max_indices.append(max_index)
    
    # Calculate sum-based accuracy
    accuracy_sum = sum(torch.tensor(max_indices) == all_test_labels) / len(all_test_labels)
    print(f'Test accuracy (sum): {accuracy_sum:.4f}')
    
    # Test the trained neural network on test data
    print("Testing trained neural network on test data...")
    
    # Normalize test LCA coefficients
    a_all_test_normalized = a_all_test.clone()
    for i in range(len(a_all_test_normalized)):
        a_all_test_normalized[i] = a_all_test_normalized[i] / torch.max(a_all_test_normalized[i])
    
    # Calculate NN-based accuracy
    with torch.no_grad():
        y_pred_test = nn_model(a_all_test_normalized.to(device))
        nn_predictions_test = torch.argmax(y_pred_test, dim=1)
        accuracy_nn_test = sum(nn_predictions_test == all_test_labels.to(device)) / len(all_test_labels)
        print(f'Test accuracy (NN): {accuracy_nn_test:.4f}')
    
    print(f'Total elapsed time: {end_time - start_time:.2f} seconds')
    
    print("Memory-efficient processing completed successfully!")

if __name__ == '__main__':
    main()
