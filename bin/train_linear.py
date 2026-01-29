"""
Training Script for Linear Models

This script trains linear models from scratch on various datasets.
Trained models are saved for later use in stitching experiments.
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
from datetime import datetime

# Setup paths - HARDCODED to work from any location
# This allows you to run the script from anywhere on the cluster
PROJECT_ROOT = Path('/home/voz/almudevar/similarity')
SCRIPT_DIR = PROJECT_ROOT / 'bin'
SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = PROJECT_ROOT / 'trained_models'
DATA_DIR = PROJECT_ROOT / 'data'
TINYIMAGENET_DEFAULT_ROOT = Path('/home/voz/shared/database/vision/tiny-imagenet-200')

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

from src.linear_models import create_linear_model, print_model_summary


def get_dataset(dataset_name: str, data_dir: str = None):
    """Load train and test datasets."""
    
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(exist_ok=True, parents=True)
    
    if dataset_name == 'cifar10':
        # CIFAR-10: 32x32 RGB images
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        
        num_classes = 10
        input_size = 3 * 32 * 32
        
    elif dataset_name == 'cifar100':
        # CIFAR-100: 32x32 RGB images, 100 classes
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        
        num_classes = 100
        input_size = 3 * 32 * 32
        
    elif dataset_name == 'mnist':
        # MNIST: 28x28 grayscale images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        
        num_classes = 10
        input_size = 1 * 28 * 28
        
    elif dataset_name == 'svhn':
        # SVHN: 32x32 RGB images
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        
        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_train)
        test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform_test)
        
        num_classes = 10
        input_size = 3 * 32 * 32
    
    elif dataset_name == 'tinyimagenet':
        # TinyImageNet: 64x64 RGB images, 200 classes
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        tiny_root = TINYIMAGENET_DEFAULT_ROOT if TINYIMAGENET_DEFAULT_ROOT.exists() else (data_dir / 'tiny-imagenet-200')
        val_images_dir = tiny_root / 'val' / 'images'
        if val_images_dir.exists():
            if any(val_images_dir.iterdir()):
                raise FileNotFoundError(
                    f"TinyImageNet val/images still contains files at {val_images_dir}. "
                    "Please reorganize val into class subfolders before training."
                )
            val_images_dir.rmdir()
        train_dataset = datasets.ImageFolder(root=tiny_root / 'train',
                                            transform=transform_train)
        test_dataset = datasets.ImageFolder(root=tiny_root / 'val',
                                           transform=transform_test)
        
        num_classes = 200
        input_size = 3 * 64 * 64
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset, num_classes, input_size


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_linear_model(model_name: str,
                       dataset_name: str,
                       num_epochs: int = 50,
                       batch_size: int = 128,
                       lr: float = 0.001,
                       weight_decay: float = 5e-4,
                       save_dir: str = None,
                       device: str = None):
    """
    Train a linear model from scratch.
    
    Args:
        model_name: Name of the linear model architecture
        dataset_name: Name of the dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        save_dir: Directory to save trained models (default: PROJECT_ROOT/trained_models)
        device: Device to train on (None = auto-detect)
    """
    
    # Setup save directory
    if save_dir is None:
        save_dir = MODELS_DIR
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    train_dataset, test_dataset, num_classes, input_size = get_dataset(dataset_name)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Input size: {input_size}")
    print(f"  Classes: {num_classes}")
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = create_linear_model(model_name, input_size, num_classes)
    print_model_summary(model)
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    print("=" * 80)
    
    best_test_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['test_loss'].append(test_loss)
        training_history['test_acc'].append(test_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
            # Save model checkpoint
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            checkpoint_path = save_dir / f"{model_name}_{dataset_name}_best.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'input_size': input_size,
                'num_classes': num_classes,
                'training_history': training_history
            }, checkpoint_path)
            
            print(f"  → Saved best model (acc: {test_acc:.4f})")
    
    print("=" * 80)
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    
    # Save final model
    final_checkpoint_path = save_dir / f"{model_name}_{dataset_name}_final.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
        'model_name': model_name,
        'dataset_name': dataset_name,
        'input_size': input_size,
        'num_classes': num_classes,
        'training_history': training_history
    }, final_checkpoint_path)
    
    # Save training metadata
    metadata = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'input_size': input_size,
        'num_classes': num_classes,
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = save_dir / f"{model_name}_{dataset_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to: {save_dir}")
    print(f"  Best checkpoint: {checkpoint_path.name}")
    print(f"  Final checkpoint: {final_checkpoint_path.name}")
    print(f"  Metadata: {metadata_path.name}")
    
    return model, training_history


def main():
    parser = argparse.ArgumentParser(description='Train linear models for representation analysis')
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['linear_small', 'linear_medium', 'linear_large', 
                               'linear_deep', 'linear_wide'],
                       help='Linear model architecture')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'mnist', 'svhn', 'tinyimagenet'],
                       help='Dataset to train on')
    
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay for regularization')
    
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save trained models')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to train on (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Train model
    train_linear_model(
        model_name=args.model,
        dataset_name=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
