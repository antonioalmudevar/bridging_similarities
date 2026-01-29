"""
Extended Model Stitching with Linear Models Support

This script extends the original run_stitching.py to support both:
1. Convolutional models (ResNet, VGG, DenseNet, etc.)
2. Linear models (trained from scratch)

Usage examples:
    # Compare two linear models:
    python run_stitching_extended.py --mode single --model1 linear_medium --model2 linear_large --dataset cifar10 --model-type linear
    
    # Compare linear with conv:
    python run_stitching_extended.py --mode single --model1 linear_medium --model2 resnet18 --dataset cifar10 --model-type mixed

    # Compare two pretrained ImageNet models (requires ImageNet data on disk):
    python run_stitching_extended.py --mode single --model1 resnet18 --model2 vgg16 --dataset imagenet --imagenet-root /path/to/imagenet
    
    # Train a linear model first:
    python train_linear.py --model linear_medium --dataset cifar10 --epochs 50
"""

import argparse
import time
import sys
import torch
import numpy as np
from pathlib import Path
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import json
import pickle

# Setup paths - HARDCODED to work from any location
# This allows you to run the script from anywhere on the cluster
PROJECT_ROOT = Path('/home/voz/almudevar/similarity')
SCRIPT_DIR = PROJECT_ROOT / 'bin'
SRC_DIR = PROJECT_ROOT / 'src'
MODELS_DIR = PROJECT_ROOT / 'trained_models'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
DATA_DIR = PROJECT_ROOT / 'data'
IMAGENET_DEFAULT_ROOT = Path('/home/voz/shared/database/vision/ILSVRC2012')
TINYIMAGENET_DEFAULT_ROOT = Path('/home/voz/shared/database/vision/tiny-imagenet-200')

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

from src.improved_stitching import ImprovedModelStitcher
from src.linear_models import create_linear_model, LinearNet
from src.cnn_models import create_cnn_model


class CIFAR100Coarse(datasets.CIFAR100):
    """CIFAR-100 dataset using coarse labels (20 superclasses)."""

    meta = {
        "filename": "meta",
        "key": "coarse_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None, download: bool = False):
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.targets = self._load_coarse_targets()
        self._load_meta()

    def _load_coarse_targets(self):
        downloaded_list = self.train_list if self.train else self.test_list
        coarse_targets = []
        for file_name, _ in downloaded_list:
            file_path = Path(self.root) / self.base_folder / file_name
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                if "coarse_labels" not in entry:
                    raise RuntimeError("CIFAR-100 file missing coarse_labels field.")
                coarse_targets.extend(entry["coarse_labels"])
        return coarse_targets


def _replace_last_linear_in_sequential(seq: nn.Sequential, num_classes: int) -> nn.Linear:
    for idx in range(len(seq) - 1, -1, -1):
        layer = seq[idx]
        if isinstance(layer, nn.Linear):
            new_layer = nn.Linear(layer.in_features, num_classes)
            seq[idx] = new_layer
            return new_layer
    raise ValueError("No Linear layer found to replace in Sequential classifier.")


def replace_classifier_for_coarse(model: nn.Module, num_classes: int = 20) -> nn.Linear:
    """Replace final classifier layer to output num_classes and return the new head."""
    device = next(model.parameters()).device
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes).to(device)
            head = model.classifier
        elif isinstance(classifier, nn.Sequential):
            head = _replace_last_linear_in_sequential(model.classifier, num_classes)
            head.to(device)
        else:
            raise ValueError("Unsupported classifier type on model.classifier.")
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)
        head = model.fc
    elif hasattr(model, 'linear') and isinstance(model.linear, nn.Linear):
        model.linear = nn.Linear(model.linear.in_features, num_classes).to(device)
        head = model.linear
    elif hasattr(model, 'network') and isinstance(model.network, nn.Sequential):
        head = _replace_last_linear_in_sequential(model.network, num_classes)
        head.to(device)
    else:
        raise ValueError("Could not locate a classifier layer to replace for coarse labels.")

    if hasattr(model, 'num_classes'):
        model.num_classes = num_classes

    return head


def train_coarse_head(model: nn.Module, head: nn.Module, train_loader: DataLoader, device: str,
                      epochs: int, lr: float, weight_decay: float):
    """Train only the classifier head for coarse labels; keep backbone frozen."""
    model.to(device)
    head.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True

    model.eval()
    head.train()

    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = running_correct / max(total, 1)
        print(f"  Head epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")


def load_linear_model(model_name: str, dataset_name: str, checkpoint_path: str = None):
    """Load a trained linear model from checkpoint."""
    
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / f"{model_name}_{dataset_name}_best.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"\n{'='*80}")
        print(f"ERROR: Linear model checkpoint not found!")
        print(f"{'='*80}")
        print(f"Looking for: {checkpoint_path}")
        print(f"\nProject structure:")
        print(f"  Script location: {SCRIPT_DIR}")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  Models directory: {MODELS_DIR}")
        print(f"\nAvailable models in {MODELS_DIR}:")
        if MODELS_DIR.exists():
            models = list(MODELS_DIR.glob('*.pth'))
            if models:
                for model in models:
                    print(f"  - {model.name}")
            else:
                print(f"  (no models found)")
        else:
            print(f"  (directory does not exist)")
        print(f"\nTo train this model, run:")
        print(f"  cd {PROJECT_ROOT / 'bin'}")
        print(f"  ./train_linear.py --model {model_name} --dataset {dataset_name}")
        print(f"{'='*80}\n")
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    model = create_linear_model(
        model_name,
        input_size=checkpoint['input_size'],
        num_classes=checkpoint['num_classes']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded linear model from: {checkpoint_path}")
    print(f"  Test accuracy: {checkpoint['test_acc']:.4f}")
    
    return model


def load_cnn_model(model_name: str, dataset_name: str, checkpoint_path: str = None):
    """Load a trained CNN model from checkpoint."""
    
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / f"{model_name}_{dataset_name}_best.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"\n{'='*80}")
        print(f"WARNING: CNN model checkpoint not found!")
        print(f"{'='*80}")
        print(f"Looking for: {checkpoint_path}")
        print(f"\nProject structure:")
        print(f"  Script location: {SCRIPT_DIR}")
        print(f"  Project root: {PROJECT_ROOT}")
        print(f"  Models directory: {MODELS_DIR}")
        print(f"\nAvailable models in {MODELS_DIR}:")
        if MODELS_DIR.exists():
            models = list(MODELS_DIR.glob('*.pth'))
            if models:
                for model_path in sorted(models):
                    print(f"  - {model_path.name}")
            else:
                print(f"  (no .pth files found)")
        else:
            print(f"  (directory does not exist)")
        
        print(f"\n{'='*80}")
        print(f"To train this model, run:")
        print(f"  ./train_cnn.py --model {model_name} --dataset {dataset_name} --epochs 100")
        print(f"{'='*80}")
        print(f"\nCreating UNTRAINED model with random weights...")
        print(f"⚠️  WARNING: Results will be meaningless with untrained models!")
        print(f"{'='*80}\n")
        
        # Create untrained model as fallback
        checkpoint = torch.load(checkpoint_path) if checkpoint_path.exists() else None
        if dataset_name in ['cifar10', 'mnist', 'svhn']:
            num_classes = 10
        elif dataset_name == 'cifar100':
            num_classes = 100
        elif dataset_name == 'cifar100_coarse':
            num_classes = 20
        elif dataset_name == 'imagenet':
            num_classes = 1000
        elif dataset_name == 'tinyimagenet':
            num_classes = 200
        else:
            num_classes = 10
        model = create_cnn_model(model_name, num_classes=num_classes)
        return model
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model info
    model_name_ckpt = checkpoint['model_name']
    dataset_name_ckpt = checkpoint['dataset_name']
    num_classes = checkpoint['num_classes']
    test_acc = checkpoint.get('test_acc', 0.0)
    
    # Create model
    model = create_cnn_model(model_name_ckpt, num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded CNN model: {model_name}")
    print(f"  Dataset: {dataset_name_ckpt}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    return model


def load_imagenet_model(model_name: str, weights_name: str = 'default'):
    """Load a torchvision ImageNet model with pretrained weights."""
    weights_name = weights_name.lower()
    imagenet_models = {
        # ResNet
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT), #Check
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
        # VGG
        'vgg11': (models.vgg11, models.VGG11_Weights.DEFAULT), #Check
        'vgg11_bn': (models.vgg11_bn, models.VGG11_BN_Weights.DEFAULT),
        'vgg13': (models.vgg13, models.VGG13_Weights.DEFAULT),
        'vgg13_bn': (models.vgg13_bn, models.VGG13_BN_Weights.DEFAULT),
        'vgg16': (models.vgg16, models.VGG16_Weights.DEFAULT),
        'vgg16_bn': (models.vgg16_bn, models.VGG16_BN_Weights.DEFAULT),
        'vgg19': (models.vgg19, models.VGG19_Weights.DEFAULT),
        'vgg19_bn': (models.vgg19_bn, models.VGG19_BN_Weights.DEFAULT),
        # DenseNet
        'densenet121': (models.densenet121, models.DenseNet121_Weights.DEFAULT), #Check
        'densenet161': (models.densenet161, models.DenseNet161_Weights.DEFAULT),
        'densenet169': (models.densenet169, models.DenseNet169_Weights.DEFAULT),
        'densenet201': (models.densenet201, models.DenseNet201_Weights.DEFAULT),
        # MobileNet
        'mobilenet_v2': (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT), #Check
        'mobilenet_v3_small': (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT),
        'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
        # ShuffleNet
        'shufflenet_v2_x0_5': (models.shufflenet_v2_x0_5, models.ShuffleNet_V2_X0_5_Weights.DEFAULT), #Check
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
        'shufflenet_v2_x1_5': (models.shufflenet_v2_x1_5, models.ShuffleNet_V2_X1_5_Weights.DEFAULT),
        'shufflenet_v2_x2_0': (models.shufflenet_v2_x2_0, models.ShuffleNet_V2_X2_0_Weights.DEFAULT),
        # SqueezeNet
        'squeezenet1_0': (models.squeezenet1_0, models.SqueezeNet1_0_Weights.DEFAULT),
        'squeezenet1_1': (models.squeezenet1_1, models.SqueezeNet1_1_Weights.DEFAULT),
        # EfficientNet
        'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        'efficientnet_b1': (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        'efficientnet_b2': (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        'efficientnet_b3': (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT), #Check
        'efficientnet_b4': (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
        'efficientnet_b5': (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
        'efficientnet_b6': (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
        'efficientnet_b7': (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        # ConvNeXt
        'convnext_tiny': (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        'convnext_small': (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),
        'convnext_base': (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT),
        'convnext_large': (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT),
        # ViT
        'vit_b_16': (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        'vit_b_32': (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
        'vit_l_16': (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT),
        'vit_l_32': (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT),
        'vit_h_14': (models.vit_h_14, models.ViT_H_14_Weights.DEFAULT),
        # Swin
        'swin_t': (models.swin_t, models.Swin_T_Weights.DEFAULT),
        'swin_s': (models.swin_s, models.Swin_S_Weights.DEFAULT),
        'swin_b': (models.swin_b, models.Swin_B_Weights.DEFAULT),
    }

    if model_name not in imagenet_models:
        available = ', '.join(sorted(imagenet_models.keys()))
        raise ValueError(f"Unknown ImageNet model: {model_name}. Available: {available}")

    builder, default_weights = imagenet_models[model_name]
    weights = None if weights_name in ['none', 'null', 'no', 'false'] else default_weights
    model = builder(weights=weights)
    model.eval()
    return model


def get_conv_model(model_name: str, num_classes: int = 10):
    """
    Create a CNN model (kept for backward compatibility).
    Use load_cnn_model() for loading trained models.
    """
    return create_cnn_model(model_name, num_classes=num_classes)


def is_linear_model(model_name: str) -> bool:
    """Check if model name refers to a linear model."""
    return model_name.startswith('linear_')


def get_model(model_name: str, dataset_name: str, num_classes: int, model_type: str = 'auto',
              imagenet_weights: str = 'default'):
    """
    Load a model (either convolutional or linear).
    
    Args:
        model_name: Name of the model
        dataset_name: Dataset name (needed for loading trained linear models)
        num_classes: Number of output classes
        model_type: 'conv', 'linear', or 'auto' (auto-detect from name)
    """
    
    if model_type == 'auto':
        model_type = 'linear' if is_linear_model(model_name) else 'conv'
    
    if model_type == 'linear':
        model = load_linear_model(model_name, dataset_name)
    elif model_type == 'conv':
        if dataset_name == 'imagenet':
            model = load_imagenet_model(model_name, weights_name=imagenet_weights)
        else:
            model = load_cnn_model(model_name, dataset_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, model_type


def get_dataset_for_model_type(dataset_name: str, model_type: str, train: bool = True,
                               subset_size: int = None, imagenet_root: Path = None,
                               tinyimagenet_root: Path = None):
    """
    Load dataset with appropriate transforms for model type.
    
    Args:
        dataset_name: Name of dataset
        model_type: 'conv' or 'linear'
        train: Whether to load train or test split
        subset_size: Optional subset size
    """
    
    if dataset_name == 'imagenet' and train:
        print("ImageNet requested with train=True; using validation split instead.")
        train = False

    if model_type == 'conv':
        # Convolutional models use native 32x32 resolution for CIFAR
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
            ])
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        elif dataset_name == 'cifar100_coarse':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        elif dataset_name == 'svhn':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.438, 0.444, 0.473], std=[0.198, 0.201, 0.197])
            ])
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.Pad(2),  # 28→32
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.131, 0.131, 0.131], std=[0.308, 0.308, 0.308])
            ])
        elif dataset_name == 'imagenet':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        elif dataset_name == 'tinyimagenet':
            if train:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(64),
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    elif model_type == 'linear':
        if dataset_name == 'imagenet':
            raise ValueError("imagenet is only supported for convolutional models.")
        # Linear models use original image sizes with simpler transforms
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        elif dataset_name == 'cifar100_coarse':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset_name == 'svhn':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        elif dataset_name == 'tinyimagenet':
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load dataset
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
        num_classes = 100
    elif dataset_name == 'cifar100_coarse':
        dataset = CIFAR100Coarse(root='./data', train=train, download=True, transform=transform)
        num_classes = 20
    elif dataset_name == 'svhn':
        split = 'train' if train else 'test'
        dataset = datasets.SVHN(root='./data', split=split, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'mnist':
        dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'imagenet':
        if imagenet_root is None:
            imagenet_root = IMAGENET_DEFAULT_ROOT
        imagenet_root = Path(imagenet_root)
        split = 'train' if train else 'val'
        split_dir = imagenet_root / split
        if split_dir.exists():
            dataset = datasets.ImageFolder(root=split_dir, transform=transform)
        else:
            if not imagenet_root.exists():
                raise FileNotFoundError(
                    f"ImageNet root not found: {imagenet_root}. "
                    f"Pass --imagenet-root with a directory containing train/ and val/."
                )
            try:
                dataset = datasets.ImageNet(root=imagenet_root, split=split, transform=transform)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not load ImageNet from {imagenet_root}. "
                    f"Expected subfolders train/ and val/ (ImageFolder) or ImageNet devkit structure."
                ) from exc
        num_classes = 1000
    elif dataset_name == 'tinyimagenet':
        if tinyimagenet_root is None:
            tinyimagenet_root = TINYIMAGENET_DEFAULT_ROOT
        tinyimagenet_root = Path(tinyimagenet_root)
        split = 'train' if train else 'val'
        split_dir = tinyimagenet_root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"TinyImageNet split not found: {split_dir}. "
                f"Pass --tinyimagenet-root with a directory containing train/ and val/."
            )
        if not train:
            val_images_dir = tinyimagenet_root / 'val' / 'images'
            if val_images_dir.exists():
                if any(val_images_dir.iterdir()):
                    raise FileNotFoundError(
                        f"TinyImageNet val/images still contains files at {val_images_dir}. "
                        "Please reorganize val into class subfolders before evaluation."
                    )
                val_images_dir.rmdir()
        dataset = datasets.ImageFolder(root=split_dir, transform=transform)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if subset_size is not None and subset_size < len(dataset):
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        dataset = Subset(dataset, indices)
    
    return dataset, num_classes


def evaluate_model_accuracy(model: nn.Module, data_loader: DataLoader, device: str,
                            max_samples: int = None) -> float:
    """Compute top-1 accuracy for a model on a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if max_samples is not None and total >= max_samples:
                break
    return correct / max(total, 1)




def run_stitching_experiment(model1_name: str, model2_name: str, dataset_name: str, args):
    """Run stitching experiment between two models (conv, linear, or mixed)."""
    
    t0 = time.perf_counter()
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
    else:
        device = args.device
    t_device = time.perf_counter()
    
    print(f"\n{'='*80}")
    print(f"Stitching: {model1_name} -> {model2_name} on {dataset_name}")
    print(f"{'='*80}")
    
    # Determine model types
    model1_type = 'linear' if is_linear_model(model1_name) else 'conv'
    model2_type = 'linear' if is_linear_model(model2_name) else 'conv'
    
    print(f"Model 1 type: {model1_type}")
    print(f"Model 2 type: {model2_type}")

    if dataset_name == 'imagenet' and (model1_type == 'linear' or model2_type == 'linear'):
        raise ValueError("imagenet experiments require convolutional models only (no linear models).")
    
    # Determine which transform to use (prefer conv if mixed, as it's more complex)
    dataset_model_type = 'conv' if (model1_type == 'conv' or model2_type == 'conv') else 'linear'
    
    # Load dataset
    dataset, num_classes = get_dataset_for_model_type(
        dataset_name, dataset_model_type, train=True, subset_size=args.subset_size,
        imagenet_root=args.imagenet_root, tinyimagenet_root=args.tinyimagenet_root
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    t_data = time.perf_counter()
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load models
    model_dataset_name = getattr(args, 'model_dataset', dataset_name)
    if dataset_name == 'cifar100_coarse':
        if model_dataset_name == dataset_name:
            print("Note: cifar100_coarse uses cifar100 checkpoints by default.")
        model_dataset_name = 'cifar100'
    print(f"\nLoading {model1_name}...")
    model1, _ = get_model(
        model1_name, model_dataset_name, num_classes, model_type=model1_type,
        imagenet_weights=args.imagenet_weights
    )
    model1 = model1.to(device)
    
    # If comparing model to itself, use the same object
    if model1_name == model2_name:
        print(f"\nUsing same model object for {model2_name} (self-comparison)")
        model2 = model1
    else:
        print(f"\nLoading {model2_name}...")
        model2, _ = get_model(
            model2_name, model_dataset_name, num_classes, model_type=model2_type,
            imagenet_weights=args.imagenet_weights
        )
        model2 = model2.to(device)
    t_models = time.perf_counter()

    if dataset_name == 'cifar100_coarse' and model_dataset_name == 'cifar100':
        if args.coarse_head_epochs > 0:
            print(f"\nAdapting classifier to 20 coarse classes (frozen backbone, epochs={args.coarse_head_epochs})")
            head1 = replace_classifier_for_coarse(model1, num_classes=20)
            train_coarse_head(model1, head1, train_loader, device,
                              epochs=args.coarse_head_epochs,
                              lr=args.coarse_head_lr,
                              weight_decay=args.coarse_head_weight_decay)
            if model2 is not model1:
                head2 = replace_classifier_for_coarse(model2, num_classes=20)
                train_coarse_head(model2, head2, train_loader, device,
                                  epochs=args.coarse_head_epochs,
                                  lr=args.coarse_head_lr,
                                  weight_decay=args.coarse_head_weight_decay)
        else:
            print("\nSkipping coarse head training (coarse_head_epochs=0)")
    
    # Set to eval mode
    model1.eval()
    model2.eval()
    
    # Print initial accuracy for ImageNet models (no checkpoint test_acc available)
    if dataset_name == 'imagenet':
        max_eval_samples = args.max_samples if args.max_samples is not None else None
        print(f"\nEvaluating initial ImageNet accuracy (max_samples={max_eval_samples})...")
        acc1 = evaluate_model_accuracy(model1, eval_loader, device, max_samples=max_eval_samples)
        print(f"  Model 1 ({model1_name}) top-1 acc: {acc1:.4f}")
        if model1 is not model2:
            acc2 = evaluate_model_accuracy(model2, eval_loader, device, max_samples=max_eval_samples)
            print(f"  Model 2 ({model2_name}) top-1 acc: {acc2:.4f}")
    elif dataset_name == 'tinyimagenet':
        max_eval_samples = args.max_samples if args.max_samples is not None else None
        print(f"\nEvaluating initial TinyImageNet accuracy (max_samples={max_eval_samples})...")
        acc1 = evaluate_model_accuracy(model1, eval_loader, device, max_samples=max_eval_samples)
        print(f"  Model 1 ({model1_name}) top-1 acc: {acc1:.4f}")
        if model1 is not model2:
            acc2 = evaluate_model_accuracy(model2, eval_loader, device, max_samples=max_eval_samples)
            print(f"  Model 2 ({model2_name}) top-1 acc: {acc2:.4f}")

    # Determine input shape and layer filter
    if dataset_model_type == 'conv':
        # CIFAR/SVHN use native 32x32; ImageNet uses 224x224; TinyImageNet uses 64x64
        if dataset_name == 'imagenet':
            input_shape = (3, 224, 224)
        elif dataset_name == 'tinyimagenet':
            input_shape = (3, 64, 64)
        else:
            input_shape = (3, 32, 32)
    else:
        # For linear models, use original image size
        if dataset_name in ['cifar10', 'cifar100', 'svhn']:
            input_shape = (3, 32, 32)
        elif dataset_name == 'cifar100_coarse':
            input_shape = (3, 32, 32)
        elif dataset_name == 'mnist':
            input_shape = (1, 28, 28)
        elif dataset_name == 'tinyimagenet':
            input_shape = (3, 64, 64)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Set layer filter based on model types
    if model1_type == 'linear' and model2_type == 'linear':
        layer_filter = 'linear'
    elif model1_type == 'conv' and model2_type == 'conv':
        layer_filter = 'conv'
    else:
        # Mixed: compare all layers
        layer_filter = 'all'
    
    print(f"\nInput shape: {input_shape}")
    print(f"Layer filter: {layer_filter}")
    print(f"Stitcher type: {args.stitcher_type}")
    
    # Setup output directory (same for all stitcher types)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exp_name = f"{model1_name}_{model2_name}_{dataset_name}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which stitcher types to run
    if args.stitcher_type == 'both':
        stitcher_types = ['affine', 'orthogonal']
    elif args.stitcher_type == 'all':
        stitcher_types = ['affine', 'orthogonal', 'orthogonal_scaled']
    else:
        stitcher_types = [args.stitcher_type]
    
    # Create stitcher (with first type as default)
    stitcher = ImprovedModelStitcher(device=device, 
                                     similarity_aggregation=args.similarity_aggregation,
                                     stitcher_type=stitcher_types[0],
                                     max_features_for_similarity=args.max_features_similarity,
                                     train_loss=args.train_loss,
                                     kl_temperature=args.kl_temperature,
                                     use_block_outputs=not args.all_layers,
                                     use_amp=args.amp)  # Default: True (block outputs only)
    
    print(f"Computing with stitcher types: {', '.join(stitcher_types)}")
    
    # Compute stitching matrix - pass stitcher_types to compute them consecutively per layer pair
    t_stitch_start = time.perf_counter()
    result = stitcher.compute_stitching_matrix(
        model1, model2, train_loader, input_shape,
        num_epochs=args.epochs, layer_filter=layer_filter, 
        max_samples=args.max_samples,
        max_samples_similarity=args.max_samples_similarity,
        stitcher_types=stitcher_types,  # Pass list of stitcher types
        verbose=True
    )
    t_stitch_end = time.perf_counter()
    
    # Handle different return formats
    if len(stitcher_types) > 1:
        # Multiple stitchers - result is a dictionary
        all_results = result
        layers1 = all_results['source_layers']
        layers2 = all_results['target_layers']
        similarity_matrices = all_results['similarity_matrices']
    else:
        # Single stitcher - result is tuple
        accuracy_ratio_matrix, stitched_acc_matrix, target_acc_matrix, entropy_ratio_matrix, similarity_matrices, layers1, layers2 = result
        all_results = {
            stitcher_types[0]: {
                'accuracy_ratio': accuracy_ratio_matrix,
                'stitched_accuracy': stitched_acc_matrix,
                'target_accuracy': target_acc_matrix,
                'entropy_ratio': entropy_ratio_matrix,
            },
            'similarity_matrices': similarity_matrices,
            'source_layers': layers1,
            'target_layers': layers2
        }
    
    # Check if we have any valid results
    if not layers1 or not layers2:
        print("\nERROR: No valid results to save.")
        return
    
    # Save all results to the SAME directory with prefixed filenames
    print(f"\nSaving all results to: {exp_dir}")
    
    for stitcher_type in stitcher_types:
        if stitcher_type in all_results:
            results = all_results[stitcher_type]
            # Prefix for this stitcher type
            prefix = f"{stitcher_type}_"
            
            # Save all matrices
            np.save(exp_dir / f'{prefix}accuracy_ratio_matrix.npy', results['accuracy_ratio'])
            np.save(exp_dir / f'{prefix}stitched_accuracy_matrix.npy', results['stitched_accuracy'])
            np.save(exp_dir / f'{prefix}target_accuracy_matrix.npy', results['target_accuracy'])
            np.save(exp_dir / f'{prefix}stitched_ce_matrix.npy', results['stitched_ce'])
            np.save(exp_dir / f'{prefix}target_ce_matrix.npy', results['target_ce'])
            np.save(exp_dir / f'{prefix}ce_ratio_matrix.npy', results['ce_ratio'])
            np.save(exp_dir / f'{prefix}entropy_ratio_matrix.npy', results['entropy_ratio'])
            np.save(exp_dir / f'{prefix}invertibility_matrix.npy', results['invertibility'])
            np.save(exp_dir / f'{prefix}numerical_rank_matrix.npy', results['numerical_rank'])
            np.save(exp_dir / f'{prefix}reconstruction_mse_matrix.npy', results['reconstruction_mse'])
            np.save(exp_dir / f'{prefix}stitched_repr_rank_matrix.npy', results['stitched_repr_rank'])
            
            print(f"  Saved {stitcher_type} results: {prefix}*.npy (accuracy, CE, entropy ratio, invertibility, ranks, MSE)")
    
    # Save similarity metrics ONCE (same for all stitcher types)
    np.save(exp_dir / 'cka_before_matrix.npy', similarity_matrices['cka_before'])
    np.save(exp_dir / 'rsa_before_matrix.npy', similarity_matrices['rsa_before'])
    np.save(exp_dir / 'svcca_before_matrix.npy', similarity_matrices['svcca_before'])
    np.save(exp_dir / 'cca_before_matrix.npy', similarity_matrices['cca_before'])
    np.save(exp_dir / 'l2_before_matrix.npy', similarity_matrices['l2_before'])
    np.save(exp_dir / 'procrustes_before_matrix.npy', similarity_matrices['procrustes_before'])
    np.save(exp_dir / 'orthogonal_scaled_before_matrix.npy', similarity_matrices['orthogonal_scaled_before'])
    np.save(exp_dir / 'invertible_affine_before_matrix.npy', similarity_matrices['invertible_affine_before'])
    
    # Save representation ranks (source & target are shared, stitched is per-stitcher)
    np.save(exp_dir / 'source_repr_rank_matrix.npy', similarity_matrices['source_repr_rank'])
    np.save(exp_dir / 'target_repr_rank_matrix.npy', similarity_matrices['target_repr_rank'])
    
    print(f"  Saved similarity metrics (CKA, RSA, SVCCA, AffineCCA, L2, Procrustes, Orth+Scale)")
    print(f"  Saved representation ranks (source, target; stitched saved per-stitcher)")
    
    # Save metadata
    metadata = {
        'model1': model1_name,
        'model2': model2_name,
        'model1_type': model1_type,
        'model2_type': model2_type,
        'dataset': dataset_name,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'subset_size': args.subset_size,
        'max_samples': args.max_samples,
        'layers1': layers1,
        'layers2': layers2,
        'num_classes': num_classes,
        'layer_filter': layer_filter,
        'similarity_aggregation': args.similarity_aggregation,
        'stitcher_types': stitcher_types,  # List of stitcher types used
        'input_shape': list(input_shape),
        'coarse_head_epochs': args.coarse_head_epochs,
        'coarse_head_lr': args.coarse_head_lr,
        'coarse_head_weight_decay': args.coarse_head_weight_decay
    }
    
    with open(exp_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    t_save = time.perf_counter()
    
    print(f"\nSummary:")
    print(f"  Directory: {exp_dir}")
    print(f"  Stitcher types: {', '.join(stitcher_types)}")
    _ = (t_device, t0, t_data, t_models, t_stitch_start, t_stitch_end, t_save)
    
    # Print comparison if both stitchers were run
    if len(stitcher_types) > 1 and 'affine' in all_results and 'orthogonal' in all_results:
        print(f"\n{'='*80}")
        print(f"COMPARISON: Affine vs Orthogonal Stitching")
        print(f"{'='*80}")
        
        affine_acc = all_results['affine']['accuracy_ratio']
        ortho_acc = all_results['orthogonal']['accuracy_ratio']
        diff = affine_acc - ortho_acc
        
        print(f"\nAffine Stitching:")
        print(f"  Best:  {affine_acc.max():.4f}")
        print(f"  Mean:  {affine_acc.mean():.4f}")
        print(f"  Worst: {affine_acc.min():.4f}")
        
        print(f"\nOrthogonal Stitching:")
        print(f"  Best:  {ortho_acc.max():.4f}")
        print(f"  Mean:  {ortho_acc.mean():.4f}")
        print(f"  Worst: {ortho_acc.min():.4f}")
        
        print(f"\nDifference (Affine - Orthogonal):")
        print(f"  Max:   {diff.max():.4f}")
        print(f"  Mean:  {diff.mean():.4f}")
        print(f"  Min:   {diff.min():.4f}")
        
        # Interpretation
        mean_diff = diff.mean()
        if mean_diff < 0.05:
            print(f"\n✅ Representations are ORTHOGONALLY RELATED (mean diff: {mean_diff:.3f})")
        elif mean_diff < 0.15:
            print(f"\n⚠️  Representations are MOSTLY ORTHOGONAL (mean diff: {mean_diff:.3f})")
        else:
            print(f"\n❌ Representations REQUIRE AFFINE TRANSFORMATION (mean diff: {mean_diff:.3f})")
        
        print(f"\nBoth results saved in: {exp_dir}")
        print(f"  - affine_accuracy_ratio_matrix.npy")
        print(f"  - orthogonal_accuracy_ratio_matrix.npy")


def main():
    parser = argparse.ArgumentParser(description='Extended model stitching with linear model support')
    
    parser.add_argument('--mode', type=str, required=True, choices=['single'],
                       help='Experiment mode (currently only single supported)')
    
    parser.add_argument('--model1', type=str, required=True,
                       help='Source model (conv: resnet18/vgg11/etc, linear: linear_small/linear_medium/etc)')
    
    parser.add_argument('--model2', type=str, required=True,
                       help='Target model (conv: resnet18/vgg11/etc, linear: linear_small/linear_medium/etc)')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'cifar100_coarse', 'mnist', 'svhn', 'imagenet', 'tinyimagenet'],
                       help='Dataset to use')

    parser.add_argument('--model-dataset', type=str, default=None,
                       choices=['cifar10', 'cifar100', 'cifar100_coarse', 'mnist', 'svhn', 'tinyimagenet'],
                       help='Dataset name used to load model checkpoints (default: same as --dataset)')

    parser.add_argument('--coarse-head-epochs', type=int, default=5,
                       help='Epochs to train new 20-class head when using cifar100_coarse (default: 5)')

    parser.add_argument('--coarse-head-lr', type=float, default=0.01,
                       help='Learning rate for coarse head training (default: 0.01)')

    parser.add_argument('--coarse-head-weight-decay', type=float, default=0.0,
                       help='Weight decay for coarse head training (default: 0.0)')
    
    parser.add_argument('--epochs', type=int, default=10,
                       help='Epochs for stitching layer training (default: 10)')
    
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    parser.add_argument('--subset-size', type=int, default=None,
                       help='Use subset of dataset (None = full dataset)')

    parser.add_argument('--imagenet-root', type=str, default=str(IMAGENET_DEFAULT_ROOT),
                       help='ImageNet root directory (expects train/ and val/ subfolders)')

    parser.add_argument('--imagenet-weights', type=str, default='default',
                       help='ImageNet pretrained weights: default or none')

    parser.add_argument('--tinyimagenet-root', type=str, default=str(TINYIMAGENET_DEFAULT_ROOT),
                       help='TinyImageNet root directory (expects train/ and val/ subfolders)')
    
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum samples for stitcher training/evaluation (default: 50000, None = use all)')
    
    parser.add_argument('--max-samples-similarity', type=int, default=10000,
                       help='Maximum samples for similarity computation (default: 10000). Use lower value to speed up similarity computation.')
    
    parser.add_argument('--max-features-similarity', type=int, default=2048,
                       help='Maximum number of features for slow similarity metrics (RSA, L2, Procrustes, Orth+Scale). '
                            'CKA and SVCCA always computed. Slow metrics only if features <= this value (default: 2048)')
    
    parser.add_argument('--output-dir', type=str, default=str(EXPERIMENTS_DIR),
                       help='Output directory (default: PROJECT_ROOT/experiments)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    parser.add_argument('--similarity-aggregation', type=str, default='flatten',
                       choices=['gap', 'flatten', 'spatial_samples'],
                       help='Spatial aggregation method for similarity metrics')

    parser.add_argument('--train-loss', type=str, default='ce',
                       choices=['ce', 'kl'],
                       help='Training loss for stitcher: ce (labels) or kl (match target logits)')

    parser.add_argument('--kl-temperature', type=float, default=1.0,
                       help='Temperature for KL loss (only used with --train-loss kl)')

    parser.add_argument('--amp', action='store_true',
                       help='Enable automatic mixed precision for stitcher training (CUDA only)')
    
    parser.add_argument('--all-layers', action='store_true',
                       help='Use ALL layers instead of just block outputs (much slower). '
                            'By default, only block outputs are used: ResNet ~9 layers, MobileNet ~15 layers')
    
    parser.add_argument('--stitcher-type', type=str, default='all',
                       choices=['affine', 'orthogonal', 'orthogonal_scaled', 'both', 'all'],
                       help='Type of stitcher: affine (Wx+b), orthogonal (Qx), orthogonal_scaled (sQx), both (affine+orthogonal), or all (affine+orthogonal+orthogonal_scaled). Default: all')
    
    args = parser.parse_args()
    
    # Run experiment
    if args.mode == 'single':
        run_stitching_experiment(args.model1, args.model2, args.dataset, args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
