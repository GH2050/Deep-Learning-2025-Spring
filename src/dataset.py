import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

# CIFAR-100 statistics
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

# ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_cifar100_transforms(use_imagenet_norm=False, augment=True):
    """
    Get training and testing transforms for CIFAR-100.
    Args:
        use_imagenet_norm (bool): If True, use ImageNet normalization. Otherwise, use CIFAR-100.
        augment (bool): If True, apply training augmentations. Otherwise, basic ToTensor and Normalize.
    Returns:
        transform_train, transform_test
    """
    mean = IMAGENET_MEAN if use_imagenet_norm else CIFAR100_MEAN
    std = IMAGENET_STD if use_imagenet_norm else CIFAR100_STD
    
    if augment:
        # Training transforms as per report (Section 3.3)
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(), # Good default, includes AutoAugment-like diversity
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0), # Cutout alternative
        ]
    else:
        # Basic transform for validation/testing or when no augmentation is needed
         transform_train_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]

    transform_test_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    
    return transforms.Compose(transform_train_list), transforms.Compose(transform_test_list)


def get_cifar100_datasets(data_dir='./data', use_imagenet_norm=False, download=True):
    """
    Gets the CIFAR-100 training and test datasets with specified transforms.
    Args:
        data_dir (str): Directory to store/load CIFAR-100 data.
        use_imagenet_norm (bool): Whether to use ImageNet normalization statistics.
        download (bool): Whether to download the dataset if not found.
    Returns:
        train_dataset, test_dataset (torchvision.datasets.CIFAR100 instances)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform_train, transform_test = get_cifar100_transforms(use_imagenet_norm=use_imagenet_norm, augment=True)
    
    train_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=True, 
        download=download, 
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, 
        train=False, 
        download=download, 
        transform=transform_test
    )
    
    return train_dataset, test_dataset


def get_dataloaders(batch_size=128, num_workers=2, use_imagenet_norm=False, data_dir='./data'):
    """
    Gets CIFAR-100 DataLoaders.
    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of worker processes for DataLoaders.
        use_imagenet_norm (bool): If True, use ImageNet normalization.
        data_dir (str): Directory for the dataset.
    Returns:
        trainloader, testloader
    """
    train_dataset, test_dataset = get_cifar100_datasets(
        data_dir=data_dir,
        use_imagenet_norm=use_imagenet_norm
    )
    
    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True # Often beneficial for training
    )
    
    testloader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2, # Typically use larger batch for testing if memory allows
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader

if __name__ == '__main__':
    # Test the dataloaders
    print("Testing CIFAR-100 Dataloader (CIFAR-100 Norm):")
    train_loader_cifar, test_loader_cifar = get_dataloaders(batch_size=4, use_imagenet_norm=False)
    
    for i, (images, labels) in enumerate(train_loader_cifar):
        print(f"CIFAR Norm - Batch {i+1}: Image shape: {images.shape}, Label: {labels[0].item()}")
        if i == 0: # Show one batch
            assert images.max().item() < 3 and images.min().item() > -3 # Check normalization roughly
            break
            
    print("\nTesting CIFAR-100 Dataloader (ImageNet Norm):")
    train_loader_imgnet, _ = get_dataloaders(batch_size=4, use_imagenet_norm=True)
    for i, (images, labels) in enumerate(train_loader_imgnet):
        print(f"ImageNet Norm - Batch {i+1}: Image shape: {images.shape}, Label: {labels[0].item()}")
        if i == 0: # Show one batch
            assert images.max().item() < 3 and images.min().item() > -3
            break
    
    # Check if datasets are returned correctly
    train_ds, test_ds = get_cifar100_datasets()
    print(f"\nTrain dataset length: {len(train_ds)}, Test dataset length: {len(test_ds)}")
    img, label = train_ds[0]
    print(f"Sample image type: {type(img)}, shape: {img.shape}, label: {label}")
    assert isinstance(img, torch.Tensor)

    # Test transform without augmentation
    transform_train_no_aug, transform_test_no_aug = get_cifar100_transforms(augment=False)
    
    ds_no_aug_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_no_aug)
    img_no_aug, _ = ds_no_aug_train[0]
    print(f"Sample image (no train aug) shape: {img_no_aug.shape}")
    assert img_no_aug.shape == (3,32,32)
    
    print("\nDataset script tests completed.") 