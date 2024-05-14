# DataLoader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crops the image and pads if needed
        transforms.Resize(32),  # Ensure the image is resized (if needed)
        transforms.RandomHorizontalFlip(),  # Flips the image horizontally with a default 50% chance
        transforms.RandomRotation(15),  # Randomly rotates the image by up to 15 degrees
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly jitters color hues and saturation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation of the image
        transforms.ToTensor(),  # Converts the image to a Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalizes with mean and std for CIFAR-10
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def apply_test_transforms(image):
    transform_test = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    return transform_test(image)

def get_dataloaders(batch_size):
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=get_train_transforms())
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=get_test_transforms())
    
    train_size = 45000
    train_finetune_size = 5000
    train_dataset, train_finetune_dataset = random_split(trainset, [train_size, train_finetune_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    finetune_loader = DataLoader(train_finetune_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, finetune_loader, testloader
