import torch
import pytest
from train import CIFAR10Albumentation, train_transform, test_transform

def test_dataset_transforms():
    dataset = CIFAR10Albumentation(root='./data', train=True, download=True, transform=train_transform)
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor), "Dataset output should be a tensor"
    assert img.shape == (3, 32, 32), f"Expected shape (3, 32, 32), got {img.shape}"
    assert isinstance(label, int), "Label should be an integer"

def test_transform_normalization():
    dataset = CIFAR10Albumentation(root='./data', train=False, download=True, transform=test_transform)
    img, _ = dataset[0]
    assert -3 <= img.min() <= 3 and -3 <= img.max() <= 3, "Image values should be normalized"

def test_dataloader_batch():
    dataset = CIFAR10Albumentation(root='./data', train=True, download=True, transform=train_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    batch = next(iter(dataloader))
    assert len(batch) == 2, "Batch should contain both images and labels"
    assert batch[0].shape == (128, 3, 32, 32), f"Expected shape (128, 3, 32, 32), got {batch[0].shape}" 