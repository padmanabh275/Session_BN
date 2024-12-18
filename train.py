from model import CIFAR10Net

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchsummary import summary
from tqdm import tqdm
from utils import (AverageMeter, plot_accuracy_curves, plot_loss_curves,
                    get_misclassified_images, plot_misclassified_images,
                    create_confusion_matrix)

# Calculate dataset mean and std
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Define transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(
        max_holes=1, max_height=8, max_width=8,
        min_holes=1, min_height=8, min_width=8,
        fill_value=(int(CIFAR_MEAN[0] * 255), int(CIFAR_MEAN[1] * 255), int(CIFAR_MEAN[2] * 255)),
        p=0.3
    ),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ToTensorV2()
])

# Custom Dataset class for Albumentations
class CIFAR10Albumentation(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = self.transform(image=img)['image']
        return img, label

# Modified training function with tqdm
def train(model, device, train_loader, optimizer, criterion, epoch, scheduler):
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update running loss
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100. * correct / total
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{acc:.2f}%'
        })
    
    return acc, avg_loss

# Modified evaluation function with tqdm
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            pbar.set_postfix({'acc': f'{acc:.2f}%'})
    
    print(f'\nTest Accuracy: {acc:.2f}%')
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading
    train_dataset = CIFAR10Albumentation(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = CIFAR10Albumentation(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Model, optimizer and loss function
    model = CIFAR10Net().to(device)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model, input_size=(3, 32, 32))
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        div_factor=10,
        final_div_factor=100,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Add weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Initialize lists to store metrics
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    
    # Training loop
    best_acc = 0
    print("\nStarting Training:")
    print("=" * 50)
    
    for epoch in range(1, 31):
        print(f"\nEpoch {epoch}/30")
        train_acc, train_loss = train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
        test_acc = evaluate(model, device, test_loader)
        
        # Store metrics
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Testing  - Accuracy: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, 'best_model.pth')
            print(f"New best accuracy! Model saved.")
    
    # Plot training curves
    plot_accuracy_curves(train_accuracies, test_accuracies)
    plot_loss_curves(train_losses)
    
    # Get and plot misclassified images
    misclassified_images, true_labels, pred_labels = get_misclassified_images(
        model, device, test_loader)
    plot_misclassified_images(misclassified_images, true_labels, pred_labels)
    
    # Create confusion matrix
    create_confusion_matrix(model, device, test_loader)
    
    print("\n" + "=" * 50)
    print(f'Training completed. Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 