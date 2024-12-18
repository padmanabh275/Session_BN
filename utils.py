import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import seaborn as sns
import os

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def remove_old_plots(file_to_remove=None):
    """Remove old plot files if they exist"""
    if file_to_remove:
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
    else:
        plot_files = ['accuracy_curves.png', 'loss_curves.png', 
                      'misclassified_images.png', 'confusion_matrix.png']
        for file in plot_files:
            if os.path.exists(file):
                os.remove(file)

def plot_accuracy_curves(train_accuracies, test_accuracies, save_path='accuracy_curves.png'):
    """Plot training and testing accuracy curves"""
    remove_old_plots(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_loss_curves(train_losses, save_path='loss_curves.png'):
    """Plot training loss curve"""
    remove_old_plots(save_path)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def get_misclassified_images(model, device, test_loader, num_images=20):
    """Get misclassified images from a batch"""
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Get indices of misclassified images
            misclassified_idx = (pred != target).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                if len(misclassified_images) >= num_images:
                    break
                    
                misclassified_images.append(data[idx].cpu())
                misclassified_labels.append(classes[target[idx].item()])
                predicted_labels.append(classes[pred[idx].item()])
                
            if len(misclassified_images) >= num_images:
                break
    
    return misclassified_images, misclassified_labels, predicted_labels

def plot_misclassified_images(images, true_labels, pred_labels, save_path='misclassified_images.png'):
    """Plot misclassified images in a grid"""
    remove_old_plots(save_path)
    fig = plt.figure(figsize=(15, 12))
    for i in range(min(20, len(images))):
        plt.subplot(4, 5, i + 1)
        
        # Denormalize the image
        img = images[i].permute(1, 2, 0)
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        plt.imshow(img)
        plt.title(f'True: {true_labels[i]}\nPred: {pred_labels[i]}', fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_confusion_matrix(model, device, test_loader, save_path='confusion_matrix.png'):
    """Create and plot confusion matrix"""
    remove_old_plots(save_path)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    confusion_matrix = torch.zeros(10, 10)
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix.numpy(), 
                xticklabels=classes,
                yticklabels=classes,
                annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close() 