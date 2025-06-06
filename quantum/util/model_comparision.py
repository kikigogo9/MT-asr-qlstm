import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer import PyTorchTrainer

def compare_models(trainer1: PyTorchTrainer, trainer2: PyTorchTrainer, dataloader: DataLoader, model_names=None, num_samples=100):
    """
    Compare two trained models by plotting:
    1. Training and validation loss curves side by side
    2. Actual vs predicted values for both models
    
    Args:
        trainer1: First PyTorchTrainer instance
        trainer2: Second PyTorchTrainer instance
        dataloader: DataLoader for getting prediction examples
        model_names: List of names for the models (default: ['Model 1', 'Model 2'])
        num_samples: Number of samples to show in prediction plot
    """
    # Handle model names
    if model_names is None:
        model_names = ['Model 1', 'Model 2']

    # Create figure for loss comparison
    plt.figure(figsize=(15, 6))
    
    # Plot training loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(trainer1.train_losses, label=f'{model_names[0]} Train')
    plt.plot(trainer2.train_losses, label=f'{model_names[1]} Train')
    if trainer1.val_losses:
        plt.plot(trainer1.val_losses, '--', label=f'{model_names[0]} Val')
    if trainer2.val_losses:
        plt.plot(trainer2.val_losses, '--', label=f'{model_names[1]} Val')
    plt.title('Training/Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Get prediction data
    data, targets = dataloader.dataset.X, dataloader.dataset.y
    data = data.to(trainer1.device)
    targets = targets.to(trainer1.device)
    
    # Get predictions
    trainer1.model.eval()
    trainer2.model.eval()
    with torch.no_grad():
        preds1 = trainer1.model(data).cpu().numpy()
        preds2 = trainer2.model(data).cpu().numpy()
    
    targets = targets.cpu().numpy()
    
    # Create figure for prediction comparison
    plt.subplot(1, 2, 2)
    
    # Plot predictions
    plt.plot(preds1, label=model_names[0])
    plt.plot(preds2, label=model_names[1])
    plt.plot(targets, label='True Data')
    # Plot perfect prediction line
    #max_val = max(np.max(targets), np.max(preds1), np.max(preds2))
    #min_val = min(np.min(targets), np.min(preds1), np.min(preds2))
    #plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.title('True vs Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()