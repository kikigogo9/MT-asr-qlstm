import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import os

class PyTorchTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            loss_fn: nn.Module,
            optimizer: Optimizer,
            val_loader: DataLoader|None =None,
            epochs:int=10,
            device:str='cpu',
            save_best=True,
            checkpoint_dir='checkpoints', 
            monitor='val',
            model_name='best_model'
            ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model_name = model_name
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        
        # Best model saving
        self.save_best = save_best
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = float('inf')
        self.monitor = monitor
        
        # Validate monitoring mode
        if self.monitor == 'val' and val_loader is None:
            raise ValueError("Cannot monitor validation loss without validation loader")
            
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _train_epoch(self):
        self.model.train()
        #def closure():
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            # Forward pass
            outputs = self.model(data)
            loss = self.loss_fn(outputs, target)
            
            # Backward pass and optimize
            loss.backward()
            
            self.optimizer.step() 

            running_loss += loss.item() * data.size(0)
             
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        
                
        return self.train_losses[-1]

    def _validate_epoch(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.loss_fn(outputs, target)
                running_loss += loss.item() * data.size(0)
                
        epoch_loss = running_loss / len(self.val_loader.dataset)
        self.val_losses.append(epoch_loss)
        return epoch_loss

    def _save_checkpoint(self):
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses,
                'validation_loss': self.val_losses,
            }, checkpoint_path)
            
    def load_best_model(self) -> nn.Module|None:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_name}.pth")
        
        if  not os.path.exists(checkpoint_path):
            return None
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.val_losses = checkpoint['validation_loss']
        self.train_losses = checkpoint['train_loss']
        return self.model    

    def fit(self, force_train=False):
        ### Try to load model
        
        # If model success
        if self.load_best_model() is not None and not force_train:
            return self.train_losses, self.val_losses
        
        
        progress_bar = tqdm(range(self.epochs), desc='Training')
        
        for epoch in progress_bar:
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            # Determine current metric
            current_metric = val_loss if self.monitor == 'val' else train_loss
            
            # Save best model
            if self.save_best and current_metric < self.best_metric:
                self.best_metric = current_metric
                self._save_checkpoint()
            
            # Update progress bar
            desc = f'Epoch {epoch+1}/{self.epochs}'
            desc += f' - Train loss: {train_loss:.4f}'
            if val_loss is not None:
                desc += f' - Val loss: {val_loss:.4f}'
            if self.save_best:
                desc += f' - Best {self.monitor}: {self.best_metric:.4f}'
            progress_bar.set_description(desc)
            
        return self.train_losses, self.val_losses
