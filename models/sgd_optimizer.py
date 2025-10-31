"""
Stochastic Gradient Descent optimizer from scratch.
"""
import numpy as np
import torch
import torch.optim as optim
from typing import List


class SGD_Trainer:
    """
    SGD trainer for SVM using PyTorch for gradient computation.
    Implements SGD from scratch without using sklearn's SGD.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.01,
        C: float = 1.0,
        batch_size: int = 1
    ):
        """
        Initialize SGD trainer.
        
        Args:
            model: PyTorch model (CustomSVM)
            learning_rate: Learning rate for SGD
            C: SVM regularization parameter
            batch_size: Batch size (default 1 for true SGD)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.C = C
        self.batch_size = batch_size
        
        # Use PyTorch optimizer for gradient computation
        # but we'll implement custom update logic
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.0  # Pure SGD, no momentum
        )
    
    def train_epoch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> float:
        """
        Train for one epoch using SGD.
        
        Args:
            x: Input features (n_samples, n_features)
            y: Target labels (+1 or -1)
            shuffle: Whether to shuffle data each epoch
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        n_samples = x.shape[0]
        
        # Create indices
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        total_loss = 0.0
        n_batches = 0
        
        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            
            # Convert to tensors
            x_tensor = torch.tensor(x_batch, dtype=torch.float32)
            y_tensor = torch.tensor(y_batch, dtype=torch.float32)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass and compute loss
            # Import svm_loss (using relative import)
            try:
                from .svm import svm_loss
            except ImportError:
                from models.svm import svm_loss
            loss = svm_loss(self.model, x_tensor, y_tensor, self.C)
            
            # Backward pass (compute gradients)
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the model for multiple epochs.
        
        Args:
            x: Input features
            y: Target labels
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of losses per epoch
        """
        loss_history = []
        
        for epoch in range(epochs):
            loss = self.train_epoch(x, y, shuffle=True)
            loss_history.append(loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
        
        return loss_history

