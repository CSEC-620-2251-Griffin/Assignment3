"""
Custom SVM implementation from scratch.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple



class CustomSVM(nn.Module):
    """
    Custom Support Vector Machine implementation.
    Uses PyTorch for gradient computation but implements SVM logic from scratch.
    """
    
    def __init__(self, n_features: int, C: float = 1.0, hard_margin: bool = False):
        """
        Initialize SVM model.
        
        Args:
            n_features: Number of features (vocabulary size)
            C: Soft margin regularization parameter (ignored if hard_margin=True)
            hard_margin: If True, use hard margin (C=inf)
        """
        super(CustomSVM, self).__init__()
        self.n_features = n_features
        self.C = C if not hard_margin else float('inf')
        self.hard_margin = hard_margin
        
        # Initialize weights and bias
        # Small random init to avoid all-zero gradients
        self.w = nn.Parameter(torch.randn(n_features, dtype=torch.float32) * 0.01)
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: prediction = w · x - b
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            Predictions of shape (batch_size,)
        """
        # w · x - b
        output = torch.matmul(x, self.w) - self.b
        return output
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            x: Input array of shape (n_samples, n_features)
            
        Returns:
            Binary predictions: +1 or -1
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.forward(x_tensor)
            predictions = torch.sign(outputs).numpy()
        return predictions
    
    def predict_proba_raw(self, x: np.ndarray) -> np.ndarray:
        """
        Get raw scores (w · x - b) without thresholding.
        
        Args:
            x: Input array of shape (n_samples, n_features)
            
        Returns:
            Raw scores
        """
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            outputs = self.forward(x_tensor)
            return outputs.numpy()


def hinge_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute hinge loss: max(0, 1 - t*y)
    where t is target (+1 or -1) and y is prediction score.
    
    Args:
        predictions: Raw model outputs (w · x - b)
        targets: Target labels (+1 or -1)
        
    Returns:
        Hinge loss per sample
    """
    # Hinge loss: max(0, 1 - t*y)
    # where t is target and y is prediction
    loss_per_sample = torch.clamp(1.0 - targets * predictions, min=0.0)
    return loss_per_sample


def svm_loss(
    model: CustomSVM,
    x: torch.Tensor,
    y: torch.Tensor,
    C: float
) -> torch.Tensor:
    """
    Compute complete SVM loss: hinge loss + regularization.
    
    Loss = (1/n) * sum(max(0, 1 - t*y)) + C * ||w||²
    
    Args:
        model: CustomSVM model
        x: Input features (batch_size, n_features)
        y: Target labels (+1 or -1)
        C: Regularization parameter
        
    Returns:
        Total loss (scalar)
    """
    # Forward pass
    predictions = model(x)
    
    # Hinge loss
    hinge = hinge_loss(predictions, y)
    data_loss = torch.mean(hinge)
    
    # Regularization: C * ||w||²
    regularization = C * torch.sum(model.w ** 2)
    
    # Total loss
    total_loss = data_loss + regularization
    
    return total_loss


def get_support_vectors(
    model: CustomSVM,
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify support vectors: samples where |y| < 1 (margin violations or on margin).
    
    Args:
        model: Trained SVM model
        x: Input features
        y: Target labels
        
    Returns:
        Tuple of (support_vector_features, support_vector_labels)
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        predictions = model.forward(x_tensor).numpy()
        
        # Support vectors: samples where |prediction * target| < 1
        # i.e., margin violations or samples on the margin
        margins = np.abs(predictions * y)
        support_mask = margins < 1.0 + 1e-6  # Small epsilon for numerical stability
        
        support_vectors_x = x[support_mask]
        support_vectors_y = y[support_mask]
    
    return support_vectors_x, support_vectors_y

