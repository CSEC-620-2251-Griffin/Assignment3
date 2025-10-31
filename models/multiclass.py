"""
Multi-class classification strategies: One-vs-All (OvA) and One-vs-One (OvO).
"""
import numpy as np
from typing import List, Dict, Tuple
from models.svm import CustomSVM
from models.sgd_optimizer import SGD_Trainer


class OneVsAll:
    """
    One-vs-All (OvA) multi-class classification.
    Trains one binary classifier per class.
    """
    
    def __init__(self, n_features: int, n_classes: int, C: float = 1.0):
        """
        Initialize OvA classifier.
        
        Args:
            n_features: Number of features
            n_classes: Number of classes
            C: SVM regularization parameter
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.C = C
        self.classifiers: List[CustomSVM] = []
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: List[str] = []
    
    def setup_classes(self, class_names: List[str]) -> None:
        """Setup class name mappings."""
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.index_to_class = class_names
    
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 10,
        verbose: bool = True
    ) -> List[List[float]]:
        """
        Train one binary classifier per class.
        
        Args:
            x: Input features (n_samples, n_features)
            y: Multi-class labels (class indices)
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            List of loss histories (one per classifier)
        """
        self.classifiers = []
        loss_histories = []
        
        for class_idx in range(self.n_classes):
            if verbose:
                print(f"\nTraining classifier {class_idx + 1}/{self.n_classes} (class: {self.index_to_class[class_idx]})")
            
            # Create binary labels: +1 for this class, -1 for others
            y_binary = np.where(y == class_idx, 1.0, -1.0)
            
            # Initialize classifier
            classifier = CustomSVM(self.n_features, C=self.C, hard_margin=False)
            trainer = SGD_Trainer(classifier, learning_rate=learning_rate, C=self.C)
            
            # Train
            loss_history = trainer.train(x, y_binary, epochs=epochs, verbose=False)
            loss_histories.append(loss_history)
            
            self.classifiers.append(classifier)
            
            if verbose:
                class_samples = np.sum(y == class_idx)
                print(f"  Class samples: {class_samples}/{len(y)}")
        
        return loss_histories
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class using OvA strategy.
        
        Args:
            x: Input features (n_samples, n_features)
            
        Returns:
            Predicted class indices
        """
        n_samples = x.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        
        # Get scores from each classifier
        for class_idx, classifier in enumerate(self.classifiers):
            scores[:, class_idx] = classifier.predict_proba_raw(x)
        
        # Predict class with highest score
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Get prediction scores for all classes.
        
        Args:
            x: Input features
            
        Returns:
            Score matrix (n_samples, n_classes)
        """
        n_samples = x.shape[0]
        scores = np.zeros((n_samples, self.n_classes))
        
        for class_idx, classifier in enumerate(self.classifiers):
            scores[:, class_idx] = classifier.predict_proba_raw(x)
        
        return scores


class OneVsOne:
    """
    One-vs-One (OvO) multi-class classification.
    Trains one binary classifier for each pair of classes.
    """
    
    def __init__(self, n_features: int, n_classes: int, C: float = 1.0):
        """
        Initialize OvO classifier.
        
        Args:
            n_features: Number of features
            n_classes: Number of classes
            C: SVM regularization parameter
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.C = C
        self.classifiers: Dict[Tuple[int, int], CustomSVM] = {}
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: List[str] = []
    
    def setup_classes(self, class_names: List[str]) -> None:
        """Setup class name mappings."""
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.index_to_class = class_names
    
    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[Tuple[int, int], List[float]]:
        """
        Train binary classifiers for all pairs of classes.
        
        Args:
            x: Input features
            y: Multi-class labels
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping (class_i, class_j) to loss history
        """
        self.classifiers = {}
        loss_histories = {}
        
        # Generate all pairs
        pairs = []
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                pairs.append((i, j))
        
        total_pairs = len(pairs)
        
        for pair_idx, (i, j) in enumerate(pairs):
            if verbose:
                print(f"\nTraining classifier {pair_idx + 1}/{total_pairs} ({self.index_to_class[i]} vs {self.index_to_class[j]})")
            
            # Filter samples for these two classes only
            mask = (y == i) | (y == j)
            x_pair = x[mask]
            y_pair = y[mask]
            
            if len(x_pair) == 0:
                continue
            
            # Create binary labels: +1 for class i, -1 for class j
            y_binary = np.where(y_pair == i, 1.0, -1.0)
            
            # Initialize classifier
            classifier = CustomSVM(self.n_features, C=self.C, hard_margin=False)
            trainer = SGD_Trainer(classifier, learning_rate=learning_rate, C=self.C)
            
            # Train
            loss_history = trainer.train(x_pair, y_binary, epochs=epochs, verbose=False)
            loss_histories[(i, j)] = loss_history
            
            self.classifiers[(i, j)] = classifier
            
            if verbose:
                class_i_samples = np.sum(y_pair == i)
                class_j_samples = np.sum(y_pair == j)
                print(f"  Samples: {class_i_samples} vs {class_j_samples}")
        
        return loss_histories
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class using OvO strategy with voting.
        
        Args:
            x: Input features
            
        Returns:
            Predicted class indices
        """
        n_samples = x.shape[0]
        votes = np.zeros((n_samples, self.n_classes))
        
        # Get votes from each pairwise classifier
        for (i, j), classifier in self.classifiers.items():
            predictions = classifier.predict(x)
            # +1 vote for class i if prediction is +1, else +1 vote for class j
            votes[:, i] += (predictions == 1).astype(float)
            votes[:, j] += (predictions == -1).astype(float)
        
        # Predict class with most votes
        predictions = np.argmax(votes, axis=1)
        return predictions

