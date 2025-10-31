"""
Feature engineering: Build vocabulary and convert samples to binary feature vectors.
"""
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict


class FeatureEngineer:
    """Converts feature lists to binary feature vectors."""
    
    def __init__(self):
        self.vocabulary = []  # List of unique features (vocabulary)
        self.feature_to_index = {}  # Map feature -> index
        self.vocab_size = 0
    
    def build_vocabulary(self, all_features: List[Set[str]]) -> None:
        """
        Build global vocabulary from all features across all samples.
        
        Args:
            all_features: List of sets, each set contains features for one sample
        """
        # Collect all unique features
        unique_features = set()
        for feature_set in all_features:
            unique_features.update(feature_set)
        
        # Sort for reproducibility
        self.vocabulary = sorted(list(unique_features))
        self.vocab_size = len(self.vocabulary)
        self.feature_to_index = {feat: idx for idx, feat in enumerate(self.vocabulary)}
        
        print(f"Built vocabulary of size: {self.vocab_size}")
    
    def features_to_binary_vector(self, features: Set[str]) -> np.ndarray:
        """
        Convert a set of features to a binary feature vector.
        
        Args:
            features: Set of feature strings
            
        Returns:
            Binary numpy array of size vocab_size
        """
        vector = np.zeros(self.vocab_size, dtype=np.float32)
        for feature in features:
            if feature in self.feature_to_index:
                idx = self.feature_to_index[feature]
                vector[idx] = 1.0
        return vector
    
    def process_samples(
        self, 
        samples_data: List[Tuple[Set[str], str, str]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Process all samples into binary feature vectors and labels.
        
        Args:
            samples_data: List of (features_set, label, family) tuples
            
        Returns:
            Tuple of:
            - X_binary: Binary feature matrix (n_samples, vocab_size)
            - y_binary: Binary labels (+1 for malware, -1 for benign)
            - X_multiclass: Same as X_binary (for multiclass)
            - y_multiclass: Multi-class labels (family indices)
            - family_to_index: Map from family name to index
        """
        n_samples = len(samples_data)
        X_binary = np.zeros((n_samples, self.vocab_size), dtype=np.float32)
        y_binary = np.zeros(n_samples, dtype=np.float32)
        
        # Build family mapping
        unique_families = set()
        for _, _, family in samples_data:
            unique_families.add(family)
        
        family_list = sorted(list(unique_families))
        family_to_index = {fam: idx for idx, fam in enumerate(family_list)}
        
        y_multiclass = np.zeros(n_samples, dtype=np.int32)
        
        print(f"Processing {n_samples} samples...")
        
        for idx, (features_set, label, family) in enumerate(samples_data):
            # Convert to binary vector
            X_binary[idx] = self.features_to_binary_vector(features_set)
            
            # Binary label: +1 for malware, -1 for benign
            y_binary[idx] = 1.0 if label.lower() == 'malware' else -1.0
            
            # Multi-class label: family index
            y_multiclass[idx] = family_to_index[family]
        
        X_multiclass = X_binary.copy()
        
        print(f"Binary labels: {np.sum(y_binary == 1)} malware, {np.sum(y_binary == -1)} benign")
        print(f"Multi-class: {len(family_list)} classes")
        
        return X_binary, y_binary, X_multiclass, y_multiclass, family_to_index
    
    def get_top_k_features(self, weights: np.ndarray, k: int = 10, top: bool = True) -> List[Tuple[str, float]]:
        """
        Get top or bottom k features by absolute weight.
        
        Args:
            weights: Weight vector from trained model
            k: Number of features to return
            top: If True, return top k; if False, return bottom k
            
        Returns:
            List of (feature_name, weight) tuples
        """
        if len(weights) != self.vocab_size:
            raise ValueError(f"Weight vector size {len(weights)} doesn't match vocab size {self.vocab_size}")
        
        # Get absolute weights with indices
        abs_weights = np.abs(weights)
        indices = np.argsort(abs_weights)
        
        if top:
            # Top k: largest absolute weights
            top_indices = indices[-k:][::-1]
        else:
            # Bottom k: smallest absolute weights
            top_indices = indices[:k]
        
        result = [(self.vocabulary[idx], weights[idx]) for idx in top_indices]
        return result

