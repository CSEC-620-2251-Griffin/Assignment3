"""
Dataset loader for malware feature vectors.
Loads feature files and handles label mapping.
"""
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import numpy as np


class MalwareDatasetLoader:
    """Loads and processes malware feature vectors from directory."""
    
    def __init__(self, feature_dir: str, labels_file: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            feature_dir: Directory containing feature vector files
            labels_file: Optional CSV file with format: hash,label,family
                        If None, will attempt to infer labels from directory structure
        """
        self.feature_dir = feature_dir
        # Auto-discover labels file if not provided
        if labels_file and os.path.exists(labels_file):
            self.labels_file = labels_file
        else:
            candidates = [
                os.path.join(feature_dir, 'labels.csv'),
                os.path.join(os.path.dirname(feature_dir), 'labels.csv'),
                os.path.join(os.getcwd(), 'labels.csv'),
                os.path.join(feature_dir, 'sha256_family.csv'),
                os.path.join(os.path.dirname(feature_dir), 'sha256_family.csv'),
                os.path.join(os.getcwd(), 'sha256_family.csv')
            ]
            self.labels_file = next((c for c in candidates if os.path.exists(c)), None)
        # Mode: if using sha256_family.csv, treat hashes in file as malware; others benign
        self.sha256_list_mode = (
            self.labels_file is not None and os.path.basename(self.labels_file).lower().startswith('sha256_family')
        )
        self.samples = []  # List of (filepath, label, family)
        
    def load_labels_from_file(self) -> Dict[str, Tuple[str, str]]:
        """
        Load labels from CSV file.
        Expected format: hash,label,family or hash,family (label inferred)
        
        Returns:
            Dict mapping hash to (label, family) where label is 'benign' or 'malware'
        """
        if not self.labels_file or not os.path.exists(self.labels_file):
            return {}
        
        labels = {}
        with open(self.labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                hash_val = parts[0].strip()
                if self.sha256_list_mode:
                    # Formats supported:
                    # hash
                    # hash,family
                    family = parts[1].strip() if len(parts) >= 2 else 'unknown'
                    labels[hash_val] = ('malware', family)
                else:
                    # labels.csv modes:
                    # hash,family  -> infer benign if family == 'benign' else malware
                    # hash,label,family
                    if len(parts) >= 2:
                        if len(parts) == 2:
                            family = parts[1].strip()
                            label = 'benign' if family.lower() == 'benign' else 'malware'
                        else:
                            label = parts[1].strip().lower()
                            family = parts[2].strip()
                        labels[hash_val] = (label, family)
        return labels

    def load_label_lists(self) -> Dict[str, Tuple[str, str]]:
        """
        Support simple hash lists:
        - benign_hashes.txt: one hash per line → (benign, benign)
        - malware_hashes.txt: one hash per line → (malware, unknown)
        Searches in feature_dir and its parent.
        """
        mapping: Dict[str, Tuple[str, str]] = {}
        roots = [self.feature_dir, os.path.dirname(self.feature_dir)]
        for root in roots:
            benign_path = os.path.join(root, 'benign_hashes.txt')
            malware_path = os.path.join(root, 'malware_hashes.txt')
            if os.path.exists(benign_path):
                with open(benign_path, 'r') as f:
                    for line in f:
                        h = line.strip()
                        if h and not h.startswith('#'):
                            mapping[h] = ('benign', 'benign')
            if os.path.exists(malware_path):
                with open(malware_path, 'r') as f:
                    for line in f:
                        h = line.strip()
                        if h and not h.startswith('#'):
                            mapping[h] = ('malware', 'unknown')
        return mapping
    
    def load_features_from_file(self, filepath: str) -> List[str]:
        """Load features from a single feature vector file."""
        features = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        features.append(line)
        return features
    
    def scan_dataset(self) -> Tuple[List[Tuple[str, str, str]], Dict[str, int]]:
        """
        Scan the feature directory and collect all samples.
        
        Returns:
            Tuple of (samples, family_counts) where:
            - samples: List of (filepath, label, family)
            - family_counts: Dict mapping family names to counts
        """
        samples = []
        family_counts = defaultdict(int)
        # Prefer CSV labels; fall back to hash lists
        labels = {}
        if self.labels_file:
            labels = self.load_labels_from_file()
        if not labels:
            labels = self.load_label_lists()
        
        if not os.path.exists(self.feature_dir):
            raise ValueError(f"Feature directory does not exist: {self.feature_dir}")
        
        # If no labels file provided, we'll create a placeholder system
        # In practice, labels should be provided separately
        file_count = 0
        for filename in os.listdir(self.feature_dir):
            filepath = os.path.join(self.feature_dir, filename)
            if os.path.isfile(filepath):
                file_count += 1
                hash_val = filename
                
                # Try to get label from file, otherwise infer
                if hash_val in labels:
                    label, family = labels[hash_val]
                else:
                    if self.sha256_list_mode:
                        # Not listed in sha256_family.csv => benign
                        label = 'benign'
                        family = 'benign'
                    else:
                        # Unknown without labels
                        label = 'unknown'
                        family = 'unknown'
                
                samples.append((filepath, label, family))
                family_counts[family] += 1
        
        self.samples = samples
        print(f"Loaded {len(samples)} samples")
        # Report label distribution
        label_counts = defaultdict(int)
        for _, label, _ in samples:
            label_counts[label] += 1
        print("Label distribution:")
        for k, v in sorted(label_counts.items(), key=lambda x: x[0]):
            print(f"  {k}: {v}")
        print(f"Found {len(family_counts)} unique families")
        
        return samples, dict(family_counts)
    
    def get_all_features(self) -> List[str]:
        """Extract all unique features from all samples."""
        all_features = set()
        for filepath, _, _ in self.samples:
            features = self.load_features_from_file(filepath)
            all_features.update(features)
        return sorted(list(all_features))
    
    def get_family_statistics(self) -> Dict[str, int]:
        """Get statistics about malware families."""
        family_counts = defaultdict(int)
        for _, _, family in self.samples:
            family_counts[family] += 1
        return dict(family_counts)

