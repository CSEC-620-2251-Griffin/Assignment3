"""
Quick test script to verify the implementation works on a small sample.
"""
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import MalwareDatasetLoader
from utils.feature_engineering import FeatureEngineer
from models.svm import CustomSVM
from models.sgd_optimizer import SGD_Trainer
from evaluation.metrics import compute_binary_metrics

def test_basic_functionality():
    """Test basic functionality on a small sample."""
    print("Testing basic functionality...")
    
    feature_dir = 'feature_vectors'
    
    if not os.path.exists(feature_dir):
        print(f"Error: {feature_dir} directory not found!")
        return False
    
    # Load a small subset
    print("\n1. Loading dataset...")
    loader = MalwareDatasetLoader(feature_dir, labels_file=None)
    samples, family_counts = loader.scan_dataset()
    
    # Limit to 100 samples for testing
    if len(samples) > 100:
        import random
        random.seed(42)
        samples = random.sample(samples, 100)
    
    print(f"   Loaded {len(samples)} samples")
    
    # Load features
    print("\n2. Loading features...")
    all_features_list = []
    labels_list = []
    
    for filepath, label, family in samples[:100]:
        features = loader.load_features_from_file(filepath)
        all_features_list.append(set(features))
        labels_list.append(label)
    
    # Build vocabulary
    print("\n3. Building vocabulary...")
    feature_engineer = FeatureEngineer()
    feature_engineer.build_vocabulary(all_features_list)
    print(f"   Vocabulary size: {feature_engineer.vocab_size}")
    
    # Process samples
    print("\n4. Processing samples...")
    samples_data = list(zip(all_features_list, labels_list, ['family1'] * len(labels_list)))
    X_binary, y_binary, _, _, _ = feature_engineer.process_samples(samples_data)
    print(f"   Feature matrix shape: {X_binary.shape}")
    
    # Split train/test
    n_samples = X_binary.shape[0]
    split_idx = int(n_samples * 0.8)
    
    X_train = X_binary[:split_idx]
    y_train = y_binary[:split_idx]
    X_test = X_binary[split_idx:]
    y_test = y_binary[split_idx:]
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    print("\n5. Training SVM...")
    n_features = X_train.shape[1]
    model = CustomSVM(n_features, C=1.0)
    trainer = SGD_Trainer(model, learning_rate=0.01, C=1.0)
    loss_history = trainer.train(X_train, y_train, epochs=10, verbose=True)
    print(f"   Final loss: {loss_history[-1]:.6f}")
    
    # Evaluate
    print("\n6. Evaluating...")
    y_pred = model.predict(X_test)
    metrics = compute_binary_metrics(y_test, y_pred)
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    print("\n✓ All tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

