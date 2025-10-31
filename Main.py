"""
Main execution script for Malware Classification Project.
Implements SVM from scratch with SGD optimization.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import argparse
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import MalwareDatasetLoader
from utils.feature_engineering import FeatureEngineer
from models.svm import CustomSVM, get_support_vectors
from models.sgd_optimizer import SGD_Trainer
from models.multiclass import OneVsAll, OneVsOne
from evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    plot_confusion_matrix
)


def load_and_process_data(
    feature_dir: str,
    labels_file: str = None,
    max_samples: int = None,
    benign_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FeatureEngineer, Dict[str, int], Dict[str, int]]:
    """
    Load and process the malware dataset.
    
    Args:
        feature_dir: Directory containing feature vector files
        labels_file: Optional path to labels CSV file
        max_samples: Maximum number of samples to load (for testing)
        benign_ratio: Ratio of benign samples to keep (for balanced dataset)
        
    Returns:
        Tuple of processed data and metadata
    """
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)
    
    # Load dataset
    loader = MalwareDatasetLoader(feature_dir, labels_file)
    samples, family_counts = loader.scan_dataset()
    
    # Get family statistics
    family_stats = loader.get_family_statistics()
    print(f"\nFamily distribution (top 20):")
    sorted_families = sorted(family_stats.items(), key=lambda x: x[1], reverse=True)
    for fam, count in sorted_families[:20]:
        print(f"  {fam}: {count}")
    
    # Limit samples if requested
    if max_samples and len(samples) > max_samples:
        print(f"\nLimiting to {max_samples} samples")
        np.random.seed(42)
        indices = np.random.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in indices]
    
    # Load features for all samples
    print("\nLoading features from files...")
    all_features_list = []
    labels_list = []
    families_list = []
    
    for filepath, label, family in samples:
        features = loader.load_features_from_file(filepath)
        all_features_list.append(set(features))
        labels_list.append(label)
        families_list.append(family)
    
    # Handle benign/malware balance for binary classification
    if benign_ratio < 1.0:
        print(f"\nBalancing dataset (keeping {benign_ratio*100:.1f}% of benign samples)...")
        benign_indices = [i for i, label in enumerate(labels_list) if label.lower() == 'benign']
        malware_indices = [i for i, label in enumerate(labels_list) if label.lower() == 'malware']
        
        if len(benign_indices) > 0:
            n_benign_keep = int(len(benign_indices) * benign_ratio)
            np.random.seed(42)
            benign_keep = np.random.choice(benign_indices, min(n_benign_keep, len(benign_indices)), replace=False)
            
            keep_indices = list(benign_keep) + malware_indices
            all_features_list = [all_features_list[i] for i in keep_indices]
            labels_list = [labels_list[i] for i in keep_indices]
            families_list = [families_list[i] for i in keep_indices]
            
            print(f"  Kept {len(benign_keep)} benign, {len(malware_indices)} malware samples")
    
    # Build vocabulary and process
    print("\nBuilding vocabulary and feature vectors...")
    feature_engineer = FeatureEngineer()
    feature_engineer.build_vocabulary(all_features_list)
    
    # Process samples
    samples_data = list(zip(all_features_list, labels_list, families_list))
    X_binary, y_binary, X_multiclass, y_multiclass, family_to_index = feature_engineer.process_samples(samples_data)
    
    # Create index to family mapping
    index_to_family = {idx: fam for fam, idx in family_to_index.items()}
    
    print(f"\nFinal dataset shape: {X_binary.shape}")
    print(f"Binary labels: {np.sum(y_binary == 1)} malware, {np.sum(y_binary == -1)} benign")
    
    return (X_binary, y_binary, X_multiclass, y_multiclass, 
            feature_engineer, family_to_index, index_to_family)


def train_binary_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    learning_rate: float = 0.01,
    epochs: int = 50
) -> Tuple[CustomSVM, List[float], Dict[str, float]]:
    """
    Train binary SVM classifier.
    
    Returns:
        Tuple of (trained_model, loss_history, validation_metrics)
    """
    print("\n" + "=" * 60)
    print("Training Binary Classifier (Malware vs Benign)")
    print("=" * 60)
    
    # Initialize model
    n_features = X_train.shape[1]
    model = CustomSVM(n_features, C=C, hard_margin=False)
    
    # Initialize trainer
    trainer = SGD_Trainer(model, learning_rate=learning_rate, C=C)
    
    # Train
    loss_history = trainer.train(X_train, y_train, epochs=epochs, verbose=True)
    
    # Evaluate
    y_pred = model.predict(X_val)
    metrics = compute_binary_metrics(y_val, y_pred)
    
    print("\nValidation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return model, loss_history, metrics


def train_multiclass_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    class_names: List[str],
    strategy: str = 'ova',
    C: float = 1.0,
    learning_rate: float = 0.01,
    epochs: int = 50
) -> Tuple[object, Dict]:
    """
    Train multi-class classifier using OvA or OvO.
    
    Returns:
        Tuple of (trained_model, validation_metrics)
    """
    print("\n" + "=" * 60)
    print(f"Training Multi-Class Classifier ({strategy.upper()})")
    print("=" * 60)
    
    n_features = X_train.shape[1]
    
    if strategy.lower() == 'ova':
        model = OneVsAll(n_features, n_classes, C=C)
        model.setup_classes(class_names)
        loss_histories = model.train(X_train, y_train, learning_rate, epochs, verbose=True)
    elif strategy.lower() == 'ovo':
        model = OneVsOne(n_features, n_classes, C=C)
        model.setup_classes(class_names)
        loss_histories = model.train(X_train, y_train, learning_rate, epochs, verbose=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Evaluate
    y_pred = model.predict(X_val)
    metrics = compute_multiclass_metrics(y_val, y_pred, class_names)
    
    print(f"\nValidation Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Number of classes: {metrics['n_classes']}")
    
    return model, metrics


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int
) -> Dict:
    """
    Perform grid search over hyperparameters.
    
    Returns:
        Dictionary with best parameters and results
    """
    print("\n" + "=" * 60)
    print("Hyperparameter Tuning")
    print("=" * 60)
    
    # Define search space
    C_values = [0.1, 1.0, 10.0, 100.0]
    lr_values = [0.001, 0.01, 0.1]
    epochs_values = [20, 50, 100]
    
    best_accuracy = 0.0
    best_params = {}
    results = []
    
    total_combinations = len(C_values) * len(lr_values) * len(epochs_values)
    current = 0
    
    for C in C_values:
        for lr in lr_values:
            for epochs in epochs_values:
                current += 1
                print(f"\n[{current}/{total_combinations}] Testing C={C}, lr={lr}, epochs={epochs}")
                
                # Train model
                model = CustomSVM(n_features, C=C, hard_margin=False)
                trainer = SGD_Trainer(model, learning_rate=lr, C=C)
                trainer.train(X_train, y_train, epochs=epochs, verbose=False)
                
                # Evaluate
                y_pred = model.predict(X_val)
                metrics = compute_binary_metrics(y_val, y_pred)
                accuracy = metrics['accuracy']
                
                results.append({
                    'C': C,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'accuracy': accuracy,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                })
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'C': C, 'learning_rate': lr, 'epochs': epochs}
                
                print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return {
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'results': results
    }


def analyze_feature_importance(
    model: CustomSVM,
    feature_engineer: FeatureEngineer,
    top_k: int = 10
) -> None:
    """
    Analyze and display feature importance.
    """
    print("\n" + "=" * 60)
    print("Feature Importance Analysis")
    print("=" * 60)
    
    # Get weights
    weights = model.w.detach().numpy()
    
    # Top features
    top_features = feature_engineer.get_top_k_features(weights, k=top_k, top=True)
    bottom_features = feature_engineer.get_top_k_features(weights, k=top_k, top=False)
    
    print(f"\nTop {top_k} Most Influential Features:")
    for i, (feat, weight) in enumerate(top_features, 1):
        print(f"  {i}. {feat[:80]}... (weight: {weight:.6f})")
    
    print(f"\nBottom {top_k} Least Influential Features:")
    for i, (feat, weight) in enumerate(bottom_features, 1):
        print(f"  {i}. {feat[:80]}... (weight: {weight:.6f})")
    
    print(f"\nWeight statistics:")
    print(f"  Mean: {np.mean(np.abs(weights)):.6f}")
    print(f"  Std: {np.std(np.abs(weights)):.6f}")
    print(f"  Max: {np.max(np.abs(weights)):.6f}")
    print(f"  Min: {np.min(np.abs(weights)):.6f}")


def compare_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    subset_size: int = 5000
) -> Dict:
    """
    Compare custom SVM with sklearn implementations.
    Uses small subset for sklearn comparison.
    """
    print("\n" + "=" * 60)
    print("Comparing with sklearn (using subset)")
    print("=" * 60)
    
    try:
        from sklearn.svm import LinearSVC, SVC
        from sklearn.preprocessing import StandardScaler
        
        # Use subset for sklearn
        indices = np.random.choice(len(X_train), min(subset_size, len(X_train)), replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = (y_train[indices] > 0).astype(int)
        
        indices_val = np.random.choice(len(X_val), min(subset_size, len(X_val)), replace=False)
        X_val_sub = X_val[indices_val]
        y_val_sub = (y_val[indices_val] > 0).astype(int)
        
        results = {}
        
        # Custom SVM
        print("\nTraining custom SVM...")
        n_features = X_train_sub.shape[1]
        custom_model = CustomSVM(n_features, C=1.0)
        trainer = SGD_Trainer(custom_model, learning_rate=0.01, C=1.0)
        trainer.train(X_train_sub, y_train_sub, epochs=50, verbose=False)
        y_pred_custom = custom_model.predict(X_val_sub)
        metrics_custom = compute_binary_metrics(y_val_sub, y_pred_custom)
        results['custom'] = metrics_custom
        print(f"  Custom SVM Accuracy: {metrics_custom['accuracy']:.4f}")
        
        # sklearn LinearSVC
        print("\nTraining sklearn LinearSVC...")
        linear_svc = LinearSVC(C=1.0, max_iter=1000, random_state=42)
        linear_svc.fit(X_train_sub, y_train_sub)
        y_pred_linear = linear_svc.predict(X_val_sub)
        metrics_linear = compute_binary_metrics(y_val_sub, y_pred_linear)
        results['sklearn_linear'] = metrics_linear
        print(f"  LinearSVC Accuracy: {metrics_linear['accuracy']:.4f}")
        
        # sklearn SVC with RBF kernel (on smaller subset)
        if len(X_train_sub) >= 2 and len(X_val_sub) >= 2:
            rbf_n = min(1000, len(X_train_sub), len(X_val_sub))
            if rbf_n >= 2:
                print("\nTraining sklearn SVC (RBF) on smaller subset...")
                indices_rbf = np.random.choice(len(X_train_sub), rbf_n, replace=False)
                X_train_rbf = X_train_sub[indices_rbf]
                y_train_rbf = y_train_sub[indices_rbf]
                
                indices_rbf_val = np.random.choice(len(X_val_sub), rbf_n, replace=False)
                X_val_rbf = X_val_sub[indices_rbf_val]
                y_val_rbf = y_val_sub[indices_rbf_val]
                
                # Ensure both classes exist in training subset for RBF
                if len(np.unique(y_train_rbf)) < 2:
                    results['sklearn_rbf'] = None
                    print("\n  Skipping RBF (training subset single-class)")
                else:
                    svc_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', max_iter=1000, random_state=42)
                    svc_rbf.fit(X_train_rbf, y_train_rbf)
                    y_pred_rbf = svc_rbf.predict(X_val_rbf)
                    metrics_rbf = compute_binary_metrics(y_val_rbf, y_pred_rbf)
                    results['sklearn_rbf'] = metrics_rbf
                    print(f"  SVC (RBF) Accuracy: {metrics_rbf['accuracy']:.4f}")
                    
                    print("\nKernel Comparison Note:")
                    print("  RBF kernel can help when classes are not linearly separable.")
                    print("  It maps data to a higher-dimensional feature space where separation is easier.")
            else:
                results['sklearn_rbf'] = None
                print("\n  Skipping RBF (subset too small)")
        else:
            results['sklearn_rbf'] = None
            print("\n  Skipping RBF (dataset too small)")
        
        return results
        
    except ImportError:
        print("  sklearn not available, skipping comparison")
        return {}


def main():
    parser = argparse.ArgumentParser(description='Malware Classification using Custom SVM')
    parser.add_argument('--feature-dir', type=str, 
                       default='feature_vectors',
                       help='Directory containing feature vector files')
    parser.add_argument('--labels-file', type=str, default=None,
                       help='Optional CSV file with labels (format: hash,label,family)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load (for testing)')
    parser.add_argument('--benign-ratio', type=float, default=0.1,
                       help='Ratio of benign samples to keep')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--C', type=float, default=1.0,
                       help='SVM regularization parameter')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for SGD')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--multiclass-strategy', type=str, default='ova',
                       choices=['ova', 'ovo'],
                       help='Multi-class strategy: ova (One-vs-All) or ovo (One-vs-One)')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning')
    parser.add_argument('--skip-sklearn', action='store_true',
                       help='Skip sklearn comparison')
    
    args = parser.parse_args()
    
    # Load and process data
    (X_binary, y_binary, X_multiclass, y_multiclass,
     feature_engineer, family_to_index, index_to_family) = load_and_process_data(
        args.feature_dir,
        args.labels_file,
        args.max_samples,
        args.benign_ratio
    )
    
    # Split data
    print("\n" + "=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    n_samples = X_binary.shape[0]
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - args.test_size))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train_bin = X_binary[train_indices]
    y_train_bin = y_binary[train_indices]
    X_val_bin = X_binary[val_indices]
    y_val_bin = y_binary[val_indices]
    
    X_train_multi = X_multiclass[train_indices]
    y_train_multi = y_multiclass[train_indices]
    X_val_multi = X_multiclass[val_indices]
    y_val_multi = y_multiclass[val_indices]
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"Validation set: {len(val_indices)} samples")
    
    # Check class balance; skip binary training if only one class is present
    unique_binary = np.unique(y_binary)
    binary_model = None
    binary_metrics = None
    loss_history = []
    if unique_binary.size < 2:
        print("\nWARNING: Only one class detected in binary labels. Skipping binary training and evaluation.")
        print("Provide a labels file via --labels-file to enable malware vs benign training.")
    else:
        # Train binary classifier
        binary_model, loss_history, binary_metrics = train_binary_classifier(
            X_train_bin, y_train_bin, X_val_bin, y_val_bin,
            C=args.C, learning_rate=args.learning_rate, epochs=args.epochs
        )
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('Training Loss (Binary Classifier)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss_binary.png', dpi=300, bbox_inches='tight')
        print("\nSaved training loss plot: training_loss_binary.png")
        
        # Feature importance
        analyze_feature_importance(binary_model, feature_engineer)
        
        # Support vectors analysis
        print("\n" + "=" * 60)
        print("Support Vectors Analysis")
        print("=" * 60)
        support_x, support_y = get_support_vectors(binary_model, X_train_bin, y_train_bin)
        print(f"Number of support vectors: {len(support_x)} / {len(X_train_bin)} ({100*len(support_x)/len(X_train_bin):.2f}%)")
        print("Support vectors are samples where |y| < 1 (margin violations or on margin)")
        print("These are the 'critical' samples that define the decision boundary.")
        print("Hinge loss emphasizes these samples because they contribute to the loss.")
    
    # Hyperparameter tuning
    if binary_model is not None and not args.skip_tuning:
        tuning_results = hyperparameter_tuning(
            X_train_bin, y_train_bin, X_val_bin, y_val_bin,
            X_train_bin.shape[1]
        )
        
        # Plot tuning results
        if tuning_results['results']:
            results_df = tuning_results['results']
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy vs C
            C_vals = [r['C'] for r in results_df]
            acc_vals = [r['accuracy'] for r in results_df]
            axes[0, 0].scatter(C_vals, acc_vals, alpha=0.6)
            axes[0, 0].set_xlabel('C')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Accuracy vs C')
            axes[0, 0].set_xscale('log')
            axes[0, 0].grid(True)
            
            # Accuracy vs Learning Rate
            lr_vals = [r['learning_rate'] for r in results_df]
            axes[0, 1].scatter(lr_vals, acc_vals, alpha=0.6)
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy vs Learning Rate')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig('hyperparameter_tuning.png', dpi=300, bbox_inches='tight')
            print("\nSaved hyperparameter tuning plot: hyperparameter_tuning.png")
    
    # Multi-class classification (excluding benign)
    print("\n" + "=" * 60)
    print("Multi-Class Classification (Malware Families Only)")
    print("=" * 60)
    
    # Filter out benign samples for multiclass
    benign_idx = family_to_index.get('benign', -1)
    if benign_idx != -1:
        malware_mask_train = y_train_multi != benign_idx
        malware_mask_val = y_val_multi != benign_idx
        
        # Only proceed if we have malware samples
        if np.any(malware_mask_train):
            X_train_malware = X_train_multi[malware_mask_train]
            y_train_malware = y_train_multi[malware_mask_train]
            X_val_malware = X_val_multi[malware_mask_val]
            y_val_malware = y_val_multi[malware_mask_val]
        else:
            print("No malware samples found (all are benign). Skipping multiclass classification.")
            X_train_malware = None
    else:
        # No benign class, use all samples
        X_train_malware = X_train_multi
        y_train_malware = y_train_multi
        X_val_malware = X_val_multi
        y_val_malware = y_val_multi
    
    if X_train_malware is not None and len(X_train_malware) > 0:
        
        # Remap labels to remove benign (use train classes only)
        unique_classes = np.unique(y_train_malware)
        class_remap = {old: new for new, old in enumerate(unique_classes)}
        y_train_malware = np.array([class_remap[c] for c in y_train_malware])
        # Filter validation to seen classes
        seen_mask_val = np.isin(y_val_malware, list(class_remap.keys()))
        if not np.any(seen_mask_val):
            print("No validation samples for seen classes. Skipping multiclass evaluation.")
            return
        X_val_malware = X_val_malware[seen_mask_val]
        y_val_malware = y_val_malware[seen_mask_val]
        y_val_malware = np.array([class_remap[c] for c in y_val_malware])
        
        class_names = [index_to_family.get(c, f'class_{c}') for c in unique_classes]
        n_classes_malware = len(unique_classes)
        
        print(f"Training on {len(X_train_malware)} malware samples, {n_classes_malware} families")
        
        multiclass_model, multiclass_metrics = train_multiclass_classifier(
            X_train_malware, y_train_malware,
            X_val_malware, y_val_malware,
            n_classes_malware, class_names,
            strategy=args.multiclass_strategy,
            C=args.C, learning_rate=args.learning_rate, epochs=args.epochs
        )
        
        # Plot confusion matrix
        if len(class_names) <= 20:
            plot_confusion_matrix(
                multiclass_metrics['confusion_matrix_normalized'],
                class_names,  # All classes
                save_path='confusion_matrix_multiclass.png',
                normalize=True,
                title='Multi-Class Confusion Matrix (Normalized)'
            )
            print("\nSaved confusion matrix: confusion_matrix_multiclass.png")
        else:
            # Show top 20 classes only by frequency
            print(f"\nShowing top 20 classes (out of {len(class_names)})")
            # Use validation frequencies to select top classes present
            class_counts_val = Counter(y_val_malware)
            sorted_class_indices = sorted(class_counts_val.keys(), 
                                          key=lambda i: class_counts_val[i], 
                                          reverse=True)[:20]
            top_class_names = [class_names[i] for i in sorted_class_indices]
            
            # Filter confusion matrix to top classes
            cm_filtered = multiclass_metrics['confusion_matrix_normalized'][np.ix_(sorted_class_indices, sorted_class_indices)]
            plot_confusion_matrix(
                cm_filtered,
                top_class_names,
                save_path='confusion_matrix_multiclass_top20.png',
                normalize=True,
                title='Multi-Class Confusion Matrix - Top 20 Classes (Normalized)'
            )
            print("\nSaved confusion matrix (top 20): confusion_matrix_multiclass_top20.png")
    
    # sklearn comparison
    if binary_model is not None and not args.skip_sklearn:
        sklearn_results = compare_with_sklearn(
            X_train_bin, y_train_bin, X_val_bin, y_val_bin
        )
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if binary_metrics is not None:
        print("Binary Classification:")
        print(f"  Accuracy: {binary_metrics['accuracy']:.4f}")
        print(f"  Precision: {binary_metrics['precision']:.4f}")
        print(f"  Recall: {binary_metrics['recall']:.4f}")
        print(f"  F1 Score: {binary_metrics['f1_score']:.4f}")
    else:
        print("Binary Classification: Skipped (single class detected)")
    
    if 'multiclass_metrics' in locals():
        print(f"\nMulti-Class Classification ({args.multiclass_strategy.upper()}):")
        print(f"  Accuracy: {multiclass_metrics['accuracy']:.4f}")
        print(f"  Number of classes: {multiclass_metrics['n_classes']}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
