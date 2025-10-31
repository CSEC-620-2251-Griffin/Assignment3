# Malware Classification Project

This project implements a malware classification system using Support Vector Machines (SVM) and Stochastic Gradient Descent (SGD) from scratch, without using sklearn's SVM or SGD implementations.

## Project Structure

```
Assignment3/
├── Main.py                 # Main execution script
├── README.md              # This file
├── feature_vectors/        # Directory containing feature vector files
├── utils/
│   ├── data_loader.py     # Dataset loading utilities
│   └── feature_engineering.py  # Feature vector construction
├── models/
│   ├── svm.py             # Custom SVM implementation
│   ├── sgd_optimizer.py   # SGD optimizer from scratch
│   └── multiclass.py      # One-vs-All and One-vs-One strategies
└── evaluation/
    └── metrics.py         # Evaluation metrics and visualization
```

## Installation

### Requirements

- Python 3.7+
- PyTorch (for gradient computation)
- numpy
- matplotlib
- scikit-learn (for comparison only, not for training)

### Setup

1. Install PyTorch:
```bash
pip install torch
```

2. Install other dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

Or install all at once:
```bash
pip install torch numpy matplotlib scikit-learn
```

## Dataset

The dataset consists of feature vector files in the `feature_vectors/` directory. Each file:
- Has a filename that is a hash (SHA256)
- Contains one feature per line
- Features are of various types: `api_call::`, `url::`, `permission::`, etc.

### Labels File (Optional)

If you have a labels file, it should be a CSV with format:
```
hash,label,family
abc123...,malware,family1
def456...,benign,benign
...
```

Or simplified format:
```
hash,family
abc123...,family1
def456...,benign
...
```

## Usage

### Basic Usage

Run with default parameters:
```bash
python Main.py --feature-dir feature_vectors
```

### Command Line Arguments

- `--feature-dir`: Directory containing feature vector files (default: `feature_vectors`)
- `--labels-file`: Optional path to labels CSV file (default: None)
- `--max-samples`: Maximum number of samples to load (default: None, loads all)
- `--benign-ratio`: Ratio of benign samples to keep (default: 0.1)
- `--test-size`: Test set size ratio (default: 0.2)
- `--C`: SVM regularization parameter (default: 1.0)
- `--learning-rate`: Learning rate for SGD (default: 0.01)
- `--epochs`: Number of training epochs (default: 50)
- `--multiclass-strategy`: Multi-class strategy: `ova` or `ovo` (default: `ova`)
- `--skip-tuning`: Skip hyperparameter tuning
- `--skip-sklearn`: Skip sklearn comparison

### Examples

Run with custom parameters:
```bash
python Main.py --feature-dir feature_vectors --C 10.0 --learning-rate 0.001 --epochs 100
```

Run with limited samples for testing:
```bash
python Main.py --feature-dir feature_vectors --max-samples 10000
```

Run with labels file:
```bash
python Main.py --feature-dir feature_vectors --labels-file labels.csv
```

Use One-vs-One strategy:
```bash
python Main.py --feature-dir feature_vectors --multiclass-strategy ovo
```

## Features

### Binary Classification
- Classifies samples as malware (+1) or benign (-1)
- Uses custom SVM with hinge loss
- Implements soft margin with regularization parameter C

### Multi-Class Classification
- Classifies malware into families
- Supports One-vs-All (OvA) and One-vs-One (OvO) strategies
- Evaluates on malware classes only (excludes benign)

### Model Components

1. **Custom SVM** (`models/svm.py`):
   - Weight vector w and bias b
   - Prediction: y = w · x - b
   - Hinge loss: max(0, 1 - t*y) where t is target

2. **SGD Optimizer** (`models/sgd_optimizer.py`):
   - Stochastic Gradient Descent from scratch
   - Uses PyTorch for gradient computation
   - Updates weights: w = w - lr * gradient

3. **Feature Engineering** (`utils/feature_engineering.py`):
   - Builds global vocabulary from all features
   - Converts samples to binary feature vectors (1 = present, 0 = absent)

### Evaluation

The script generates:
- Training loss plots
- Confusion matrices (for multi-class)
- Hyperparameter tuning results
- Feature importance analysis
- Support vector analysis

### Output Files

- `training_loss_binary.png`: Training loss over epochs
- `hyperparameter_tuning.png`: Hyperparameter search results
- `confusion_matrix_multiclass.png`: Multi-class confusion matrix (if applicable)

## SVM Theory Questions

### When do kernels help?
Kernels help when the data is not linearly separable in the original feature space. By mapping data to a higher-dimensional feature space (e.g., via RBF kernel), the data may become linearly separable. RBF kernel is particularly useful when classes have non-linear boundaries.

### Meaning of |y| < 1 (Support Vectors)
Support vectors are training samples where the prediction margin |w · x - b| < 1. These are samples that:
- Are misclassified (margin < 0)
- Are on the margin (margin = 1)
- Are within the margin (0 < margin < 1)

These samples define the decision boundary and are critical for the SVM.

### Why hinge loss emphasizes support vectors
Hinge loss L = max(0, 1 - t*y) only contributes when t*y < 1 (i.e., when the sample is on or within the margin). Samples correctly classified with large margins (t*y > 1) contribute 0 to the loss. This makes the SVM focus on the "hard" examples (support vectors) that are near the decision boundary.

## Implementation Notes

- **No sklearn SVM/SGD**: The implementation uses only PyTorch for gradient computation, not sklearn's SVM or SGD classes
- **From scratch**: All SVM logic (prediction, loss, optimization) is implemented manually
- **PyTorch for gradients**: PyTorch is used solely for automatic differentiation, not for its optimizers

## Troubleshooting

### Memory Issues
If you run out of memory:
- Use `--max-samples` to limit the dataset size
- Reduce `--benign-ratio` to use fewer benign samples
- Process features in batches

### Slow Training
- Reduce `--epochs` for faster training
- Use `--max-samples` to train on a subset
- Skip hyperparameter tuning with `--skip-tuning`

### No Labels File
If you don't have a labels file:
- The script will assume all samples are malware
- You can create a labels file with the format described above
- Or modify the code to infer labels from directory structure

## License

This project is for educational purposes.

