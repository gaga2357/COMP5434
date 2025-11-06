# COMP5434 Dataset3 - Hotel Booking Cancellation Prediction

A machine learning project for predicting hotel booking cancellations, implemented **completely from scratch without sklearn**.

## Project Overview

This project implements two classification algorithms to predict whether a hotel booking will be canceled:
- **Neural Network (MLP)**: 83% validation accuracy
- **Logistic Regression**: 80% validation accuracy

**Key Constraint**: All algorithms implemented using only NumPy and Pandas, no scikit-learn allowed.

## Dataset

- **Training Set**: 25,417 records with 17 features
- **Test Set**: 10,858 records
- **Task**: Binary classification (0 = Not Canceled, 1 = Canceled)
- **Features**: Numerical (e.g., adults, children, lead time) and categorical (meal plan, room type, market segment)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python src/train_dataset3.py
```

The script will:
1. Load and preprocess the data
2. Train both neural network and logistic regression models
3. Evaluate on validation set
4. Generate predictions in `output/submission.csv`

### Output

- `output/submission.csv` - Best model predictions (ready for Kaggle submission)
- `output/submission_nn.csv` - Neural network predictions
- `output/submission_lr.csv` - Logistic regression predictions

## Model Architecture

### Neural Network

- **Architecture**: Multi-layer perceptron with 3 hidden layers [64, 32, 16]
- **Activation**: ReLU (hidden) + Softmax (output)
- **Loss**: Cross-entropy
- **Optimizer**: Mini-batch gradient descent
- **Weight Initialization**: He initialization

### Logistic Regression

- **Model**: Softmax regression
- **Regularization**: L2 penalty
- **Optimizer**: Batch gradient descent

## Implementation Details

All core algorithms are implemented from scratch in NumPy:

- Forward and backward propagation
- Gradient descent optimization
- Cross-entropy loss computation
- Data preprocessing (standardization, encoding)
- Train/validation split

**No sklearn components used.**

## Project Structure

```
├── src/
│   ├── train_dataset3.py          # Main training script
│   ├── dataset3_solution.py       # Neural network implementation
│   └── dataset3_logistic.py       # Logistic regression implementation
├── data/
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── output/
│   └── submission.csv             # Predictions for submission
├── tests/
│   └── test_solution.py           # Algorithm validation
└── docs/                          # Additional documentation
```

## Performance

| Model | Training Accuracy | Validation Accuracy |
|-------|------------------|---------------------|
| Neural Network | 85.3% | 83.1% |
| Logistic Regression | 81.5% | 80.2% |

## Configuration

Edit `CONFIG` in `src/train_dataset3.py` to adjust hyperparameters:

```python
CONFIG = {
    'hidden_layers': [64, 32, 16],
    'learning_rate': 0.01,
    'epochs': 150,
    'batch_size': 64,
    'validation_split': 0.2
}
```

## Testing

```bash
python tests/test_solution.py
```

## Requirements

- Python 3.x
- NumPy
- Pandas

## License

Academic project for COMP5434.
