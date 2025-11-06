"""
COMP5434 Dataset3 分类预测解决方案
不使用sklearn,完全基于numpy实现神经网络分类器
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class NeuralNetwork:
    """多层感知机神经网络实现"""
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01, 
                 random_state: int = 42):
        """
        初始化神经网络
        
        Args:
            layer_sizes: 每层的神经元数量列表,例如[input_size, 64, 32, output_size]
            learning_rate: 学习率
            random_state: 随机种子
        """
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for better convergence
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU导数"""
        return (x > 0).astype(float)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        前向传播
        
        Args:
            X: 输入数据
            
        Returns:
            activations: 每层的激活值
            z_values: 每层的加权和
        """
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < self.num_layers - 2:
                # Hidden layers use ReLU
                a = self.relu(z)
            else:
                # Output layer uses softmax for classification
                a = self.softmax(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def backward(self, X: np.ndarray, y: np.ndarray, 
                 activations: List[np.ndarray], z_values: List[np.ndarray]) -> None:
        """
        反向传播并更新权重
        
        Args:
            X: 输入数据
            y: 真实标签(one-hot编码)
            activations: 前向传播的激活值
            z_values: 前向传播的加权和
        """
        m = X.shape[0]
        
        # Compute gradients
        deltas = [None] * (self.num_layers - 1)
        
        # Output layer delta
        deltas[-1] = activations[-1] - y
        
        # Hidden layers delta
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(deltas[i + 1], self.weights[i + 1].T) * self.relu_derivative(z_values[i])
            deltas[i] = delta
        
        # Update weights and biases
        for i in range(self.num_layers - 1):
            dw = np.dot(activations[i].T, deltas[i]) / m
            db = np.sum(deltas[i], axis=0, keepdims=True) / m
            
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        Args:
            y_true: 真实标签(one-hot编码)
            y_pred: 预测概率
            
        Returns:
            损失值
        """
        m = y_true.shape[0]
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        loss = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return loss
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 32, verbose: bool = True) -> None:
        """
        训练神经网络
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                activations, z_values = self.forward(X_batch)
                self.backward(X_batch, y_batch, activations, z_values)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                train_activations, _ = self.forward(X_train)
                train_loss = self.compute_loss(y_train, train_activations[-1])
                train_acc = self.accuracy(y_train, train_activations[-1])
                
                log_msg = f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
                
                if X_val is not None and y_val is not None:
                    val_activations, _ = self.forward(X_val)
                    val_loss = self.compute_loss(y_val, val_activations[-1])
                    val_acc = self.accuracy(y_val, val_activations[-1])
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
                
                print(log_msg)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入数据
            
        Returns:
            预测概率
        """
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 输入数据
            
        Returns:
            预测类别
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算准确率
        
        Args:
            y_true: 真实标签(one-hot编码)
            y_pred: 预测概率或one-hot编码
            
        Returns:
            准确率
        """
        y_true_labels = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred
        return np.mean(y_true_labels == y_pred_labels)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.num_classes = None
    
    def standardize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        标准化特征
        
        Args:
            X: 输入数据
            fit: 是否计算均值和标准差
            
        Returns:
            标准化后的数据
        """
        if fit:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Avoid division by zero
            self.std[self.std == 0] = 1
        
        return (X - self.mean) / self.std
    
    def one_hot_encode(self, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        One-hot编码
        
        Args:
            y: 标签数组
            fit: 是否计算类别数量
            
        Returns:
            One-hot编码后的标签
        """
        if fit:
            self.num_classes = len(np.unique(y))
        
        one_hot = np.zeros((y.shape[0], self.num_classes))
        one_hot[np.arange(y.shape[0]), y.astype(int)] = 1
        return one_hot
    
    def train_test_split(self, X: np.ndarray, y: np.ndarray, 
                        test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        划分训练集和验证集
        
        Args:
            X: 特征数据
            y: 标签数据
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        np.random.seed(random_state)
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def load_data(train_path: str, test_path: str) -> Tuple:
    """
    加载数据
    
    Args:
        train_path: 训练集路径
        test_path: 测试集路径
        
    Returns:
        X_train, y_train, X_test, test_ids
    """
    print("Loading data...")
    
    # Load training data
    train_df = pd.read_csv(train_path)
    print(f"Train data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # 假设第一列是ID,最后一列是标签(训练集),其余是特征
    if 'id' in train_df.columns.str.lower():
        id_col = [col for col in train_df.columns if 'id' in col.lower()][0]
        train_df = train_df.drop(columns=[id_col])
    
    if 'id' in test_df.columns.str.lower():
        id_col = [col for col in test_df.columns if 'id' in col.lower()][0]
        test_ids = test_df[id_col].values
        test_df = test_df.drop(columns=[id_col])
    else:
        test_ids = np.arange(len(test_df))
    
    # 假设最后一列是标签
    if 'label' in train_df.columns.str.lower() or 'target' in train_df.columns.str.lower():
        label_col = [col for col in train_df.columns if 'label' in col.lower() or 'target' in col.lower()][0]
    else:
        label_col = train_df.columns[-1]
    
    X_train = train_df.drop(columns=[label_col]).values
    y_train = train_df[label_col].values
    X_test = test_df.values
    
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, test_ids


def main():
    """主函数"""
    
    # 配置参数
    TRAIN_PATH = 'train.csv'
    TEST_PATH = 'test.csv'
    OUTPUT_PATH = 'submission.csv'
    
    # 超参数
    HIDDEN_LAYERS = [128, 64, 32]  # 隐藏层结构
    LEARNING_RATE = 0.01
    EPOCHS = 200
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Step 1: Load data
    X_train, y_train, X_test, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    
    # Step 2: Preprocess data
    preprocessor = DataPreprocessor()
    
    # Standardize features
    X_train_scaled = preprocessor.standardize(X_train, fit=True)
    X_test_scaled = preprocessor.standardize(X_test, fit=False)
    
    # One-hot encode labels
    y_train_onehot = preprocessor.one_hot_encode(y_train, fit=True)
    
    # Split validation set
    X_tr, X_val, y_tr, y_val = preprocessor.train_test_split(
        X_train_scaled, y_train_onehot, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_STATE
    )
    
    print(f"\nTraining set: {X_tr.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Step 3: Build model
    input_size = X_train.shape[1]
    output_size = preprocessor.num_classes
    layer_sizes = [input_size] + HIDDEN_LAYERS + [output_size]
    
    print(f"\nNeural Network Architecture: {layer_sizes}")
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE
    )
    
    # Step 4: Train model
    print("\nTraining model...")
    model.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    # Step 5: Make predictions
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test_scaled)
    
    # Step 6: Save submission
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")
    
    # Final evaluation on validation set
    val_predictions = model.predict(X_val)
    val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
    print(f"Final validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
