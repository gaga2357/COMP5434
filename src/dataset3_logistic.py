"""
COMP5434 Dataset3 逻辑回归解决方案
不使用sklearn,完全基于numpy实现Softmax回归(多分类逻辑回归)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class LogisticRegression:
    """Softmax逻辑回归实现(支持多分类)"""
    
    def __init__(self, learning_rate: float = 0.01, l2_lambda: float = 0.0,
                 random_state: int = 42):
        """
        初始化逻辑回归模型
        
        Args:
            learning_rate: 学习率
            l2_lambda: L2正则化系数
            random_state: 随机种子
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None
    
    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax激活函数
        
        Args:
            z: 输入值
            
        Returns:
            Softmax输出
        """
        # 减去最大值防止数值溢出
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算交叉熵损失
        
        Args:
            X: 特征数据
            y: 标签(one-hot编码)
            
        Returns:
            损失值
        """
        m = X.shape[0]
        
        # 前向传播
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        
        # 交叉熵损失
        epsilon = 1e-8
        cross_entropy = -np.sum(y * np.log(y_pred + epsilon)) / m
        
        # L2正则化
        l2_reg = (self.l2_lambda / (2 * m)) * np.sum(self.weights ** 2)
        
        return cross_entropy + l2_reg
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 1000, batch_size: int = 128, verbose: bool = True) -> None:
        """
        训练逻辑回归模型
        
        Args:
            X: 训练集特征
            y: 训练集标签(one-hot编码)
            X_val: 验证集特征
            y_val: 验证集标签
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否打印训练信息
        """
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        
        # 初始化权重和偏置
        self.weights = np.random.randn(n_features, n_classes) * 0.01
        self.bias = np.zeros((1, n_classes))
        
        for epoch in range(epochs):
            # Mini-batch梯度下降
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                batch_size_actual = X_batch.shape[0]
                
                # 前向传播
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.softmax(z)
                
                # 计算梯度
                dz = y_pred - y_batch
                dw = np.dot(X_batch.T, dz) / batch_size_actual
                db = np.sum(dz, axis=0, keepdims=True) / batch_size_actual
                
                # L2正则化梯度
                dw += (self.l2_lambda / batch_size_actual) * self.weights
                
                # 更新参数
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # 打印训练信息
            if verbose and (epoch + 1) % 50 == 0:
                train_loss = self.compute_loss(X, y)
                train_acc = self.accuracy(X, y)
                
                log_msg = f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}"
                
                if X_val is not None and y_val is not None:
                    val_loss = self.compute_loss(X_val, y_val)
                    val_acc = self.accuracy(X_val, y_val)
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
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
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
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算准确率
        
        Args:
            X: 特征数据
            y: 标签(one-hot编码)
            
        Returns:
            准确率
        """
        predictions = self.predict(X)
        y_labels = np.argmax(y, axis=1)
        return np.mean(predictions == y_labels)


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
            # 避免除以零
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
    
    # 加载训练数据
    train_df = pd.read_csv(train_path)
    print(f"Train data shape: {train_df.shape}")
    
    # 加载测试数据
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # 处理ID列
    if 'id' in train_df.columns.str.lower():
        id_col = [col for col in train_df.columns if 'id' in col.lower()][0]
        train_df = train_df.drop(columns=[id_col])
    
    if 'id' in test_df.columns.str.lower():
        id_col = [col for col in test_df.columns if 'id' in col.lower()][0]
        test_ids = test_df[id_col].values
        test_df = test_df.drop(columns=[id_col])
    else:
        test_ids = np.arange(len(test_df))
    
    # 处理标签列
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
    LEARNING_RATE = 0.1
    L2_LAMBDA = 0.01  # L2正则化系数
    EPOCHS = 1000
    BATCH_SIZE = 128
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Step 1: 加载数据
    X_train, y_train, X_test, test_ids = load_data(TRAIN_PATH, TEST_PATH)
    
    # Step 2: 数据预处理
    preprocessor = DataPreprocessor()
    
    # 标准化特征
    X_train_scaled = preprocessor.standardize(X_train, fit=True)
    X_test_scaled = preprocessor.standardize(X_test, fit=False)
    
    # One-hot编码标签
    y_train_onehot = preprocessor.one_hot_encode(y_train, fit=True)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = preprocessor.train_test_split(
        X_train_scaled, y_train_onehot,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE
    )
    
    print(f"\nTraining set: {X_tr.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Step 3: 构建模型
    print(f"\nLogistic Regression Configuration:")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"L2 Lambda: {L2_LAMBDA}")
    print(f"Epochs: {EPOCHS}")
    
    model = LogisticRegression(
        learning_rate=LEARNING_RATE,
        l2_lambda=L2_LAMBDA,
        random_state=RANDOM_STATE
    )
    
    # Step 4: 训练模型
    print("\nTraining model...")
    model.fit(
        X_tr, y_tr,
        X_val=X_val, y_val=y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    # Step 5: 预测
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test_scaled)
    
    # Step 6: 保存提交文件
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    submission_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSubmission saved to {OUTPUT_PATH}")
    
    # 最终验证集评估
    val_predictions = model.predict(X_val)
    val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
    print(f"Final validation accuracy: {val_accuracy:.4f}")


if __name__ == "__main__":
    main()
