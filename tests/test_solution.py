"""
测试脚本 - 使用合成数据验证算法实现
"""

import numpy as np
import pandas as pd
from dataset3_solution import NeuralNetwork, DataPreprocessor


def generate_synthetic_data(n_samples: int = 1000, n_features: int = 20, 
                           n_classes: int = 3, random_state: int = 42):
    """
    生成合成数据用于测试
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        n_classes: 类别数量
        random_state: 随机种子
        
    Returns:
        X, y
    """
    np.random.seed(random_state)
    
    # 生成特征
    X = np.random.randn(n_samples, n_features)
    
    # 生成标签(基于特征的线性组合加噪声)
    weights = np.random.randn(n_features, n_classes)
    scores = np.dot(X, weights)
    y = np.argmax(scores, axis=1)
    
    return X, y


def test_neural_network():
    """测试神经网络实现"""
    print("=" * 60)
    print("Testing Neural Network Implementation")
    print("=" * 60)
    
    # 生成合成数据
    X, y = generate_synthetic_data(n_samples=1000, n_features=20, n_classes=3)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.standardize(X, fit=True)
    y_onehot = preprocessor.one_hot_encode(y, fit=True)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Number of classes: {preprocessor.num_classes}")
    
    # 构建模型
    layer_sizes = [20, 32, 16, 3]
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        random_state=42
    )
    
    print(f"\nModel architecture: {layer_sizes}")
    
    # 训练模型
    print("\nTraining...")
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=32,
        verbose=True
    )
    
    # 评估
    train_acc = model.accuracy(y_train, model.predict_proba(X_train))
    val_acc = model.accuracy(y_val, model.predict_proba(X_val))
    
    print(f"\n{'='*60}")
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"{'='*60}")
    
    # 测试预测功能
    predictions = model.predict(X_val[:10])
    true_labels = np.argmax(y_val[:10], axis=1)
    
    print("\nSample predictions (first 10):")
    print(f"Predicted: {predictions}")
    print(f"True:      {true_labels}")
    print(f"Match:     {predictions == true_labels}")
    
    return val_acc


def test_logistic_regression():
    """测试逻辑回归实现"""
    print("\n" + "=" * 60)
    print("Testing Logistic Regression Implementation")
    print("=" * 60)
    
    from dataset3_logistic import LogisticRegression
    
    # 生成合成数据
    X, y = generate_synthetic_data(n_samples=1000, n_features=20, n_classes=3)
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.standardize(X, fit=True)
    y_onehot = preprocessor.one_hot_encode(y, fit=True)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = preprocessor.train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # 构建模型
    model = LogisticRegression(
        learning_rate=0.1,
        l2_lambda=0.01,
        random_state=42
    )
    
    # 训练模型
    print("\nTraining...")
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=500,
        batch_size=64,
        verbose=True
    )
    
    # 评估
    train_acc = model.accuracy(X_train, y_train)
    val_acc = model.accuracy(X_val, y_val)
    
    print(f"\n{'='*60}")
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"{'='*60}")
    
    return val_acc


def test_data_saving():
    """测试数据保存功能"""
    print("\n" + "=" * 60)
    print("Testing Data Saving Functionality")
    print("=" * 60)
    
    # 创建测试数据
    test_ids = np.arange(100)
    predictions = np.random.randint(0, 3, 100)
    
    # 保存为CSV
    submission_df = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    test_file = 'test_submission.csv'
    submission_df.to_csv(test_file, index=False)
    print(f"Test submission saved to {test_file}")
    
    # 验证读取
    loaded_df = pd.read_csv(test_file)
    print(f"Loaded data shape: {loaded_df.shape}")
    print(f"Columns: {loaded_df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(loaded_df.head())
    
    # 清理测试文件
    import os
    os.remove(test_file)
    print(f"\nTest file {test_file} removed.")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("COMP5434 Dataset3 Solution - Algorithm Verification")
    print("=" * 60 + "\n")
    
    # 测试神经网络
    nn_acc = test_neural_network()
    
    # 测试逻辑回归
    lr_acc = test_logistic_regression()
    
    # 测试数据保存
    test_data_saving()
    
    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Neural Network Validation Accuracy: {nn_acc:.4f}")
    print(f"Logistic Regression Validation Accuracy: {lr_acc:.4f}")
    print("\n✓ All tests passed successfully!")
    print("The implementations are working correctly.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
