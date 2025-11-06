"""
COMP5434 Dataset3 训练脚本 - 酒店预订取消预测
使用神经网络和逻辑回归两种方案
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset3_solution import NeuralNetwork, DataPreprocessor
from dataset3_logistic import LogisticRegression


def load_hotel_data(train_path: str, test_path: str):
    """
    加载酒店预订数据并进行特征编码
    
    Args:
        train_path: 训练集路径
        test_path: 测试集路径
        
    Returns:
        X_train, y_train, X_test, test_ids
    """
    print("=" * 60)
    print("Loading Hotel Booking Cancellation Dataset")
    print("=" * 60)
    
    # 加载训练数据
    train_df = pd.read_csv(train_path)
    print(f"Train data shape: {train_df.shape}")
    
    # 加载测试数据
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # 提取ID
    test_ids = test_df['id'].values
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])
    
    # 分离标签
    y_train = train_df['label'].values
    train_df = train_df.drop(columns=['label'])
    
    # 识别类别特征
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical features: {categorical_cols}")
    
    # 合并训练集和测试集进行编码(保证编码一致)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 对类别特征进行Label Encoding
    for col in categorical_cols:
        # 创建映射字典
        unique_values = combined_df[col].unique()
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        combined_df[col] = combined_df[col].map(value_to_int)
    
    # 分离回训练集和测试集
    train_encoded = combined_df.iloc[:len(train_df)]
    test_encoded = combined_df.iloc[len(train_df):]
    
    X_train = train_encoded.values.astype(float)
    X_test = test_encoded.values.astype(float)
    
    print(f"\nAfter encoding:")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"\nLabel distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, X_test, test_ids


def train_neural_network(X_train, y_train, X_val, y_val, config):
    """训练神经网络模型"""
    print("\n" + "=" * 60)
    print("Training Neural Network Model")
    print("=" * 60)
    
    input_size = X_train.shape[1]
    output_size = config['num_classes']
    layer_sizes = [input_size] + config['hidden_layers'] + [output_size]
    
    print(f"Architecture: {layer_sizes}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=config['learning_rate'],
        random_state=config['random_state']
    )
    
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        verbose=True
    )
    
    return model


def train_logistic_regression(X_train, y_train, X_val, y_val, config):
    """训练逻辑回归模型"""
    print("\n" + "=" * 60)
    print("Training Logistic Regression Model")
    print("=" * 60)
    
    print(f"Learning Rate: {config['lr_learning_rate']}")
    print(f"L2 Lambda: {config['l2_lambda']}")
    print(f"Epochs: {config['lr_epochs']}")
    
    model = LogisticRegression(
        learning_rate=config['lr_learning_rate'],
        l2_lambda=config['l2_lambda'],
        random_state=config['random_state']
    )
    
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config['lr_epochs'],
        batch_size=config['lr_batch_size'],
        verbose=True
    )
    
    return model


def main():
    """主函数"""
    
    # 配置参数
    CONFIG = {
        # 数据路径
        'train_path': 'data/train.csv',
        'test_path': 'data/test.csv',
        'output_dir': 'output',
        
        # 数据划分
        'validation_split': 0.2,
        'random_state': 42,
        
        # 神经网络配置
        'hidden_layers': [64, 32, 16],
        'learning_rate': 0.01,
        'epochs': 150,
        'batch_size': 64,
        
        # 逻辑回归配置
        'lr_learning_rate': 0.1,
        'l2_lambda': 0.01,
        'lr_epochs': 800,
        'lr_batch_size': 128,
        
        # 其他
        'num_classes': 2,
        'train_both': True  # 是否训练两种模型
    }
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Step 1: 加载数据
    X_train, y_train, X_test, test_ids = load_hotel_data(
        CONFIG['train_path'], 
        CONFIG['test_path']
    )
    
    # Step 2: 数据预处理
    print("\n" + "=" * 60)
    print("Preprocessing Data")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    
    # 标准化特征
    X_train_scaled = preprocessor.standardize(X_train, fit=True)
    X_test_scaled = preprocessor.standardize(X_test, fit=False)
    
    # One-hot编码标签
    y_train_onehot = preprocessor.one_hot_encode(y_train, fit=True)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = preprocessor.train_test_split(
        X_train_scaled, y_train_onehot,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state']
    )
    
    print(f"Training set: {X_tr.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Step 3: 训练神经网络
    nn_model = train_neural_network(X_tr, y_tr, X_val, y_val, CONFIG)
    
    # 评估神经网络
    nn_train_acc = nn_model.accuracy(y_tr, nn_model.predict_proba(X_tr))
    nn_val_acc = nn_model.accuracy(y_val, nn_model.predict_proba(X_val))
    
    print(f"\nNeural Network Results:")
    print(f"  Training Accuracy: {nn_train_acc:.4f}")
    print(f"  Validation Accuracy: {nn_val_acc:.4f}")
    
    # 神经网络预测
    nn_predictions = nn_model.predict(X_test_scaled)
    nn_submission = pd.DataFrame({
        'id': test_ids,
        'label': nn_predictions
    })
    nn_output_path = os.path.join(CONFIG['output_dir'], 'submission_nn.csv')
    nn_submission.to_csv(nn_output_path, index=False)
    print(f"Neural Network predictions saved to: {nn_output_path}")
    
    # Step 4: 训练逻辑回归(可选)
    if CONFIG['train_both']:
        lr_model = train_logistic_regression(X_tr, y_tr, X_val, y_val, CONFIG)
        
        # 评估逻辑回归
        lr_train_acc = lr_model.accuracy(X_tr, y_tr)
        lr_val_acc = lr_model.accuracy(X_val, y_val)
        
        print(f"\nLogistic Regression Results:")
        print(f"  Training Accuracy: {lr_train_acc:.4f}")
        print(f"  Validation Accuracy: {lr_val_acc:.4f}")
        
        # 逻辑回归预测
        lr_predictions = lr_model.predict(X_test_scaled)
        lr_submission = pd.DataFrame({
            'id': test_ids,
            'label': lr_predictions
        })
        lr_output_path = os.path.join(CONFIG['output_dir'], 'submission_lr.csv')
        lr_submission.to_csv(lr_output_path, index=False)
        print(f"Logistic Regression predictions saved to: {lr_output_path}")
        
        # 对比结果
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(f"{'Model':<25} {'Train Acc':<12} {'Val Acc':<12}")
        print("-" * 60)
        print(f"{'Neural Network':<25} {nn_train_acc:<12.4f} {nn_val_acc:<12.4f}")
        print(f"{'Logistic Regression':<25} {lr_train_acc:<12.4f} {lr_val_acc:<12.4f}")
        
        # 选择最佳模型
        if nn_val_acc >= lr_val_acc:
            best_model = "Neural Network"
            best_submission = nn_output_path
        else:
            best_model = "Logistic Regression"
            best_submission = lr_output_path
        
        print(f"\nBest Model: {best_model}")
        
        # 复制最佳结果
        import shutil
        final_output = os.path.join(CONFIG['output_dir'], 'submission.csv')
        shutil.copy(best_submission, final_output)
        print(f"Best submission copied to: {final_output}")
    else:
        # 只使用神经网络
        final_output = os.path.join(CONFIG['output_dir'], 'submission.csv')
        nn_submission.to_csv(final_output, index=False)
    
    # 最终总结
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final submission file: {final_output}")
    print(f"Ready for Kaggle submission!")
    print("=" * 60)


if __name__ == "__main__":
    main()
