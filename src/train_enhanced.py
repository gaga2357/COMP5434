"""
COMP5434 Dataset3 增强训练脚本 - 冲击88%+准确率
特征工程 + 优化网络架构 + 集成学习
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset3_solution import NeuralNetwork, DataPreprocessor
from dataset3_logistic import LogisticRegression


class FeatureEngineer:
    """高级特征工程"""
    
    def __init__(self):
        self.feature_names = []
    
    def create_features(self, df):
        """创建新特征"""
        df = df.copy()
        
        # 填充可能的缺失值
        df = df.fillna(0)
        
        # 1. 总住宿天数
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        
        # 2. 总人数
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        
        # 3. 周末比例
        df['weekend_ratio'] = df['no_of_weekend_nights'] / (df['total_nights'] + 1e-8)
        
        # 4. 人均价格
        df['price_per_guest'] = df['avg_price_per_room'] / (df['total_guests'] + 1e-8)
        
        # 5. 每晚平均价格
        df['price_per_night'] = df['avg_price_per_room'] / (df['total_nights'] + 1e-8)
        
        # 6. 提前预订程度（简化为数值分档）
        lead_time = df['lead_time'].values
        df['booking_urgency'] = np.where(lead_time <= 7, 0,
                                np.where(lead_time <= 30, 1,
                                np.where(lead_time <= 90, 2,
                                np.where(lead_time <= 365, 3, 4))))
        
        # 7. 是否带孩子
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        
        # 8. 是否要求特殊服务
        df['has_special_request'] = (df['no_of_special_requests'] > 0).astype(int)
        
        # 9. 取消历史比率
        total_prev = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
        df['cancellation_rate'] = df['no_of_previous_cancellations'] / (total_prev + 1e-8)
        
        # 10. 是否高价
        df['is_high_price'] = (df['avg_price_per_room'] > df['avg_price_per_room'].median()).astype(int)
        
        # 11. 长期住宿
        df['is_long_stay'] = (df['total_nights'] > 5).astype(int)
        
        return df


def load_and_engineer_features(train_path, test_path):
    """加载数据并进行特征工程"""
    print("=" * 60)
    print("加载数据并进行特征工程")
    print("=" * 60)
    
    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"原始训练集: {train_df.shape}")
    print(f"原始测试集: {test_df.shape}")
    
    # 保存ID和标签
    test_ids = test_df['id'].values
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])
    y_train = train_df['label'].values
    train_df = train_df.drop(columns=['label'])
    
    # 合并进行一致处理
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # 特征工程
    engineer = FeatureEngineer()
    combined_df = engineer.create_features(combined_df)
    
    # 类别编码
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    print(f"\\n类别特征: {categorical_cols}")
    
    for col in categorical_cols:
        unique_values = combined_df[col].unique()
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        combined_df[col] = combined_df[col].map(value_to_int)
    
    # 分离训练集和测试集
    train_encoded = combined_df.iloc[:len(train_df)]
    test_encoded = combined_df.iloc[len(train_df):]
    
    X_train = train_encoded.values.astype(float)
    X_test = test_encoded.values.astype(float)
    
    print(f"\\n特征工程后:")
    print(f"  特征数: {X_train.shape[1]} (原始17个 -> {X_train.shape[1]}个)")
    print(f"  训练样本: {X_train.shape[0]}")
    print(f"  测试样本: {X_test.shape[0]}")
    
    # 类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\\n标签分布:")
    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count} ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, X_test, test_ids


class EnhancedNeuralNetwork(NeuralNetwork):
    """增强神经网络，支持Dropout和学习率衰减"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, dropout_rate=0.0, random_state=42):
        super().__init__(layer_sizes, learning_rate, random_state)
        self.dropout_rate = dropout_rate
        self.initial_lr = learning_rate
        
    def dropout(self, X, training=True):
        """Dropout正则化"""
        if not training or self.dropout_rate == 0:
            return X
        mask = np.random.rand(*X.shape) > self.dropout_rate
        return X * mask / (1 - self.dropout_rate)
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
            verbose=True, lr_decay=0.95):
        """训练模型，支持学习率衰减"""
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 学习率衰减
            if epoch > 0 and epoch % 10 == 0:
                self.learning_rate *= lr_decay
            
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播（带Dropout）
                activations = [X_batch]
                for i in range(len(self.weights)):
                    z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
                    if i < len(self.weights) - 1:
                        a = self.relu(z)
                        a = self.dropout(a, training=True)  # Dropout
                    else:
                        a = self.softmax(z)
                    activations.append(a)
                
                # 计算损失
                loss = self.compute_loss(y_batch, activations[-1])
                epoch_loss += loss
                
                # 反向传播
                deltas = [activations[-1] - y_batch]
                for i in range(len(self.weights) - 1, 0, -1):
                    delta = np.dot(deltas[0], self.weights[i].T)
                    delta = delta * self.relu_derivative(activations[i])
                    deltas.insert(0, delta)
                
                # 更新权重
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / batch_size
                    self.biases[i] -= self.learning_rate * np.mean(deltas[i], axis=0, keepdims=True)
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_acc = self.accuracy(y_val, val_pred)
                train_acc = self.accuracy(y, self.predict_proba(X))
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss/n_batches:.4f}, "
                          f"Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"LR: {self.learning_rate:.6f}")
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停于 epoch {epoch+1}")
                        break


def train_ensemble_models(X_train, y_train, X_val, y_val, X_test):
    """训练多个模型并集成"""
    print("\\n" + "=" * 60)
    print("训练集成模型（多个网络 + 逻辑回归）")
    print("=" * 60)
    
    models = []
    predictions = []
    
    # 配置1: 深层网络
    config1 = {
        'layer_sizes': [X_train.shape[1], 128, 64, 32, 16, 2],
        'learning_rate': 0.01,
        'dropout_rate': 0.2,
        'epochs': 200,
        'batch_size': 64
    }
    
    print("\\n模型1: 深层网络 [128, 64, 32, 16]")
    model1 = EnhancedNeuralNetwork(
        config1['layer_sizes'],
        config1['learning_rate'],
        config1['dropout_rate']
    )
    model1.fit(X_train, y_train, X_val, y_val, 
               config1['epochs'], config1['batch_size'])
    pred1 = model1.predict_proba(X_test)
    predictions.append(pred1)
    val_acc1 = model1.accuracy(y_val, model1.predict_proba(X_val))
    print(f"验证准确率: {val_acc1:.4f}")
    
    # 配置2: 宽网络
    config2 = {
        'layer_sizes': [X_train.shape[1], 256, 128, 2],
        'learning_rate': 0.005,
        'dropout_rate': 0.3,
        'epochs': 180,
        'batch_size': 32
    }
    
    print("\\n模型2: 宽网络 [256, 128]")
    model2 = EnhancedNeuralNetwork(
        config2['layer_sizes'],
        config2['learning_rate'],
        config2['dropout_rate'],
        random_state=123
    )
    model2.fit(X_train, y_train, X_val, y_val, 
               config2['epochs'], config2['batch_size'])
    pred2 = model2.predict_proba(X_test)
    predictions.append(pred2)
    val_acc2 = model2.accuracy(y_val, model2.predict_proba(X_val))
    print(f"验证准确率: {val_acc2:.4f}")
    
    # 配置3: 逻辑回归（作为baseline）
    print("\\n模型3: 逻辑回归")
    model3 = LogisticRegression(learning_rate=0.1, l2_lambda=0.01)
    model3.fit(X_train, y_train, X_val, y_val, epochs=1000, batch_size=256, verbose=False)
    pred3 = model3.predict_proba(X_test)
    predictions.append(pred3)
    val_acc3 = model3.accuracy(X_val, y_val)
    print(f"验证准确率: {val_acc3:.4f}")
    
    # 集成预测（加权平均）
    print("\\n" + "=" * 60)
    print("集成模型结果")
    print("=" * 60)
    
    weights = np.array([val_acc1, val_acc2, val_acc3])
    weights = weights / weights.sum()  # 归一化
    
    print(f"模型权重: {weights}")
    
    ensemble_pred = np.zeros_like(pred1)
    for i, (pred, w) in enumerate(zip(predictions, weights)):
        ensemble_pred += w * pred
    
    # 验证集上的集成准确率
    ensemble_val_pred = np.zeros((len(X_val), 2))
    for i, w in enumerate(weights):
        if i == 0:
            ensemble_val_pred += w * model1.predict_proba(X_val)
        elif i == 1:
            ensemble_val_pred += w * model2.predict_proba(X_val)
        else:
            ensemble_val_pred += w * model3.predict_proba(X_val)
    
    ensemble_val_acc = np.mean(np.argmax(ensemble_val_pred, axis=1) == np.argmax(y_val, axis=1))
    print(f"\\n集成模型验证准确率: {ensemble_val_acc:.4f}")
    
    return ensemble_pred, ensemble_val_acc


def main():
    """主函数"""
    
    # 配置
    CONFIG = {
        'train_path': 'data/train.csv',
        'test_path': 'data/test.csv',
        'output_dir': 'output',
        'validation_split': 0.2,
        'random_state': 42,
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. 特征工程
    X_train, y_train, X_test, test_ids = load_and_engineer_features(
        CONFIG['train_path'], CONFIG['test_path']
    )
    
    # 2. 数据预处理
    print("\\n" + "=" * 60)
    print("数据预处理")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.standardize(X_train, fit=True)
    X_test_scaled = preprocessor.standardize(X_test, fit=False)
    y_train_onehot = preprocessor.one_hot_encode(y_train, fit=True)
    
    # 划分验证集
    X_tr, X_val, y_tr, y_val = preprocessor.train_test_split(
        X_train_scaled, y_train_onehot,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state']
    )
    
    print(f"训练集: {X_tr.shape}")
    print(f"验证集: {X_val.shape}")
    
    # 3. 训练集成模型
    ensemble_pred, val_acc = train_ensemble_models(X_tr, y_tr, X_val, y_val, X_test_scaled)
    
    # 4. 生成提交文件
    final_predictions = np.argmax(ensemble_pred, axis=1)
    submission = pd.DataFrame({
        'id': test_ids,
        'label': final_predictions
    })
    
    output_path = os.path.join(CONFIG['output_dir'], 'submission_enhanced.csv')
    submission.to_csv(output_path, index=False)
    
    print("\\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"提交文件: {output_path}")
    print("=" * 60)
    
    # 同时保存为主提交文件
    import shutil
    main_output = os.path.join(CONFIG['output_dir'], 'submission.csv')
    shutil.copy(output_path, main_output)
    print(f"已复制到: {main_output}")


if __name__ == "__main__":
    main()
