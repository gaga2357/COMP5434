"""
COMP5434 Dataset3 终极优化版 - 冲击88%准确率
超强特征工程 + 深度网络 + 高级集成策略
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset3_solution import NeuralNetwork, DataPreprocessor
from dataset3_logistic import LogisticRegression


class UltraFeatureEngineer:
    """终极特征工程 - 创建更多交互特征"""
    
    def create_features(self, df):
        """创建高级特征"""
        df = df.copy()
        df = df.fillna(0)
        
        # === 基础衍生特征 ===
        # 1. 住宿相关
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        df['weekend_ratio'] = df['no_of_weekend_nights'] / (df['total_nights'] + 1e-8)
        df['is_weekend_only'] = ((df['no_of_week_nights'] == 0) & (df['no_of_weekend_nights'] > 0)).astype(int)
        df['is_weekday_only'] = ((df['no_of_weekend_nights'] == 0) & (df['no_of_week_nights'] > 0)).astype(int)
        df['is_long_stay'] = (df['total_nights'] >= 7).astype(int)
        df['is_short_stay'] = (df['total_nights'] <= 2).astype(int)
        
        # 2. 人数相关
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        df['is_alone'] = (df['total_guests'] == 1).astype(int)
        df['is_group'] = (df['total_guests'] >= 4).astype(int)
        df['adult_child_ratio'] = df['no_of_adults'] / (df['no_of_children'] + 1e-8)
        
        # 3. 价格相关
        df['price_per_guest'] = df['avg_price_per_room'] / (df['total_guests'] + 1e-8)
        df['price_per_night'] = df['avg_price_per_room'] / (df['total_nights'] + 1e-8)
        df['total_price'] = df['avg_price_per_room'] * df['total_nights']
        price_median = df['avg_price_per_room'].median()
        price_75 = df['avg_price_per_room'].quantile(0.75)
        df['is_high_price'] = (df['avg_price_per_room'] > price_75).astype(int)
        df['is_low_price'] = (df['avg_price_per_room'] < price_median).astype(int)
        df['price_level'] = pd.qcut(df['avg_price_per_room'], q=5, labels=False, duplicates='drop')
        
        # 4. 预订时间相关
        df['booking_urgency'] = pd.cut(df['lead_time'], 
                                       bins=[-1, 7, 30, 90, 180, 1000], 
                                       labels=[0, 1, 2, 3, 4]).astype(int)
        df['is_last_minute'] = (df['lead_time'] <= 7).astype(int)
        df['is_far_advance'] = (df['lead_time'] > 180).astype(int)
        df['lead_time_log'] = np.log1p(df['lead_time'])
        
        # 5. 特殊需求相关
        df['has_special_request'] = (df['no_of_special_requests'] > 0).astype(int)
        df['many_special_requests'] = (df['no_of_special_requests'] >= 3).astype(int)
        df['needs_parking'] = df['required_car_parking_space']
        
        # 6. 历史行为
        df['is_repeated'] = df['repeated_guest']
        total_prev = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
        df['has_history'] = (total_prev > 0).astype(int)
        df['cancellation_rate'] = df['no_of_previous_cancellations'] / (total_prev + 1e-8)
        df['loyalty_score'] = df['no_of_previous_bookings_not_canceled'] / (total_prev + 1)
        df['cancel_tendency'] = (df['no_of_previous_cancellations'] > 0).astype(int)
        
        # === 交互特征（关键！） ===
        # 7. 价格 x 时间交互
        df['price_lead_interaction'] = df['avg_price_per_room'] * df['lead_time_log']
        df['price_nights_interaction'] = df['avg_price_per_room'] * df['total_nights']
        df['urgency_price_high'] = df['is_last_minute'] * df['is_high_price']
        
        # 8. 人数 x 价格交互
        df['guests_price_interaction'] = df['total_guests'] * df['avg_price_per_room']
        df['alone_high_price'] = df['is_alone'] * df['is_high_price']
        df['group_low_price'] = df['is_group'] * df['is_low_price']
        
        # 9. 历史 x 当前行为
        df['repeater_cancel_tendency'] = df['is_repeated'] * df['cancel_tendency']
        df['new_high_price'] = (1 - df['is_repeated']) * df['is_high_price']
        df['loyalty_special_req'] = df['loyalty_score'] * df['no_of_special_requests']
        
        # 10. 时间相关交互
        df['weekend_high_price'] = (df['weekend_ratio'] > 0.5).astype(int) * df['is_high_price']
        df['long_stay_low_price'] = df['is_long_stay'] * df['is_low_price']
        
        # 11. 到达时间特征
        df['arrival_month_sin'] = np.sin(2 * np.pi * df['arrival_month'] / 12)
        df['arrival_month_cos'] = np.cos(2 * np.pi * df['arrival_month'] / 12)
        df['is_peak_season'] = df['arrival_month'].isin([7, 8, 12, 1]).astype(int)
        df['is_weekend_arrival'] = (df['arrival_date'] % 7 >= 5).astype(int)
        
        # 12. 复杂交互
        df['children_weekend'] = df['has_children'] * (df['weekend_ratio'] > 0)
        df['parking_needs_kids'] = df['needs_parking'] * df['has_children']
        df['special_req_per_guest'] = df['no_of_special_requests'] / (df['total_guests'] + 1e-8)
        
        return df


class UltraEnhancedNN(NeuralNetwork):
    """超级增强神经网络 - 支持Batch Normalization和更强的正则化"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, dropout_rate=0.0, 
                 l2_lambda=0.0, random_state=42):
        super().__init__(layer_sizes, learning_rate, random_state)
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.initial_lr = learning_rate
        
    def dropout(self, X, training=True):
        """Dropout正则化"""
        if not training or self.dropout_rate == 0:
            return X
        mask = np.random.rand(*X.shape) > self.dropout_rate
        return X * mask / (1 - self.dropout_rate)
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
            verbose=True, lr_decay=0.95, patience=30):
        """训练模型 - 增强版"""
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        best_val_acc = 0
        best_weights = None
        best_biases = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # 学习率衰减
            if epoch > 0 and epoch % 15 == 0:
                self.learning_rate *= lr_decay
            
            # 打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 前向传播
                activations = [X_batch]
                for j in range(len(self.weights)):
                    z = np.dot(activations[-1], self.weights[j]) + self.biases[j]
                    if j < len(self.weights) - 1:
                        a = self.relu(z)
                        a = self.dropout(a, training=True)
                    else:
                        a = self.softmax(z)
                    activations.append(a)
                
                # 损失（加L2正则化）
                loss = self.compute_loss(y_batch, activations[-1])
                if self.l2_lambda > 0:
                    l2_loss = sum(np.sum(w ** 2) for w in self.weights)
                    loss += 0.5 * self.l2_lambda * l2_loss
                epoch_loss += loss
                
                # 反向传播
                deltas = [activations[-1] - y_batch]
                for j in range(len(self.weights) - 1, 0, -1):
                    delta = np.dot(deltas[0], self.weights[j].T)
                    delta = delta * self.relu_derivative(activations[j])
                    deltas.insert(0, delta)
                
                # 更新权重（加L2正则化）
                for j in range(len(self.weights)):
                    grad = np.dot(activations[j].T, deltas[j]) / len(X_batch)
                    if self.l2_lambda > 0:
                        grad += self.l2_lambda * self.weights[j]
                    self.weights[j] -= self.learning_rate * grad
                    self.biases[j] -= self.learning_rate * np.mean(deltas[j], axis=0, keepdims=True)
            
            # 验证集评估
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_acc = self.accuracy(y_val, val_pred)
                train_acc = self.accuracy(y, self.predict_proba(X))
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss/n_batches:.4f}, "
                          f"Train Acc: {train_acc:.4f}, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"LR: {self.learning_rate:.6f}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停于 epoch {epoch+1}, 最佳验证准确率: {best_val_acc:.4f}")
                        break
        
        # 恢复最佳权重
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
            print(f"已恢复最佳模型权重 (验证准确率: {best_val_acc:.4f})")


def train_ultra_ensemble(X_train, y_train, X_val, y_val, X_test):
    """训练超强集成模型"""
    print("\n" + "=" * 60)
    print("训练终极集成模型（5个强力模型）")
    print("=" * 60)
    
    models_info = []
    predictions = []
    val_accs = []
    
    # 模型1: 超深网络
    print("\n模型1: 超深网络 [256, 128, 64, 32]")
    model1 = UltraEnhancedNN(
        [X_train.shape[1], 256, 128, 64, 32, 2],
        learning_rate=0.008,
        dropout_rate=0.25,
        l2_lambda=0.0001,
        random_state=42
    )
    model1.fit(X_train, y_train, X_val, y_val, epochs=250, batch_size=64)
    pred1 = model1.predict_proba(X_test)
    val_acc1 = model1.accuracy(y_val, model1.predict_proba(X_val))
    predictions.append(pred1)
    val_accs.append(val_acc1)
    print(f"验证准确率: {val_acc1:.4f}")
    
    # 模型2: 超宽网络
    print("\n模型2: 超宽网络 [384, 192]")
    model2 = UltraEnhancedNN(
        [X_train.shape[1], 384, 192, 2],
        learning_rate=0.005,
        dropout_rate=0.35,
        l2_lambda=0.0001,
        random_state=123
    )
    model2.fit(X_train, y_train, X_val, y_val, epochs=220, batch_size=32)
    pred2 = model2.predict_proba(X_test)
    val_acc2 = model2.accuracy(y_val, model2.predict_proba(X_val))
    predictions.append(pred2)
    val_accs.append(val_acc2)
    print(f"验证准确率: {val_acc2:.4f}")
    
    # 模型3: 中等深度高dropout
    print("\n模型3: 平衡网络 [192, 96, 48]")
    model3 = UltraEnhancedNN(
        [X_train.shape[1], 192, 96, 48, 2],
        learning_rate=0.01,
        dropout_rate=0.3,
        l2_lambda=0.0002,
        random_state=456
    )
    model3.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=48)
    pred3 = model3.predict_proba(X_test)
    val_acc3 = model3.accuracy(y_val, model3.predict_proba(X_val))
    predictions.append(pred3)
    val_accs.append(val_acc3)
    print(f"验证准确率: {val_acc3:.4f}")
    
    # 模型4: 深窄网络
    print("\n模型4: 深窄网络 [128, 64, 32, 16, 8]")
    model4 = UltraEnhancedNN(
        [X_train.shape[1], 128, 64, 32, 16, 8, 2],
        learning_rate=0.01,
        dropout_rate=0.2,
        l2_lambda=0.0001,
        random_state=789
    )
    model4.fit(X_train, y_train, X_val, y_val, epochs=230, batch_size=64)
    pred4 = model4.predict_proba(X_test)
    val_acc4 = model4.accuracy(y_val, model4.predict_proba(X_val))
    predictions.append(pred4)
    val_accs.append(val_acc4)
    print(f"验证准确率: {val_acc4:.4f}")
    
    # 模型5: 逻辑回归（多样性）
    print("\n模型5: 强正则化逻辑回归")
    model5 = LogisticRegression(learning_rate=0.1, l2_lambda=0.05)
    model5.fit(X_train, y_train, X_val, y_val, epochs=1200, batch_size=256, verbose=False)
    pred5 = model5.predict_proba(X_test)
    val_acc5 = model5.accuracy(X_val, y_val)
    predictions.append(pred5)
    val_accs.append(val_acc5)
    print(f"验证准确率: {val_acc5:.4f}")
    
    # 集成策略：加权平均（基于验证准确率）
    print("\n" + "=" * 60)
    print("集成模型结果")
    print("=" * 60)
    
    val_accs = np.array(val_accs)
    weights = val_accs ** 2  # 平方加权，更偏向高准确率模型
    weights = weights / weights.sum()
    
    print(f"验证准确率: {val_accs}")
    print(f"模型权重: {weights}")
    
    ensemble_pred = np.zeros_like(pred1)
    for pred, w in zip(predictions, weights):
        ensemble_pred += w * pred
    
    # 计算集成验证准确率
    ensemble_val_pred = np.zeros((len(X_val), 2))
    all_models = [model1, model2, model3, model4, model5]
    for i, (model, w) in enumerate(zip(all_models, weights)):
        if i < 4:
            ensemble_val_pred += w * model.predict_proba(X_val)
        else:
            ensemble_val_pred += w * model.predict_proba(X_val)
    
    ensemble_val_acc = np.mean(np.argmax(ensemble_val_pred, axis=1) == np.argmax(y_val, axis=1))
    print(f"\n🎯 集成模型验证准确率: {ensemble_val_acc:.4f} ({ensemble_val_acc*100:.2f}%)")
    
    # Stacking策略：如果集成不如最佳单模型，使用最佳单模型
    best_single_idx = np.argmax(val_accs)
    best_single_acc = val_accs[best_single_idx]
    
    if best_single_acc > ensemble_val_acc:
        print(f"\n⚠️  最佳单模型({best_single_idx+1})准确率更高: {best_single_acc:.4f}")
        print("使用最佳单模型预测")
        final_pred = predictions[best_single_idx]
        final_acc = best_single_acc
    else:
        print("\n✅ 集成模型效果最佳")
        final_pred = ensemble_pred
        final_acc = ensemble_val_acc
    
    return final_pred, final_acc


def main():
    """主函数"""
    
    CONFIG = {
        'train_path': 'data/train.csv',
        'test_path': 'data/test.csv',
        'output_dir': 'output',
        'validation_split': 0.2,
        'random_state': 42,
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 1. 加载数据
    print("=" * 60)
    print("🚀 终极优化版 - 目标88%准确率")
    print("=" * 60)
    
    train_df = pd.read_csv(CONFIG['train_path'])
    test_df = pd.read_csv(CONFIG['test_path'])
    
    test_ids = test_df['id'].values
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])
    y_train = train_df['label'].values
    train_df = train_df.drop(columns=['label'])
    
    # 2. 超强特征工程
    print("\n" + "=" * 60)
    print("特征工程（创建50+特征）")
    print("=" * 60)
    
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    engineer = UltraFeatureEngineer()
    combined_df = engineer.create_features(combined_df)
    
    # 类别编码
    categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        unique_values = combined_df[col].unique()
        value_to_int = {val: idx for idx, val in enumerate(unique_values)}
        combined_df[col] = combined_df[col].map(value_to_int)
    
    train_encoded = combined_df.iloc[:len(train_df)]
    test_encoded = combined_df.iloc[len(train_df):]
    
    X_train = train_encoded.values.astype(float)
    X_test = test_encoded.values.astype(float)
    
    print(f"特征数: 17 → {X_train.shape[1]}")
    print(f"训练样本: {X_train.shape[0]}")
    
    # 3. 数据预处理
    print("\n" + "=" * 60)
    print("数据标准化")
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
    
    # 4. 训练终极集成模型
    final_pred, final_acc = train_ultra_ensemble(X_tr, y_tr, X_val, y_val, X_test_scaled)
    
    # 5. 生成提交文件
    final_predictions = np.argmax(final_pred, axis=1)
    submission = pd.DataFrame({
        'id': test_ids,
        'label': final_predictions
    })
    
    output_path = os.path.join(CONFIG['output_dir'], 'submission_ultra.csv')
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)
    print(f"最终验证准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"提交文件: {output_path}")
    
    # 复制为主文件
    import shutil
    main_output = os.path.join(CONFIG['output_dir'], 'submission.csv')
    shutil.copy(output_path, main_output)
    print(f"已复制到: {main_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
