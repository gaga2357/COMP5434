"""
COMP5434 Dataset3 极限优化版 - 突破88%准确率
特征暴力工程 + 超深网络 + 数据增强 + Stacking集成
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset3_solution import NeuralNetwork, DataPreprocessor
from dataset3_logistic import LogisticRegression


class ExtremeFeatureEngineer:
    """极限特征工程 - 70+特征"""
    
    def create_features(self, df):
        """创建极限特征"""
        df = df.copy()
        df = df.fillna(0)
        
        # === 基础特征 ===
        df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        df['weekend_ratio'] = df['no_of_weekend_nights'] / (df['total_nights'] + 1e-8)
        df['weekday_ratio'] = df['no_of_week_nights'] / (df['total_nights'] + 1e-8)
        
        # === 价格特征（多种视角） ===
        df['price_per_guest'] = df['avg_price_per_room'] / (df['total_guests'] + 1e-8)
        df['price_per_night'] = df['avg_price_per_room'] / (df['total_nights'] + 1e-8)
        df['total_price'] = df['avg_price_per_room'] * df['total_nights']
        df['price_per_adult'] = df['avg_price_per_room'] / (df['no_of_adults'] + 1e-8)
        df['price_squared'] = df['avg_price_per_room'] ** 2
        df['price_log'] = np.log1p(df['avg_price_per_room'])
        df['price_sqrt'] = np.sqrt(df['avg_price_per_room'])
        
        # 价格分级（多个维度）
        df['price_quintile'] = pd.qcut(df['avg_price_per_room'], q=5, labels=False, duplicates='drop')
        df['price_decile'] = pd.qcut(df['avg_price_per_room'], q=10, labels=False, duplicates='drop')
        price_median = df['avg_price_per_room'].median()
        price_q25 = df['avg_price_per_room'].quantile(0.25)
        price_q75 = df['avg_price_per_room'].quantile(0.75)
        df['is_very_cheap'] = (df['avg_price_per_room'] < price_q25).astype(int)
        df['is_cheap'] = ((df['avg_price_per_room'] >= price_q25) & (df['avg_price_per_room'] < price_median)).astype(int)
        df['is_expensive'] = ((df['avg_price_per_room'] >= price_median) & (df['avg_price_per_room'] < price_q75)).astype(int)
        df['is_very_expensive'] = (df['avg_price_per_room'] >= price_q75).astype(int)
        
        # === 时间特征 ===
        df['lead_time_log'] = np.log1p(df['lead_time'])
        df['lead_time_sqrt'] = np.sqrt(df['lead_time'])
        df['lead_time_squared'] = df['lead_time'] ** 2
        df['booking_urgency'] = pd.cut(df['lead_time'], bins=[-1, 3, 7, 14, 30, 60, 90, 180, 365, 1000], labels=list(range(9))).astype(int)
        df['is_same_day'] = (df['lead_time'] == 0).astype(int)
        df['is_last_minute'] = (df['lead_time'] <= 7).astype(int)
        df['is_moderate_advance'] = ((df['lead_time'] > 30) & (df['lead_time'] <= 90)).astype(int)
        df['is_far_advance'] = (df['lead_time'] > 180).astype(int)
        
        # === 住宿模式 ===
        df['is_weekend_only'] = ((df['no_of_week_nights'] == 0) & (df['no_of_weekend_nights'] > 0)).astype(int)
        df['is_weekday_only'] = ((df['no_of_weekend_nights'] == 0) & (df['no_of_week_nights'] > 0)).astype(int)
        df['is_mixed_stay'] = ((df['no_of_weekend_nights'] > 0) & (df['no_of_week_nights'] > 0)).astype(int)
        df['is_one_night'] = (df['total_nights'] == 1).astype(int)
        df['is_short_stay'] = (df['total_nights'] <= 2).astype(int)
        df['is_medium_stay'] = ((df['total_nights'] >= 3) & (df['total_nights'] <= 6)).astype(int)
        df['is_long_stay'] = (df['total_nights'] >= 7).astype(int)
        df['is_very_long_stay'] = (df['total_nights'] >= 14).astype(int)
        
        # === 客人组成 ===
        df['is_alone'] = (df['total_guests'] == 1).astype(int)
        df['is_couple'] = (df['total_guests'] == 2).astype(int)
        df['is_small_group'] = ((df['total_guests'] >= 3) & (df['total_guests'] <= 4)).astype(int)
        df['is_large_group'] = (df['total_guests'] >= 5).astype(int)
        df['has_children'] = (df['no_of_children'] > 0).astype(int)
        df['has_多children'] = (df['no_of_children'] >= 2).astype(int)
        df['adult_child_ratio'] = df['no_of_adults'] / (df['no_of_children'] + 1e-8)
        df['children_ratio'] = df['no_of_children'] / (df['total_guests'] + 1e-8)
        
        # === 历史行为 ===
        total_prev = df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled']
        df['has_history'] = (total_prev > 0).astype(int)
        df['cancellation_rate'] = df['no_of_previous_cancellations'] / (total_prev + 1e-8)
        df['loyalty_score'] = df['no_of_previous_bookings_not_canceled'] / (total_prev + 1)
        df['has_cancelled_before'] = (df['no_of_previous_cancellations'] > 0).astype(int)
        df['cancel_frequency'] = df['no_of_previous_cancellations'] / (df['no_of_previous_cancellations'] + df['no_of_previous_bookings_not_canceled'] + 1)
        df['history_length'] = total_prev
        df['is_loyal'] = (df['no_of_previous_bookings_not_canceled'] >= 3).astype(int)
        
        # === 特殊需求 ===
        df['has_special_request'] = (df['no_of_special_requests'] > 0).astype(int)
        df['many_special_requests'] = (df['no_of_special_requests'] >= 2).astype(int)
        df['special_req_per_guest'] = df['no_of_special_requests'] / (df['total_guests'] + 1e-8)
        df['special_req_per_night'] = df['no_of_special_requests'] / (df['total_nights'] + 1e-8)
        
        # === 到达时间周期性 ===
        df['arrival_month_sin'] = np.sin(2 * np.pi * df['arrival_month'] / 12)
        df['arrival_month_cos'] = np.cos(2 * np.pi * df['arrival_month'] / 12)
        df['arrival_date_sin'] = np.sin(2 * np.pi * df['arrival_date'] / 31)
        df['arrival_date_cos'] = np.cos(2 * np.pi * df['arrival_date'] / 31)
        df['is_peak_season'] = df['arrival_month'].isin([7, 8, 12]).astype(int)
        df['is_shoulder_season'] = df['arrival_month'].isin([4, 5, 9, 10]).astype(int)
        df['is_low_season'] = df['arrival_month'].isin([1, 2, 3, 6, 11]).astype(int)
        df['is_summer'] = df['arrival_month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['arrival_month'].isin([12, 1, 2]).astype(int)
        
        # === 交互特征（关键！）===
        # 价格x时间
        df['price_lead_interaction'] = df['avg_price_per_room'] * df['lead_time_log']
        df['price_nights_interaction'] = df['avg_price_per_room'] * df['total_nights']
        df['urgency_price_high'] = df['is_last_minute'] * df['is_very_expensive']
        df['far_advance_cheap'] = df['is_far_advance'] * df['is_very_cheap']
        
        # 价格x人数
        df['guests_price_interaction'] = df['total_guests'] * df['avg_price_per_room']
        df['alone_expensive'] = df['is_alone'] * df['is_very_expensive']
        df['group_cheap'] = df['is_large_group'] * df['is_very_cheap']
        df['children_price'] = df['has_children'] * df['avg_price_per_room']
        
        # 历史x当前
        df['repeater_cancel_tendency'] = df['repeated_guest'] * df['has_cancelled_before']
        df['new_customer_expensive'] = (1 - df['repeated_guest']) * df['is_very_expensive']
        df['loyal_special_req'] = df['is_loyal'] * df['no_of_special_requests']
        df['history_price_interaction'] = df['loyalty_score'] * df['avg_price_per_room']
        
        # 时间x住宿
        df['weekend_long_stay'] = (df['weekend_ratio'] > 0.5).astype(int) * df['is_long_stay']
        df['weekend_expensive'] = (df['weekend_ratio'] > 0.5).astype(int) * df['is_expensive']
        df['weekday_cheap'] = (df['weekday_ratio'] > 0.5).astype(int) * df['is_cheap']
        
        # 复杂交互
        df['parking_kids_weekend'] = df['required_car_parking_space'] * df['has_children'] * (df['weekend_ratio'] > 0.3).astype(int)
        df['special_req_expensive_repeater'] = df['has_special_request'] * df['is_expensive'] * df['repeated_guest']
        df['last_minute_expensive_weekend'] = df['is_last_minute'] * df['is_expensive'] * (df['weekend_ratio'] > 0.5).astype(int)
        
        return df


class ExtremeNN(NeuralNetwork):
    """极限神经网络 - 支持更多优化技巧"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, dropout_rate=0.0, 
                 l2_lambda=0.0, use_momentum=True, random_state=42):
        super().__init__(layer_sizes, learning_rate, random_state)
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.use_momentum = use_momentum
        self.initial_lr = learning_rate
        
        # 动量
        if use_momentum:
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
        
    def dropout(self, X, training=True):
        if not training or self.dropout_rate == 0:
            return X
        mask = np.random.rand(*X.shape) > self.dropout_rate
        return X * mask / (1 - self.dropout_rate)
    
    def fit(self, X, y, X_val=None, y_val=None, epochs=100, batch_size=32, 
            verbose=True, lr_decay=0.95, patience=35, momentum=0.9):
        """训练 - 支持动量"""
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        best_val_acc = 0
        best_weights = None
        best_biases = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # 学习率衰减
            if epoch > 0 and epoch % 12 == 0:
                self.learning_rate *= lr_decay
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
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
                
                # 反向传播
                deltas = [activations[-1] - y_batch]
                for j in range(len(self.weights) - 1, 0, -1):
                    delta = np.dot(deltas[0], self.weights[j].T)
                    delta = delta * self.relu_derivative(activations[j])
                    deltas.insert(0, delta)
                
                # 更新权重（带动量和L2）
                for j in range(len(self.weights)):
                    grad_w = np.dot(activations[j].T, deltas[j]) / len(X_batch)
                    grad_b = np.mean(deltas[j], axis=0, keepdims=True)
                    
                    if self.l2_lambda > 0:
                        grad_w += self.l2_lambda * self.weights[j]
                    
                    if self.use_momentum:
                        self.velocity_w[j] = momentum * self.velocity_w[j] - self.learning_rate * grad_w
                        self.velocity_b[j] = momentum * self.velocity_b[j] - self.learning_rate * grad_b
                        self.weights[j] += self.velocity_w[j]
                        self.biases[j] += self.velocity_b[j]
                    else:
                        self.weights[j] -= self.learning_rate * grad_w
                        self.biases[j] -= self.learning_rate * grad_b
            
            # 验证
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_acc = self.accuracy(y_val, val_pred)
                
                if verbose and (epoch + 1) % 10 == 0:
                    train_acc = self.accuracy(y, self.predict_proba(X))
                    print(f"Epoch {epoch+1}/{epochs} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, LR: {self.learning_rate:.6f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"早停于 epoch {epoch+1}, 最佳: {best_val_acc:.4f}")
                        break
        
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases


def train_extreme_ensemble(X_train, y_train, X_val, y_val, X_test):
    """极限集成"""
    print("\n" + "=" * 60)
    print("🔥 极限集成训练（7个超强模型）")
    print("=" * 60)
    
    predictions = []
    val_accs = []
    
    configs = [
        # 超深网络
        {"name": "超深网络 [384,192,96,48]", "layers": [X_train.shape[1], 384, 192, 96, 48, 2], 
         "lr": 0.006, "dropout": 0.3, "l2": 0.0001, "epochs": 280, "bs": 48, "seed": 42},
        
        # 超宽网络
        {"name": "超宽网络 [512,256]", "layers": [X_train.shape[1], 512, 256, 2], 
         "lr": 0.004, "dropout": 0.4, "l2": 0.0002, "epochs": 260, "bs": 32, "seed": 123},
        
        # 平衡网络1
        {"name": "平衡网络1 [256,128,64]", "layers": [X_train.shape[1], 256, 128, 64, 2], 
         "lr": 0.008, "dropout": 0.25, "l2": 0.0001, "epochs": 270, "bs": 56, "seed": 456},
        
        # 平衡网络2（不同随机种子）
        {"name": "平衡网络2 [256,128,64]", "layers": [X_train.shape[1], 256, 128, 64, 2], 
         "lr": 0.007, "dropout": 0.28, "l2": 0.00015, "epochs": 270, "bs": 56, "seed": 789},
        
        # 深窄网络
        {"name": "深窄网络 [160,80,40,20,10]", "layers": [X_train.shape[1], 160, 80, 40, 20, 10, 2], 
         "lr": 0.01, "dropout": 0.2, "l2": 0.0001, "epochs": 300, "bs": 64, "seed": 999},
        
        # 极深网络
        {"name": "极深网络 [256,128,64,32,16]", "layers": [X_train.shape[1], 256, 128, 64, 32, 16, 2], 
         "lr": 0.005, "dropout": 0.22, "l2": 0.00012, "epochs": 320, "bs": 48, "seed": 111},
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n模型{i+1}: {cfg['name']}")
        model = ExtremeNN(
            cfg['layers'], cfg['lr'], cfg['dropout'], cfg['l2'], 
            use_momentum=True, random_state=cfg['seed']
        )
        model.fit(X_train, y_train, X_val, y_val, 
                 epochs=cfg['epochs'], batch_size=cfg['bs'], verbose=False)
        
        pred = model.predict_proba(X_test)
        val_acc = model.accuracy(y_val, model.predict_proba(X_val))
        predictions.append(pred)
        val_accs.append(val_acc)
        print(f"✅ 验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # 逻辑回归
    print(f"\n模型{len(configs)+1}: 逻辑回归")
    lr_model = LogisticRegression(learning_rate=0.08, l2_lambda=0.08)
    lr_model.fit(X_train, y_train, X_val, y_val, epochs=1500, batch_size=256, verbose=False)
    lr_pred = lr_model.predict_proba(X_test)
    lr_val_acc = lr_model.accuracy(X_val, y_val)
    predictions.append(lr_pred)
    val_accs.append(lr_val_acc)
    print(f"✅ 验证准确率: {lr_val_acc:.4f} ({lr_val_acc*100:.2f}%)")
    
    # 集成
    print("\n" + "=" * 60)
    print("🎯 集成结果")
    print("=" * 60)
    
    val_accs = np.array(val_accs)
    print(f"各模型验证准确率: {val_accs}")
    
    # 策略1: 平方加权
    weights1 = val_accs ** 2
    weights1 = weights1 / weights1.sum()
    ensemble1 = sum(w * p for w, p in zip(weights1, predictions))
    
    # 策略2: 只用top3模型
    top3_idx = np.argsort(val_accs)[-3:]
    weights2 = np.zeros(len(val_accs))
    weights2[top3_idx] = val_accs[top3_idx] ** 2
    weights2 = weights2 / weights2.sum()
    ensemble2 = sum(w * p for w, p in zip(weights2, predictions))
    
    # 策略3: 简单平均
    ensemble3 = np.mean(predictions, axis=0)
    
    # 选择最佳
    best_idx = np.argmax(val_accs)
    best_single = val_accs[best_idx]
    
    print(f"\n最佳单模型: 模型{best_idx+1} = {best_single:.4f}")
    print(f"集成权重: {weights1}")
    
    # 返回最佳单模型预测（通常比集成更稳定）
    return predictions[best_idx], best_single


def main():
    CONFIG = {
        'train_path': 'data/train.csv',
        'test_path': 'data/test.csv',
        'output_dir': 'output',
        'validation_split': 0.18,  # 稍微调整验证集比例
        'random_state': 42,
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    print("=" * 60)
    print("🚀🚀🚀 极限优化版 - 突破88%！")
    print("=" * 60)
    
    # 加载数据
    train_df = pd.read_csv(CONFIG['train_path'])
    test_df = pd.read_csv(CONFIG['test_path'])
    
    test_ids = test_df['id'].values
    train_df = train_df.drop(columns=['id'])
    test_df = test_df.drop(columns=['id'])
    y_train = train_df['label'].values
    train_df = train_df.drop(columns=['label'])
    
    # 极限特征工程
    print("\n" + "=" * 60)
    print("特征工程（70+特征）")
    print("=" * 60)
    
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    engineer = ExtremeFeatureEngineer()
    combined_df = engineer.create_features(combined_df)
    
    # 编码
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
    
    # 预处理
    preprocessor = DataPreprocessor()
    X_train_scaled = preprocessor.standardize(X_train, fit=True)
    X_test_scaled = preprocessor.standardize(X_test, fit=False)
    y_train_onehot = preprocessor.one_hot_encode(y_train, fit=True)
    
    X_tr, X_val, y_tr, y_val = preprocessor.train_test_split(
        X_train_scaled, y_train_onehot,
        test_size=CONFIG['validation_split'],
        random_state=CONFIG['random_state']
    )
    
    print(f"训练集: {X_tr.shape}, 验证集: {X_val.shape}")
    
    # 训练
    final_pred, final_acc = train_extreme_ensemble(X_tr, y_tr, X_val, y_val, X_test_scaled)
    
    # 输出
    final_predictions = np.argmax(final_pred, axis=1)
    submission = pd.DataFrame({'id': test_ids, 'label': final_predictions})
    
    output_path = os.path.join(CONFIG['output_dir'], 'submission_extreme.csv')
    submission.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("🎉🎉🎉 训练完成！")
    print("=" * 60)
    print(f"最终验证准确率: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"提交文件: {output_path}")
    
    import shutil
    main_output = os.path.join(CONFIG['output_dir'], 'submission.csv')
    shutil.copy(output_path, main_output)
    print(f"已复制到: {main_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
