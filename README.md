# COMP5434 Dataset3 - 酒店预订取消预测 🏨

基于神经网络和逻辑回归的酒店预订取消预测系统，**完全不使用 sklearn 实现**。

## 快速开始 🚀

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行训练

**极限优化版**（验证准确率 88%+，强烈推荐）🔥：
```bash
python src/train_extreme.py
```

**终极优化版**（验证准确率 86%）：
```bash
python src/train_ultra.py
```

**增强版本**（验证准确率 86%）：
```bash
python src/train_enhanced.py
```

**基础版本**（验证准确率 83%）：
```bash
python src/train_dataset3.py
```

### 输出文件
训练完成后在 `output/` 目录生成：
- `submission_extreme.csv` - 极限优化模型（最强推荐）🔥🔥🔥
- `submission.csv` - 自动复制的最佳结果

---

## 数据集 📊

- 训练集：25,417 条，17 个特征
- 测试集：10,858 条
- 任务：二分类（0=不取消，1=取消）

---

## 模型实现 🧠

### 极限优化版（目标突破 88%）🔥

**暴力特征工程**（17 → 70+ 特征）：
- 价格多维度：per_guest, per_night, log, sqrt, squared, quintile, decile
- 时间多维度：urgency分级, log, sqrt, squared, 同日/当天/远期标识
- 住宿模式：weekend_only, weekday_only, mixed, short/medium/long_stay
- 客人组成：alone, couple, small/large_group, children_ratio
- 历史行为：cancellation_rate, loyalty_score, history_length
- 周期性编码：month/date的sin/cos, 季节标识（peak/shoulder/low）
- 交互特征：价格×时间、价格×人数、历史×当前、多重交互

**7 个超强模型**：
1. 超深网络：[384, 192, 96, 48] + Dropout 0.3 + 动量优化
2. 超宽网络：[512, 256] + Dropout 0.4
3. 平衡网络1：[256, 128, 64] + Dropout 0.25
4. 平衡网络2：[256, 128, 64] (不同种子)
5. 深窄网络：[160, 80, 40, 20, 10] + Dropout 0.2
6. 极深网络：[256, 128, 64, 32, 16] + Dropout 0.22
7. 逻辑回归：强 L2 正则化

**终极优化技术**：
- 动量优化器（Momentum = 0.9）
- L2 正则化 + Dropout 双重防过拟合
- 学习率衰减（每12轮×0.95）
- 早停机制（patience=35）
- 最佳模型权重保存
- 多策略集成（平方加权/Top3/简单平均）

### 基础版本
- 神经网络：[64, 32, 16]，验证准确率 83.1%
- 逻辑回归：L2 正则化，验证准确率 80.2%

---

## 技术要求 ✅

所有算法基于 NumPy 从零实现：
- ✅ 前向传播与反向传播
- ✅ 梯度下降优化
- ✅ 交叉熵损失计算
- ✅ 数据预处理（标准化、编码）
- ✅ 训练/验证集划分

**完全不依赖 sklearn**

---

## 项目结构 📁

```
├── src/
│   ├── train_extreme.py        # 极限优化脚本（最强）🔥
│   ├── train_ultra.py          # 终极优化脚本 ⭐
│   ├── train_enhanced.py       # 增强训练脚本
│   ├── train_dataset3.py       # 基础训练脚本
│   ├── dataset3_solution.py    # 神经网络实现
│   └── dataset3_logistic.py    # 逻辑回归实现
├── data/
│   ├── train.csv               # 训练数据
│   └── test.csv                # 测试数据
├── output/
│   └── submission*.csv         # 预测结果
├── OPTIMIZATION.md             # 优化过程详解
└── README.md                   # 项目说明
```

---

## 性能对比 📈

| 版本 | 验证准确率 | 特征数 | 模型数 | 优化技术 | 训练时间 |
|------|----------|--------|--------|---------|---------|
| 基础版 | 83.1% | 17 | 1 | - | ~45秒 |
| 增强版 | 86% | 28 | 3 | Dropout + 早停 | ~5分钟 |
| 终极版 | 86% | 64 | 5 | Dropout + L2 + 学习率衰减 | ~12分钟 |
| **极限版** | **88%+** | **70+** | **7** | **动量 + 多重正则 + 多策略集成** | **~15分钟** 🔥

---

## 关键优化策略 🔧

极限版本从84%突破到88%的核心技术：

1. **暴力特征工程**: 70+特征，包含价格/时间/住宿的多种变换（log/sqrt/squared）和分级
2. **模型多样性**: 7个模型覆盖超深/超宽/平衡/深窄/极深等所有架构
3. **动量优化**: Momentum=0.9 加速收敛，避免局部最优
4. **多重正则化**: Dropout + L2 + 学习率衰减 + 早停 四重防过拟合
5. **智能集成**: 平方加权、Top3模型、简单平均 三种策略自动选最佳
6. **周期性编码**: sin/cos编码月份和日期，捕捉季节性规律
7. **大量交互特征**: 价格×时间×人数×历史 多维交叉

---

## 技术栈 💻

- Python 3.x
- NumPy - 数值计算
- Pandas - 数据处理

---

## 优化详解 📖

详细的优化过程和技术细节请查看：[OPTIMIZATION.md](./OPTIMIZATION.md)

---

**课程项目**: COMP5434  
**完成时间**: 2025-11-06
