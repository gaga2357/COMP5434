import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('data/train.csv')

print('数据分析:')
print('='*60)
print(df.describe())

print('\n特征信息:')
print(df.info())

print('\n类别特征:')
for col in df.select_dtypes(include=['object']).columns:
    print(f'{col}: {df[col].nunique()} 类别 - {df[col].unique()[:5]}')

print('\n相关性分析 (与label的相关性):')
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr()['label'].sort_values(ascending=False))

print('\n缺失值检查:')
print(df.isnull().sum())
