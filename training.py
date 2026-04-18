import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
df = pd.read_csv('cs-training.csv')
df = df.drop(df.columns[0], axis=1)

# 2. 将列名改为中文
df.columns = [
    '是否逾期', '额度使用率', '年龄', '逾期30-59天次数', '负债比率',
    '月收入', '未偿还贷款数量', '逾期90天以上次数', '房地产贷款数量',
    '逾期60-89天次数', '家属人数'
]

# ========== 3. 处理缺失值 ==========
df['月收入'] = df['月收入'].fillna(df['月收入'].median())#用中位数可以减小极端值影响
df['家属人数'] = df['家属人数'].fillna(0)
print("处理后缺失值统计:")
print(df.isnull().sum())

# ========== 4. 处理异常值 ==========
df = df[df['年龄'] > 0]
df['负债比率'] = df['负债比率'].clip(upper=100) #截断负债率大于100的数据
#============逾期次数不为负值==========
df['逾期30-59天次数'] = df['逾期30-59天次数'].clip(lower=0)
df['逾期60-89天次数'] = df['逾期60-89天次数'].clip(lower=0)
df['逾期90天以上次数'] = df['逾期90天以上次数'].clip(lower=0)
print("\n清洗后数据形状:", df.shape)

# ========== 5. 探索性分析（在标准化之前） ==========
# 年龄分箱
df['年龄段'] = pd.cut(df['年龄'], bins=[0, 30, 40, 50, 60, 100],
                        labels=['30岁以下', '30-40岁', '40-50岁', '50-60岁', '60岁以上'])
age_churn = df.groupby('年龄段', observed=False)['是否逾期'].mean()
print("\n各年龄段逾期率:")
print(age_churn)

# 逾期次数分析
churn_by_30_59 = df.groupby('逾期30-59天次数', observed=False)['是否逾期'].mean()
print("\n逾期30-59天次数对应的逾期率:")
print(churn_by_30_59)

# 收入分箱
df['收入分组'] = pd.cut(df['月收入'], bins=[0, 2000, 5000, 10000, 50000, 200000],
                          labels=['0-2k', '2k-5k', '5k-10k', '10k-50k', '50k+'])
income_churn = df.groupby('收入分组', observed=False)['是否逾期'].mean()
print("\n各收入段逾期率:")
print(income_churn)

# ========== 生成图片：年龄与逾期率柱状图 ==========
plt.figure(figsize=(10, 6))
age_churn.plot(kind='bar', color='steelblue')
plt.title('各年龄段逾期率', fontsize=14)
plt.xlabel('年龄段', fontsize=12)
plt.ylabel('逾期率', fontsize=12)
plt.xticks(rotation=45,ha='right')
plt.ylim(0, age_churn.max()*1.15)  # 设置y轴范围，让图表更直观
# 在柱子上方添加数值标签
for i, v in enumerate(age_churn):
    plt.text(i, v + (age_churn.max()*0.01), f'{v:.2%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('age_churn.png', dpi=300, bbox_inches='tight')
plt.show()
print("图片已保存：age_churn.png")

# ========== 生成图片：收入与逾期率柱状图 ==========
plt.figure(figsize=(10, 6))
income_churn.plot(kind='bar', color='coral')
plt.title('各收入段逾期率', fontsize=14)
plt.xlabel('月收入', fontsize=12)
plt.ylabel('逾期率', fontsize=12)
plt.xticks(rotation=45,ha='right')
plt.ylim(0, income_churn.max()*1.15)
for i, v in enumerate(income_churn):
    plt.text(i, v + (income_churn.max()*0.01), f'{v:.2%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('income_churn.png', dpi=300, bbox_inches='tight')
plt.show()
print("图片已保存：income_churn.png")

# ========== 生成图片：逾期次数与逾期率折线图 ==========
# 只取前10次逾期次数（后面次数样本量小，波动大）
churn_plot = churn_by_30_59.head(10)
plt.figure(figsize=(10, 6))
plt.plot(churn_plot.index, churn_plot.values, marker='o', linestyle='-', color='green')
plt.title('逾期30-59天次数与逾期率的关系', fontsize=14)
plt.xlabel('逾期30-59天次数', fontsize=12)
plt.ylabel('逾期率', fontsize=12)
plt.xticks(range(0, 11))
plt.ylim(0, churn_plot.max()*1.15)
# 添加数值标签
for i, v in enumerate(churn_plot.values):
    plt.text(churn_plot.index[i], v + (churn_plot.max()*0.02), f'{v:.1%}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('churn_by_days.png', dpi=300, bbox_inches='tight')
plt.show()
print("图片已保存：churn_by_days.png")

# ========== 6. 特征工程：对数变换 + 标准化 ==========
# 对数变换：log(1+x)
df['月收入_log'] = np.log1p(df['月收入'])
df['额度使用率_log'] = np.log1p(df['额度使用率'])

# 标准化（使用对数变换后的列）
features_to_scale = ['额度使用率_log', '年龄', '逾期30-59天次数', '负债比率',
                      '月收入_log', '未偿还贷款数量', '逾期90天以上次数',
                      '房地产贷款数量', '逾期60-89天次数', '家属人数']

# 删除临时分组列（这些列不参与建模）
df_model = df.drop(['年龄段', '收入分组'], axis=1)
print(f"✅ 建模特征: {features_to_scale}")
print(f"✅ 模型数据形状: {df_model.shape}")

# ========== 7. 建模 ==========
print("\n7. 建模与评估")
print("-" * 40)

# 定义特征和目标
X = df_model[features_to_scale]
y = df_model['是否逾期']

# 1. 先划分训练集和测试集（70% 训练，30% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y  # stratify=y 保证逾期比例一致
)

print(f"训练集大小: {len(X_train)} 条")
print(f"测试集大小: {len(X_test)} 条")
print(f"训练集逾期率: {y_train.mean():.4f}")
print(f"测试集逾期率: {y_test.mean():.4f}")

# 2. 在训练集上创建并训练 StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 只在训练集上 fit

# 3. 用同一个 scaler 转换测试集（只 transform，不 fit）
X_test_scaled = scaler.transform(X_test)

print("✅ 标准化完成（无数据泄露：仅在训练集上计算均值和标准差）")

# 4. 训练模型
model = LogisticRegression(
    class_weight='balanced',  # 处理类别不平衡
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("✅ 模型训练完成")

# 5. 模型评估
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n📊 模型评估结果:")
print(f"   AUC: {auc:.4f}")

# 可选：打印分类报告
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_scaled)
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['未逾期', '逾期']))

# 6. 特征重要性（系数分析）
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_[0]
}).sort_values('系数', ascending=False)

print("\n📊 特征重要性（系数越大，对逾期预测的影响越大）:")
print(feature_importance)

# 可视化特征重要性
fig,ax=plt.subplots(figsize=(10, 6))
top_features = feature_importance.head(8)
colors = ['red' if x > 0 else 'green' for x in top_features['系数']]

bars=ax.barh(range(len(top_features)), top_features['系数'], color=colors)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['特征'])
ax.set_xlabel('系数值', fontsize=12)
ax.set_title('特征重要性排名（正系数=增加逾期风险）', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.invert_yaxis()

ax.bar_label(bars, fmt='%.3f', padding=5, fontsize=9, fontweight='bold')

ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ 已保存: feature_importance.png")