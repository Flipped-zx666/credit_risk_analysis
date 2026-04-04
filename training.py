import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# ========== 解决中文显示问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 1. 读取数据
df = pd.read_csv('cs-training.csv')
df = df.drop(df.columns[0], axis=1)

# 2. 将列名改为中文
df.columns = [
    '是否逾期',           # SeriousDlqin2yrs
    '额度使用率',         # RevolvingUtilizationOfUnsecuredLines
    '年龄',               # age
    '逾期30-59天次数',    # NumberOfTime30-59DaysPastDueNotWorse
    '负债比率',           # DebtRatio
    '月收入',             # MonthlyIncome
    '未偿还贷款数量',     # NumberOfOpenCreditLinesAndLoans
    '逾期90天以上次数',   # NumberOfTimes90DaysLate
    '房地产贷款数量',     # NumberRealEstateLoansOrLines
    '逾期60-89天次数',    # NumberOfTime60-89DaysPastDueNotWorse
    '家属人数'            # NumberOfDependents
]

# ========== 1. 处理缺失值（改用新写法，避免警告） ==========

# 月收入：用中位数填充
df['月收入'] = df['月收入'].fillna(df['月收入'].median())

# 家属人数：用0填充
df['家属人数'] = df['家属人数'].fillna(0)

# 验证缺失值是否已处理完
print("处理后缺失值统计:")
print(df.isnull().sum())


# ========== 2. 处理异常值 ==========

# 年龄不能为0或负数
df = df[df['年龄'] > 0]

# 负债比率：超过100的按100算
df['负债比率'] = df['负债比率'].clip(upper=100)

# 逾期次数不能为负数
df['逾期30-59天次数'] = df['逾期30-59天次数'].clip(lower=0)
df['逾期60-89天次数'] = df['逾期60-89天次数'].clip(lower=0)
df['逾期90天以上次数'] = df['逾期90天以上次数'].clip(lower=0)


# ========== 3. 查看清洗后数据 ==========

print("\n清洗后数据形状:", df.shape)
print("\n清洗后描述性统计:")
print(df.describe())


# ========== 4. 探索性分析：年龄与逾期的关系 ==========

# 年龄分箱
df['年龄段'] = pd.cut(df['年龄'], bins=[0, 30, 40, 50, 60, 100],
                        labels=['30岁以下', '30-40岁', '40-50岁', '50-60岁', '60岁以上'])

# 各年龄段逾期率
age_churn = df.groupby('年龄段', observed=False)['是否逾期'].mean().sort_values(ascending=False)
print("\n各年龄段逾期率:")
print(age_churn)

# 可视化
plt.figure(figsize=(10, 6))
age_churn.plot(kind='bar', color='steelblue')
plt.title('各年龄段逾期率')
plt.xlabel('年龄段')
plt.ylabel('逾期率')
plt.xticks(rotation=45)
plt.show()

# ========== 5. 逾期次数与是否逾期的关系 ==========

churn_by_30_59 = df.groupby('逾期30-59天次数',observed=False)['是否逾期'].mean()
print("\n逾期30-59天次数对应的逾期率:")
print(churn_by_30_59)


# ========== 6. 收入与逾期的关系 ==========

# 收入分箱（单位：千元）
df['收入分组'] = pd.cut(df['月收入'], bins=[0, 2000, 5000, 10000, 50000, 200000],
                          labels=['0-2k', '2k-5k', '5k-10k', '10k-50k', '50k+'])

income_churn = df.groupby('收入分组',observed=False)['是否逾期'].mean()
print("\n各收入段逾期率:")
print(income_churn)

# 可视化
plt.figure(figsize=(10, 6))
income_churn.plot(kind='bar', color='coral')
plt.title('各收入段逾期率')
plt.xlabel('月收入')
plt.ylabel('逾期率')
plt.xticks(rotation=45)
plt.show()

# 1. 准备数据（删除临时用的分组列）
df_model = df.drop(['年龄段', '收入分组'], axis=1)

# 2. 定义特征和目标
X = df_model.drop('是否逾期', axis=1)
y = df_model['是否逾期']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 建模（用 class_weight 处理样本不平衡）
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. 预测和评估
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC: {auc:.4f}")

# 6. 特征重要性
feature_importance = pd.DataFrame({
    '特征': X.columns,
    '系数': model.coef_[0]
}).sort_values('系数', ascending=False)

print("\n特征重要性（系数越大越重要）:")
print(feature_importance)