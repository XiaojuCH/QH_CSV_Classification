import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取所有数据文件
data_files = ['DH.csv', 'KD.csv', 'PS10.csv', 'PS10-H.csv', 'QZ.csv', 'YM.csv']
labels = ['DH', 'KD', 'PS10', 'PS10-H', 'QZ', 'YM']

# 加载数据
all_data = []
all_labels = []

for file, label in zip(data_files, labels):
    df = pd.read_csv(file, header=None)
    all_data.append(df.values)
    all_labels.extend([label] * len(df))

# 合并数据
X = np.vstack(all_data)
y = np.array(all_labels)

print(f"数据形状: {X.shape}")
print(f"类别分布:")
for label in labels:
    count = np.sum(y == label)
    print(f"  {label}: {count} ({count/len(y)*100:.2f}%)")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练多个模型
models = {
    'Random Forest': RandomForestClassifier(n_estimators=400, max_depth=30, min_samples_split=3, random_state=42, n_jobs=-1, verbose=0),
    'LightGBM': LGBMClassifier(n_estimators=500, max_depth=15, learning_rate=0.03, num_leaves=80, random_state=42, n_jobs=-1, verbose=-1),
    'SVM': SVC(kernel='rbf', C=150, gamma='scale', random_state=42)
}

results = {}

for name, model in tqdm(models.items(), desc="训练模型"):
    print(f"\n{'='*60}")
    print(f"训练 {name}...")
    print(f"{'='*60}")
    model.fit(X_train_scaled, y_train)

    # 预测
    print("预测中...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # 交叉验证
    print("交叉验证中...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, verbose=0)

    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"\n✓ {name} 测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ {name} 交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred)}")

# 找出最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"\n{'='*60}")
print(f"最佳模型: {best_model_name}")
print(f"测试准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"{'='*60}")

# 绘制最佳模型的混淆矩阵
print(f"\n绘制 {best_model_name} 混淆矩阵...")
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best, labels=labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 计算TPR和FNR
tpr = np.diag(cm) / cm.sum(axis=1) * 100
fnr = 100 - tpr

# 创建带TPR/FNR的扩展矩阵
cm_extended = np.zeros((len(labels), len(labels) + 2))
cm_extended[:, :-2] = cm_percent
cm_extended[:, -2] = tpr
cm_extended[:, -1] = fnr

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm_extended, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=list(labels) + ['TPR', 'FNR'],
            yticklabels=labels,
            cbar_kws={'label': '百分比 (%)'},
            linewidths=0.5, linecolor='white', ax=ax)
plt.title(f'{best_model_name}\n准确率: {best_accuracy*100:.2f}%',
          fontsize=13, pad=15)
plt.ylabel('真实类别', fontsize=11)
plt.xlabel('预测类别', fontsize=11)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 confusion_matrix.png")

# 保存结果
with open('results.txt', 'w', encoding='utf-8') as f:
    f.write("模型性能对比\n")
    f.write("="*60 + "\n\n")
    for name, result in results.items():
        f.write(f"{name}:\n")
        f.write(f"  测试准确率: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
        f.write(f"  交叉验证: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n\n")
    f.write(f"\n最佳模型: {best_model_name}\n")
    f.write(f"最终准确率: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")

print("\n结果已保存到 results.txt")
