import matplotlib.pyplot as plt
import numpy as np

# ========== 中文字体设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 全局设置 ==========
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf'
})

# ========== 数据 ==========
rounds = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

fedavg = [26.67, 36.35, 39.15, 41.49, 50.74, 50.77, 49.76, 55.87, 58.18, 49.05,
          55.97, 61.25, 59.56, 59.97, 54.88, 51.11, 57.62, 59.10, 51.44, 55.01, 62.47]

fedprox = [26.37, 36.92, 40.83, 42.50, 51.86, 50.77, 50.45, 56.70, 58.79, 49.30,
           56.45, 60.90, 61.07, 61.22, 55.94, 52.70, 58.20, 60.48, 51.10, 55.21, 63.23]

fedlwr = [26.62, 37.92, 37.71, 43.46, 49.50, 50.44, 48.25, 56.97, 59.72, 48.72,
          56.25, 60.58, 59.67, 59.18, 54.52, 51.34, 57.69, 58.72, 50.27, 55.79, 63.52]

lambda_fc2 = [0.0167, 0.0206, 0.0137, 0.0207, 0.0333, 0.0288, 0.0285, 0.0344, 0.0379,
              0.0343, 0.0374, 0.0386, 0.0364, 0.0423, 0.0388, 0.0441, 0.0456, 0.0448,
              0.0407, 0.0447, 0.0472]

methods = ['完整FedLWR', '无自适应机制\n(固定λ系数)', '无逐层机制\n(全局λ系数)']
ablation_acc = [63.52, 63.26, 63.12]
colors_abl = ['#2E86AB', '#A23B72', '#F18F01']

# ========== 图1: 收敛曲线 (中文版) ==========
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(rounds, fedavg, 's-', color='#D81159', alpha=0.8, label='FedAvg')
ax1.plot(rounds, fedprox, '^--', color='#8F3985', alpha=0.8, label='FedProx')
ax1.plot(rounds, fedlwr, 'o-', color='#2E86AB', linewidth=3, label='FedLWR (本文)')

ax1.set_xlabel('通信轮次')
ax1.set_ylabel('测试准确率 (%)')
ax1.set_title('收敛曲线对比 (CIFAR-10, α=0.1, 种子42单次运行)')

ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(0, 105)
ax1.set_ylim(20, 70)

# 左上角文本框
textstr = f'最终准确率:\nFedAvg: {fedavg[-1]:.2f}%\nFedProx: {fedprox[-1]:.2f}%\nFedLWR: {fedlwr[-1]:.2f}%'
props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
fig1.savefig('convergence_cifar10.pdf')
print('图1 已保存: convergence_cifar10.pdf')

# ========== 图2: λ 动态变化 (中文版) ==========
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(rounds, lambda_fc2, 'D-', color='#A23B72', markersize=7, label='分类器层 (fc2)')

ax2.set_xlabel('通信轮次')
ax2.set_ylabel('自适应系数 λ')
ax2.set_title('FedLWR 自适应正则系数变化曲线 (CIFAR-10, α=0.1)')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(0, 105)
ax2.set_ylim(0, 0.06)

# 趋势线
z = np.polyfit(rounds, lambda_fc2, 3)
p = np.poly1d(z)
ax2.plot(rounds, p(rounds), '--', color='gray', alpha=0.5, label='趋势线')
ax2.legend()

plt.tight_layout()
fig2.savefig('lambda_dynamics.pdf')
print('图2 已保存: lambda_dynamics.pdf')

# ========== 图3: 消融实验 (中文版) ==========
fig3, ax3 = plt.subplots(figsize=(8, 6))
bars = ax3.bar(methods, ablation_acc, color=colors_abl, edgecolor='black', linewidth=1.2, width=0.5)

for bar, acc in zip(bars, ablation_acc):
    ax3.text(bar.get_x() + bar.get_width()/2., acc + 0.15, f'{acc:.2f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=13)

ax3.set_ylabel('测试准确率 (%)')
ax3.set_title('消融实验 (CIFAR-10, α=0.1, 100轮)')
ax3.set_ylim(62.0, 64.0)
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
fig3.savefig('ablation_study.pdf')
print('图3 已保存: ablation_study.pdf')

print('\n全部中文图表已生成！')