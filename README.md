# FedLWR：面向异构联邦学习的逐层自适应正则化算法

## 运行环境

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib

## 环境配置

```bash

# 创建虚拟环境

python -m venv fl_env

# 激活虚拟环境（Windows）

fl_env\Scripts\activate

# 激活虚拟环境（macOS / Linux）

source fl_env/bin/activate

# 安装依赖

pip install torch torchvision numpy matplotlib

数据集准备
本项目使用 CIFAR-10 和 CIFAR-100 数据集。

方式一：自动下载（推荐）
直接运行任意实验脚本（如 python main.py），torchvision 会自动下载数据集到 data/ 文件夹。首次运行需要联网，下载进度条会显示在终端。

方式二：手动下载（网络受限时）
从以下链接下载数据集压缩包：

CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz

将下载的 .tar.gz 文件放入 data/ 文件夹。

用解压软件（如 7-Zip）解压到当前目录，确保最终路径为：

data/cifar-10-batches-py/

data/cifar-100-python/

（非必需）如需关闭自动下载，可修改 data/data_loader.py 中的 download=True 改为 download=False。

运行命令
运行前请先激活虚拟环境：

bash

# Windows

fl_env\Scripts\activate

# macOS / Linux

source fl_env/bin/activate
FedAvg 基线

bash

python main.py
FedProx 基线

bash

python main_prox.py
FedLWR（本文方法）

bash

python main_lwr_v2.py        # CIFAR-10
python main_lwr_cifar100.py  # CIFAR-100
SCAFFOLD 基线

bash

python main_scaffold.py
FedDyn 基线

bash

python main_feddyn.py
消融实验

bash

python main_fixed.py    # 消融1：无自适应机制（固定λ）
python main_global.py   # 消融2：无逐层机制（全局λ）
CIFAR-100 实验

bash

python main_cifar100.py        # FedAvg 基线
python main_lwr_cifar100.py    # FedLWR
生成论文图表

bash

python plot.py

项目结构

text

FedLWR/
├── data/
│   └── data_loader.py          # 数据加载与Dirichlet非IID划分
├── models/
│   └── nets.py                 # 模型定义（SimpleCNN / MobileNetV2Cifar）
├── algorithms/
│   ├── fedavg.py               # FedAvg 算法
│   ├── fedprox.py              # FedProx 算法
│   ├── fedlwr_v2.py            # FedLWR 算法（本文核心）
│   ├── fedlwr_fixed.py         # 消融变体：固定λ
│   ├── fedlwr_global.py        # 消融变体：全局λ
│   ├── scaffold.py             # SCAFFOLD 算法
│   └── feddyn.py               # FedDyn 算法
├── main.py                     # FedAvg 运行脚本
├── main_prox.py                # FedProx 运行脚本
├── main_lwr_v2.py              # FedLWR CIFAR-10 运行脚本
├── main_lwr_cifar100.py        # FedLWR CIFAR-100 运行脚本
├── main_cifar100.py            # CIFAR-100 FedAvg 运行脚本
├── main_scaffold.py            # SCAFFOLD 运行脚本
├── main_feddyn.py              # FedDyn 运行脚本
├── main_fixed.py               # 消融1运行脚本
├── main_global.py              # 消融2运行脚本
├── plot.py                     # 论文图表生成脚本
├── convergence_cifar10.pdf     # 收敛曲线图
├── lambda_dynamics.pdf         # λ动态变化图
├── ablation_study.pdf          # 消融实验图
└── README.md

License

MIT License

text
