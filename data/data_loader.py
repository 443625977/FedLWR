import torch
from torchvision import datasets, transforms
import numpy as np

def get_cifar10(data_root='./data'):
    """下载并返回CIFAR-10的训练集和测试集（未切分）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10(root=data_root, train=True,
                                     download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False,
                                    download=False, transform=transform)
    return train_dataset, test_dataset

def dirichlet_partition(dataset, num_clients=50, alpha=0.5, seed=42):
    """
    使用Dirichlet分布将数据集划分为num_clients份，
    alpha越小，每个客户端的数据类别分布越不均匀。
    返回: client_indices: list of lists，每个元素是该客户端所含样本的索引列表。
    """
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    n_classes = len(np.unique(labels))
    proportions = np.random.dirichlet(np.repeat(alpha, n_classes),
                                      size=num_clients)

    class_indices = [np.where(labels == i)[0] for i in range(n_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        n_samples_per_class = len(class_indices[c])
        proportions_c = proportions[:, c]
        proportions_c = proportions_c / proportions_c.sum()
        n_split = (proportions_c * n_samples_per_class).astype(int)
        diff = n_samples_per_class - n_split.sum()
        n_split[0] += diff
        np.random.shuffle(class_indices[c])
        split_points = np.cumsum(n_split)[:-1]
        splits = np.split(class_indices[c], split_points)
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    for k in range(num_clients):
        np.random.shuffle(client_indices[k])

    return client_indices
def get_cifar100(data_root='./data'):
    """下载/加载 CIFAR-100（手动下载后设置 download=False）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    # 如果已手动下载并解压到 data/cifar-100-python，则 download=False
    train_dataset = datasets.CIFAR100(root=data_root, train=True,
                                      download=False, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_root, train=False,
                                     download=False, transform=transform)
    return train_dataset, test_dataset