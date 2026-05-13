import torch
from torch.utils.data import DataLoader, Subset

def client_update(model, train_dataset, indices, lr=0.01, local_epochs=5,
                  batch_size=64, device='cpu', mu=0.01):
    """
    FedProx 客户端本地训练，返回更新后的模型参数字典。
    mu: 近端项系数
    """
    model = model.to(device)
    model.train()
    # 保存全局模型权重，用于正则项
    global_weights = {name: param.clone().detach() for name, param in model.named_parameters()}

    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # ---- FedProx 近端项 ----
            prox_term = 0.0
            for name, param in model.named_parameters():
                prox_term += torch.sum((param - global_weights[name]) ** 2)
            loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

    return model.state_dict()


def server_aggregate(global_model, client_weights, client_sizes):
    """FedProx 聚合（与 FedAvg 完全相同，加权平均）"""
    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()
    avg_dict = {key: torch.zeros_like(value) for key, value in global_dict.items()}

    for w_dict, size in zip(client_weights, client_sizes):
        for key in avg_dict:
            avg_dict[key] += (size / total_size) * w_dict[key]

    global_model.load_state_dict(avg_dict)
    return global_model