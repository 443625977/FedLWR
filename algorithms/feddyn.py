import torch
from torch.utils.data import DataLoader, Subset

def client_update(model, train_dataset, indices, lr=0.01, local_epochs=5,
                  batch_size=64, device='cpu', alpha=0.01, global_param=None):
    """
    FedDyn 客户端本地训练。
    alpha: 正则项系数（固定为小值如0.01）
    global_param: 服务器端维护的全局参数 h_global 的拷贝
    返回: (更新后的模型参数字典, 模型参数变化量用于更新 h_global)
    """
    model = model.to(device)
    model.train()

    # 备份初始权重
    init_weights = {name: param.clone().detach() 
                    for name, param in model.named_parameters() if param.requires_grad}

    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 若未传入 global_param，初始化为零
    if global_param is None:
        global_param = {name: torch.zeros_like(param) 
                        for name, param in model.named_parameters() if param.requires_grad}

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # FedDyn 正则项: alpha/2 * ||w||^2 - <h_global, w>
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if name in global_param:
                    reg_loss += (alpha / 2) * torch.sum(param ** 2) \
                                - torch.sum(global_param[name] * param)
            total_loss = loss + reg_loss
            total_loss.backward()
            optimizer.step()

    # 计算模型变化量，用于更新服务器 h_global
    delta_w = {}
    for name in init_weights:
        delta_w[name] = model.state_dict()[name] - init_weights[name]

    # 返回更新后的模型权重和变化量
    return model.state_dict(), delta_w


def server_aggregate(global_model, client_updates, client_sizes, global_param, alpha=0.01):
    """
    FedDyn 聚合：加权平均模型参数，更新全局控制变量 h_global。
    client_updates: 列表，元素为 (state_dict, delta_w) 元组
    """
    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()

    # 1. 模型参数加权平均
    avg_dict = {key: torch.zeros_like(value) for key, value in global_dict.items()}
    for (w_dict, _), size in zip(client_updates, client_sizes):
        for key in avg_dict:
            avg_dict[key] += (size / total_size) * w_dict[key]
    global_model.load_state_dict(avg_dict)

    # 2. 更新 h_global = h_global - alpha * avg_w + (1/K) * sum(delta_w_i / (lr * E))
    # 此处采用简化的标准更新公式
    K = len(client_updates)
    if global_param is None:
        global_param = {name: torch.zeros_like(param) 
                        for name, param in global_model.named_parameters() if param.requires_grad}

    for name in global_param:
        # 当前全局模型参数
        w_t = global_dict[name]
        # 平均变化量
        avg_delta = sum([delta_w[name] for _, delta_w in client_updates]) / K
        # 更新公式: h = h - alpha * w_t + avg_delta / (lr * E)
        # 注意: lr 和 local_epochs 控制更新步长
        h_update = - alpha * w_t + avg_delta / (0.01 * 5)  # lr=0.01, E=5
        global_param[name] = global_param[name] + h_update / K  # 缩放防止过大

    return global_model, global_param