import torch
from torch.utils.data import DataLoader, Subset

def client_update(model, train_dataset, indices, lr=0.01, local_epochs=5,
                  batch_size=64, device='cpu',
                  c_global=None, c_local=None):
    """
    SCAFFOLD 客户端本地训练（稳健版，加入数值裁剪）。
    返回: (更新后的模型参数字典, 更新后的本地控制变量 c_local_new)
    """
    model = model.to(device)
    model.train()
    
    global_weights = {name: param.clone().detach() 
                      for name, param in model.named_parameters() if param.requires_grad}

    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    if c_local is None:
        c_local = {name: torch.zeros_like(param) 
                   for name, param in model.named_parameters() if param.requires_grad}
    if c_global is None:
        c_global = {name: torch.zeros_like(param) 
                    for name, param in model.named_parameters() if param.requires_grad}

    correction = {name: c_global[name] - c_local[name] for name in c_local}

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # 施加控制变量修正，并裁剪防止爆炸
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in correction and param.grad is not None:
                        # 裁剪控制变量修正量，最大不超过梯度的10倍
                        corr = correction[name]
                        grad_norm = param.grad.norm()
                        corr_norm = corr.norm()
                        if corr_norm > 10 * grad_norm + 1e-8:
                            corr = corr * (10 * grad_norm / corr_norm)
                        param.grad += corr

            optimizer.step()

    # 更新本地控制变量（温和版）：直接使用本轮模型变化量的归一化
    new_c_local = {}
    for name in c_local:
        delta_w = model.state_dict()[name] - global_weights[name]
        # 关键：除以 (lr * local_epochs)，并限制变化幅度
        update = delta_w / (lr * local_epochs + 1e-8)
        # 裁剪控制变量更新量，防止数值过大
        update_norm = update.norm()
        if update_norm > 1.0:  # 限制最大范数为1
            update = update * (1.0 / update_norm)
        new_c_local[name] = c_local[name] - c_global[name] + update

    return model.state_dict(), new_c_local


def server_aggregate(global_model, client_updates, client_sizes, c_global):
    """
    SCAFFOLD 聚合（稳健版）。
    """
    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()

    # 1. 模型参数加权平均
    avg_dict = {key: torch.zeros_like(value) for key, value in global_dict.items()}
    for (w_dict, _), size in zip(client_updates, client_sizes):
        for key in avg_dict:
            avg_dict[key] += (size / total_size) * w_dict[key]
    global_model.load_state_dict(avg_dict)

    # 2. 更新服务器控制变量：使用平滑移动平均，避免剧烈变化
    K = len(client_updates)
    if K > 0:
        for name in c_global:
            avg_c_new = sum([new_c[name] for _, new_c in client_updates]) / K
            # 温和更新：新值 = 0.9 * 旧值 + 0.1 * 新平均值
            c_global[name] = 0.9 * c_global[name] + 0.1 * avg_c_new

    return global_model, c_global