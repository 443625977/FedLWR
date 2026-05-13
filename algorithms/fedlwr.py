import torch
from torch.utils.data import DataLoader, Subset
from copy import deepcopy

def client_update_fedlwr(model, train_dataset, indices, lr=0.01, local_epochs=5,
                         batch_size=64, device='cpu',
                         beta=0.9, eta=0.1, init_lambda=0.01):
    """
    FedLWR 客户端本地训练：
    - 训练前记录初始梯度 g_init
    - 本地训练中加入逐层 L2 正则项
    - 训练后计算 g_last，更新每层的自适应 lambda
    返回：更新后的参数字典、 更新后的 lambda 字典
    """
    model = model.to(device)
    model.train()
    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # ---------- 1. 记录初始梯度 ----------
    # 取一个 mini-batch
    init_data, init_target = next(iter(loader))
    init_data, init_target = init_data.to(device), init_target.to(device)
    optimizer.zero_grad()
    loss = criterion(model(init_data), init_target)
    loss.backward()
    # 保存每层梯度
    g_init = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g_init[name] = param.grad.clone()

    # ---------- 2. 初始化或读取 lambda ----------
    if not hasattr(client_update_fedlwr, 'lambda_dict'):
        # 首次调用，为每层创建初始 lambda
        client_update_fedlwr.lambda_dict = {}
        for name in g_init.keys():
            client_update_fedlwr.lambda_dict[name] = init_lambda
    lambda_dict = client_update_fedlwr.lambda_dict

    # ---------- 3. 本地训练（带逐层正则） ----------
    # 备份全局权重用于正则项
    global_weights = {name: param.clone() for name, param in model.named_parameters()
                      if name in lambda_dict}

    # 重新创建 loader（因为前面用 next 取过一批）
    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # 添加逐层 L2 正则项： sum_l lambda_l * ||w_l - w_glob_l||^2
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if name in lambda_dict:
                    reg_loss += lambda_dict[name] * torch.sum(
                        (param - global_weights[name]) ** 2
                    )
            total_loss = loss + 0.5 * reg_loss
            total_loss.backward()
            optimizer.step()

    # ---------- 4. 计算最终梯度，更新 lambda ----------
    # 取最后一个 mini-batch
    try:
        last_data, last_target = next(iter(loader))
    except StopIteration:
        # loader 已空，重新创建
        loader = DataLoader(Subset(train_dataset, indices),
                            batch_size=batch_size, shuffle=True)
        last_data, last_target = next(iter(loader))
    last_data, last_target = last_data.to(device), last_target.to(device)
    optimizer.zero_grad()
    loss = criterion(model(last_data), last_target)
    loss.backward()

    g_last = {}
    for name, param in model.named_parameters():
        if name in lambda_dict and param.grad is not None:
            g_last[name] = param.grad.clone()

    # 计算 LGUD 并更新 lambda
    eps = 1e-8
    for name in lambda_dict.keys():
        if name in g_last:
            lgud = torch.norm(g_last[name] - g_init[name]) / \
                   (torch.norm(g_init[name]) + eps)
            lambda_dict[name] = beta * lambda_dict[name] + \
                                (1 - beta) * eta * lgud.item()

    return model.state_dict(), lambda_dict