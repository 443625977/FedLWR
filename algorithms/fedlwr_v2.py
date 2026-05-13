import torch
from torch.utils.data import DataLoader, Subset

def client_update_fedlwr(model, train_dataset, indices, lr=0.01, local_epochs=5,
                         batch_size=64, device='cpu',
                         beta=0.9, eta=0.1, init_lambda=0.01,
                         lambda_dict=None):
    """
    FedLWR 客户端本地训练（V2：每个客户端独立 lambda）。
    lambda_dict: 该客户端自身的 lambda 字典，若为 None 则初始化。
    返回：更新后的参数字典、更新后的 lambda 字典
    """
    model = model.to(device)
    model.train()
    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # ------- 1. 初始梯度 -------
    init_data, init_target = next(iter(loader))
    init_data, init_target = init_data.to(device), init_target.to(device)
    optimizer.zero_grad()
    loss = criterion(model(init_data), init_target)
    loss.backward()
    g_init = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g_init[name] = param.grad.clone()

    # ------- 2. 该客户端的 lambda（独立） -------
    if lambda_dict is None:
        lambda_dict = {}
        for name in g_init.keys():
            lambda_dict[name] = init_lambda
    else:
        # 确保新层也有 lambda（模型结构不变，通常不会执行）
        for name in g_init.keys():
            if name not in lambda_dict:
                lambda_dict[name] = init_lambda

    # ------- 3. 本地训练（逐层正则） -------
    global_weights = {name: param.clone() for name, param in model.named_parameters()
                      if name in lambda_dict}

    loader = DataLoader(Subset(train_dataset, indices),
                        batch_size=batch_size, shuffle=True)

    for _ in range(local_epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            reg_loss = 0.0
            for name, param in model.named_parameters():
                if name in lambda_dict:
                    reg_loss += lambda_dict[name] * torch.sum(
                        (param - global_weights[name]) ** 2
                    )
            total_loss = loss + 0.5 * reg_loss
            total_loss.backward()
            optimizer.step()

    # ------- 4. 最终梯度并更新 lambda -------
    try:
        last_data, last_target = next(iter(loader))
    except StopIteration:
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

    eps = 1e-8
    for name in lambda_dict.keys():
        if name in g_last:
            lgud = torch.norm(g_last[name] - g_init[name]) / \
                   (torch.norm(g_init[name]) + eps)
            lambda_dict[name] = beta * lambda_dict[name] + \
                                (1 - beta) * eta * lgud.item()

    return model.state_dict(), lambda_dict