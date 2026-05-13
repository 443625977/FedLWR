import torch 
from torch.utils.data import DataLoader, Subset 
 
def client_update(model, train_dataset, indices, lr=0.01, local_epochs=5, batch_size=64, device='cpu'): 
    model = model.to(device) 
    model.train() 
    loader = DataLoader(Subset(train_dataset, indices), batch_size=batch_size, shuffle=True) 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    criterion = torch.nn.CrossEntropyLoss() 
 
    for _ in range(local_epochs): 
        for data, target in loader: 
            data, target = data.to(device), target.to(device) 
            optimizer.zero_grad() 
            output = model(data) 
            loss = criterion(output, target) 
            loss.backward() 
            optimizer.step() 
 
    return model.state_dict() 
 
 
def server_aggregate(global_model, client_weights, client_sizes):
    """
    FedAvg 聚合：按客户端数据量加权平均。
    跳过非浮点参数（如 BatchNorm 的 num_batches_tracked）。
    """
    total_size = sum(client_sizes)
    global_dict = global_model.state_dict()
    avg_dict = {}
    # 只对浮点参数进行聚合，其他参数保持全局模型原值
    for key, value in global_dict.items():
        if value.dtype in (torch.float32, torch.float64, torch.float16):
            avg_dict[key] = torch.zeros_like(value)
        else:
            avg_dict[key] = value.clone()  # 保持原值

    for w_dict, size in zip(client_weights, client_sizes):
        for key in avg_dict:
            if w_dict[key].dtype in (torch.float32, torch.float64, torch.float16):
                avg_dict[key] += (size / total_size) * w_dict[key]
            # 非浮点参数不参与加权平均，维持不变

    global_model.load_state_dict(avg_dict)
    return global_model