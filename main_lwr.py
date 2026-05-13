import torch
import numpy as np
from torch.utils.data import DataLoader
from data.data_loader import get_cifar10, dirichlet_partition
from models.nets import SimpleCNN
from algorithms.fedavg import server_aggregate
from algorithms.fedlwr import client_update_fedlwr

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total

def main():
    num_clients = 50
    rounds = 30
    fraction = 0.2
    alpha = 0.5
    batch_size = 64
    local_epochs = 5
    lr = 0.01
    device = 'cpu'

    # FedLWR 超参数
    beta = 0.9
    eta = 0.1
    init_lambda = 0.01

    print("准备数据...")
    train_dataset, test_dataset = get_cifar10('./data')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha)
    client_sizes = [len(indices) for indices in client_indices]

    global_model = SimpleCNN()
    global_model.to(device)

    # 清除 FedLWR 缓存（避免上次运行的 lambda 残留）
    if hasattr(client_update_fedlwr, 'lambda_dict'):
        del client_update_fedlwr.lambda_dict

    print(f'开始 FedLWR 训练，共 {rounds} 轮')
    for r in range(rounds):
        num_selected = max(1, int(fraction * num_clients))
        selected_clients = np.random.choice(num_clients, num_selected, replace=False)

        client_weights = []
        selected_sizes = []
        lambda_records = []

        for client_id in selected_clients:
            local_model = SimpleCNN().to(device)
            local_model.load_state_dict(global_model.state_dict())

            w, lam = client_update_fedlwr(
                local_model, train_dataset, client_indices[client_id],
                lr=lr, local_epochs=local_epochs, batch_size=batch_size,
                device=device, beta=beta, eta=eta, init_lambda=init_lambda
            )
            client_weights.append(w)
            selected_sizes.append(client_sizes[client_id])
            lambda_records.append(lam)

        global_model = server_aggregate(global_model, client_weights, selected_sizes)

        if (r+1) % 5 == 0 or r == 0:
            acc = evaluate(global_model, test_loader, device)
            # 打印某层 lambda 观察变化
            try:
                lam_cls = lambda_records[0]['fc2.weight']
            except:
                lam_cls = 0
            print(f'第 {r+1:3d} 轮  准确率: {acc*100:.2f}%  λ(fc2): {lam_cls:.4f}')

    print('FedLWR 训练完成！')

if __name__ == '__main__':
    main()