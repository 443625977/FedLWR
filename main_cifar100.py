import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from data.data_loader import get_cifar100, dirichlet_partition
from models.nets import MobileNetV2Cifar
from algorithms.fedavg import client_update, server_aggregate

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
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    num_clients = 50
    rounds = 100
    fraction = 0.2
    alpha = 0.1
    batch_size = 64
    local_epochs = 5
    lr = 0.01
    device = 'cpu'

    print("准备数据...")
    train_dataset, test_dataset = get_cifar100('./data')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha)
    client_sizes = [len(indices) for indices in client_indices]

    global_model = MobileNetV2Cifar(num_classes=100).to(device)

    print(f'开始训练，共 {rounds} 轮')
    for r in range(rounds):
        num_selected = max(1, int(fraction * num_clients))
        selected_clients = np.random.choice(num_clients, num_selected, replace=False)

        client_weights = []
        selected_sizes = []
        for client_id in selected_clients:
            local_model = MobileNetV2Cifar(num_classes=100).to(device)
            local_model.load_state_dict(global_model.state_dict())
            w = client_update(local_model, train_dataset,
                              client_indices[client_id],
                              lr=lr, local_epochs=local_epochs,
                              batch_size=batch_size, device=device)
            client_weights.append(w)
            selected_sizes.append(client_sizes[client_id])

        global_model = server_aggregate(global_model, client_weights, selected_sizes)

        if (r+1) % 5 == 0 or r == 0:
            acc = evaluate(global_model, test_loader, device)
            print(f'第 {r+1:3d} 轮  测试准确率: {acc*100:.2f}%')

    print('训练完成！')

if __name__ == '__main__':
    main()