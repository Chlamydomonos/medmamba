import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import random_split
import torch.optim as optim
from tqdm import tqdm

from med_mamba import VSSM as medmamba


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='通用模型训练脚本')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['resnet50', 'densenet169', 'medmamba'],
                        help='模型类型: resnet50, densenet169, medmamba')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='分类类别数 (默认: 6)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默认: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率 (默认: 0.0001)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='输出目录 (默认: ./results)')

    return parser.parse_args()


def create_model(model_name, num_classes):
    """根据模型名称创建模型"""
    if model_name == 'resnet50':
        net = models.resnet50()
    elif model_name == 'densenet169':
        net = models.densenet169()
    elif model_name == 'medmamba':
        net = medmamba(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    return net


def prepare_data(data_path, batch_size):
    """准备数据集，按80/20分割训练集和测试集"""
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        "val": transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_path, transform=data_transform["train"])

    # 计算训练集和验证集的大小（80/20分割）
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    # 随机分割数据集
    train_dataset, val_dataset_temp = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为验证集创建一个新的数据集，应用验证变换
    val_dataset = datasets.ImageFolder(root=data_path, transform=data_transform["val"])
    # 使用相同的索引
    val_dataset = torch.utils.data.Subset(val_dataset, val_dataset_temp.indices)

    # 保存类别索引
    class_to_idx = full_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())

    # 确保输出目录存在
    os.makedirs('./results', exist_ok=True)
    json_str = json.dumps(cla_dict, indent=4)
    with open('./results/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 创建数据加载器
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'使用 {nw} 个dataloader workers')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=nw
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=nw
    )

    return train_loader, val_loader, train_size, val_size


def train_epoch(net, train_loader, loss_function, optimizer, device, epoch, epochs):
    """训练一个epoch"""
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, file=sys.stdout, desc=f'训练 Epoch [{epoch+1}/{epochs}]')
    for step, (images, labels) in enumerate(train_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        train_bar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc


def validate(net, val_loader, loss_function, device):
    """验证模型"""
    net.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, file=sys.stdout, desc='验证')
        for images, labels in val_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = loss_function(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc


def main():
    # 解析参数
    args = parse_args()

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"模型: {args.model}")
    print(f"数据集路径: {args.data_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("-" * 50)

    # 准备数据
    print("准备数据集...")
    train_loader, val_loader, train_size, val_size = prepare_data(args.data_path, args.batch_size)
    print(f"训练集样本数: {train_size}, 验证集样本数: {val_size}")
    print(f"数据集按 80/20 比例分割")
    print("-" * 50)

    # 创建模型
    print(f"创建模型: {args.model}")
    net = create_model(args.model, args.num_classes)
    net.to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # 创建输出目录
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(model_output_dir, f'{args.model}_training_{timestamp}.log')

    # 训练
    best_acc = 0.0
    best_save_path = os.path.join(model_output_dir, f'{args.model}_best.pth')
    last_save_path = os.path.join(model_output_dir, f'{args.model}_last.pth')

    print(f"开始训练...")
    print(f"日志文件: {log_file}")
    print("-" * 50)

    # 创建日志文件并写入表头
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"数据集: {args.data_path}\n")
        f.write(f"训练集大小: {train_size}, 验证集大小: {val_size}\n")
        f.write(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}, 学习率: {args.lr}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Val Loss':<15}{'Val Acc':<15}{'Best':<10}\n")
        f.write("-" * 80 + "\n")

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            net, train_loader, loss_function, optimizer, device, epoch, args.epochs
        )

        # 验证
        val_loss, val_acc = validate(net, val_loader, loss_function, device)

        # 打印当前epoch的结果
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(net.state_dict(), best_save_path)

        # 输出到命令行
        print(f'\n[Epoch {epoch+1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
              f'{" [BEST]" if is_best else ""}')
        print("-" * 50)

        # 写入日志文件（只记录每个epoch的汇总信息）
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f'{epoch+1:<10}{train_loss:<15.4f}{train_acc:<15.4f}'
                   f'{val_loss:<15.4f}{val_acc:<15.4f}{"*" if is_best else "":<10}\n')

    # 保存最后的模型
    torch.save(net.state_dict(), last_save_path)

    # 写入训练完成信息
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("-" * 80 + "\n")
        f.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"最佳验证准确率: {best_acc:.4f}\n")
        f.write(f"最佳模型保存路径: {best_save_path}\n")
        f.write(f"最终模型保存路径: {last_save_path}\n")

    print(f'\n训练完成！')
    print(f'最佳验证准确率: {best_acc:.4f}')
    print(f'最佳模型保存至: {best_save_path}')
    print(f'最终模型保存至: {last_save_path}')
    print(f'详细日志保存至: {log_file}')


if __name__ == '__main__':
    main()
