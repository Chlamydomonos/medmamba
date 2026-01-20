import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from med_mamba import VSSM as medmamba


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train classification models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet50', 'densenet169', 'medmamba'],
                        help='Model type to train')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes in dataset')
    parser.add_argument('--output_dir', type=str, default='./trainer-results',
                        help='Directory to save results')
    parser.add_argument('--log_name', type=str, default=None,
                        help='Log file name (default: auto-generated)')
    return parser.parse_args()


def get_model(model_type, num_classes):
    """Create model based on type."""
    if model_type == 'resnet50':
        model = models.resnet50()
        # Modify the final layer for custom number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'densenet169':
        model = models.densenet169()
        # Modify the classifier for custom number of classes
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_type == 'medmamba':
        model = medmamba(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    # Parse arguments
    args = parse_args()

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)

    # Setup logging
    if args.log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"{args.model}_train_{timestamp}.log"
    else:
        log_name = args.log_name

    log_path = os.path.join(model_output_dir, log_name)
    log_file = open(log_path, "a")

    # Setup tensorboard
    tensorboard_dir = os.path.join(model_output_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    print(f"Training Configuration:")
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Output directory: {model_output_dir}")
    print(f"Device: {device}")
    print("-" * 50)

    # Data transforms
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load training dataset
    train_dataset = datasets.ImageFolder(root=args.data_path, transform=data_transform)
    train_num = len(train_dataset)

    # Save class indices
    class_to_idx = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())
    json_str = json.dumps(cla_dict, indent=4)
    class_indices_path = os.path.join(model_output_dir, 'class_indices.json')
    with open(class_indices_path, 'w') as json_file:
        json_file.write(json_str)

    print(f"Class indices saved to: {class_indices_path}")
    print(f"Classes: {cla_dict}")

    # Data loader settings
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw
    )

    print(f"Using {train_num} images for training.")

    # Create model
    net = get_model(args.model, args.num_classes)
    net.to(device)

    # Print model info
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 50)

    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # Training settings
    epochs = 200
    best_acc = 0.0
    best_save_path = os.path.join(model_output_dir, f'{args.model}_best.pth')
    last_save_path = os.path.join(model_output_dir, f'{args.model}_last.pth')
    train_steps = len(train_loader)

    print(f"Starting training for {epochs} epochs...")
    print("-" * 50)

    for epoch in range(epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss
            )

        # Calculate training accuracy
        net.eval()
        acc = 0.0
        with torch.no_grad():
            train_bar = tqdm(train_loader, desc="Evaluating")
            for train_data in train_bar:
                train_images, train_labels = train_data
                outputs = net(train_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, train_labels.to(device)).sum().item()

        train_accurate = acc / train_num
        avg_loss = running_loss / train_steps

        # Log to tensorboard
        writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', train_accurate, epoch + 1)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)

        epoch_log = '[epoch %d] train_loss: %.3f  train_accuracy: %.3f' % (
            epoch + 1, avg_loss, train_accurate)
        print(epoch_log)

        # Write epoch summary to log file
        log_file.write(epoch_log + '\n')
        log_file.flush()

        # Save best model
        if train_accurate > best_acc:
            best_acc = train_accurate
            torch.save(net.state_dict(), best_save_path)
            best_msg = f"Saved best model with accuracy: {best_acc:.3f}"
            print(best_msg)
            log_file.write(best_msg + '\n')
            log_file.flush()

    # Save last epoch model
    torch.save(net.state_dict(), last_save_path)

    print("-" * 50)
    print('Finished Training')
    print(f'Best training accuracy: {best_acc:.3f}')
    print(f'Best model saved to: {best_save_path}')
    print(f'Last model saved to: {last_save_path}')
    print(f'Tensorboard logs saved to: {tensorboard_dir}')

    # Write final summary to log file
    log_file.write("-" * 50 + '\n')
    log_file.write('Finished Training\n')
    log_file.write(f'Best training accuracy: {best_acc:.3f}\n')
    log_file.write(f'Best model saved to: {best_save_path}\n')
    log_file.write(f'Last model saved to: {last_save_path}\n')
    log_file.write(f'Tensorboard logs saved to: {tensorboard_dir}\n')
    log_file.close()

    writer.close()


if __name__ == '__main__':
    main()