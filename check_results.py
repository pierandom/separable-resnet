import os
import torch
from separable_resnet import SeparableResNet
from train import get_data, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = SeparableResNet(num_classes=10, kernel_size=5, width_factor=4, depth_factor=3).to(device)

weights_path = os.path.join("trained-models", "CIFAR10", "separable-resnet4-3", "weights.pth")

net.load_state_dict(torch.load(weights_path))

_, test_dataloader = get_data(dataset_name="cifar10", batch_size=64)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

print(evaluate(net, criterion, test_dataloader, device))