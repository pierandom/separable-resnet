import wandb
import time

import torch
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from separable_resnet import SeparableResNet
from utils import Mean, Accuracy

config = {
    "net_width_factor": 1,
    "net_depth_factor": 1,
    "epochs": 100,
    "warmup_epochs": 5,
    "lr_max": 1e-1,
    "lr_min": 1e-5,
    "momentum": 0.9,
    "lr_period": 24,
    "batch_size": 64,
    "label_smoothing": 0.1,
    "dataset": "CIFAR10",
    "num_classes": 10
}

TRACK = False
if TRACK:
    wandb.init(
        project="separable-resnet",
        entity="pierand",
        config=config
    )




def get_data():
    transforms = T.Compose([
        T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.RandomErasing()
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(
            root=".datasets",
            train=True,
            transform=transforms,
            download=True
        ),
        batch_size=config["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(
            root=".datasets",
            train=False,
            transform=T.ToTensor(),
            download=True
        ),
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, test_loader



def train_one_epoch(model, criterion, optimizer, data_loader, device):
    model.train()
    epoch_loss = Mean()
    epoch_accuracy = Accuracy()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        logits = model(image)
        loss = criterion(logits, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.update(loss.item())
        epoch_accuracy.update(logits, target)
    
    stats = {
        "loss": epoch_loss.compute(),
        "accuracy": epoch_accuracy.compute()
    }
    return stats



def evaluate(model, criterion, data_loader, device):
    model.eval()
    val_loss = Mean()
    val_accuracy = Accuracy()

    with torch.inference_mode():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)

            loss = criterion(logits, target)

            val_loss.update(loss.item())
            val_accuracy.update(logits, target)
        
        stats = {
            "val_loss": val_loss.compute(),
            "val_accuracy": val_accuracy.compute()
        }
    return stats
    


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_test = get_data()
    model = SeparableResNet(
        num_classes=config["num_classes"],
        width_factor=config["net_width_factor"],
        depth_factor=config["net_depth_factor"]
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr_max"], momentum=config["momentum"])
    scheduler1 = lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config["warmup_epochs"])
    scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["lr_period"], eta_min=config["lr_min"])
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[config["warmup_epochs"]])

    EPOCHS = config["epochs"]
    epoch_fmt = f">{len(str(EPOCHS))}"

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_parameters:,}")

    if torch.cuda.is_available():
        print("Training on GPU")
    else:
        print("Training on CPU")

    for epoch in range(1, 1+EPOCHS):
        start_time = time.time()
        train_stats = train_one_epoch(model, criterion, optimizer, data_loader_train, device)
        scheduler.step()
        val_stats = evaluate(model, criterion, data_loader_test, device)
        epoch_time = time.time() - start_time

        if TRACK:
            wandb.log(train_stats, step=epoch)
            wandb.log(val_stats, step=epoch)
        
        print(
            f"Epoch: {epoch:{epoch_fmt}}/{EPOCHS} - Time: {int(epoch_time)}s - "
            f"Loss: {train_stats['loss']:.6f} - Accuracy: {train_stats['accuracy']:.2%} - "
            f"Test Loss: {val_stats['val_loss']:.6f} - Test Accuracy: {val_stats['val_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()