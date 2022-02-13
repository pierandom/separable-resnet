import os
import json
import wandb
from tqdm import tqdm

import torch
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from separable_resnet import SeparableResNet
from utils import Mean, Accuracy, weight_decay

config = {
    "net_width_factor": 4,
    "net_depth_factor": 3,
    "kernel_size": 5,
    "epochs": 480,
    "warmup_epochs": 15,
    "lr_max": 1e-1,
    "lr_min": 1e-5,
    "momentum": 0.9,
    "lr_base_period": 15,
    "lr_period_factor": 2,
    "batch_size": 64,
    "weight_decay_factor": 1e-2,
    "clip_grad_max_norm": 1,
    "label_smoothing": 0.1,
    "dataset": "CIFAR10",
    "num_classes": 10
}

TRACK = True
if TRACK:
    wandb.init(
        project="separable-resnet",
        entity="pierand",
        config=config
    )


# Save only one run for different datasets and depth/width factors
SAVE = True
if SAVE:
    print("Saving model weights at the end of training")
    save_path = os.path.join(
        "pretrained", f"{config['dataset']}",
        f"separable-resnet{config['net_width_factor']}-{config['net_depth_factor']}"
    )
    # This will overwrite existing saved weights
    os.makedirs(save_path, exist_ok=True)
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)



def get_data():
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
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



def train_one_epoch(model, criterion, optimizer, data_loader, device, config, tqdm_desc: str = None):
    model.train()
    epoch_loss = Mean()
    epoch_accuracy = Accuracy()
    tqdm_desc += " - Training  "

    with tqdm(data_loader, desc=tqdm_desc, unit="batch", bar_format="{l_bar}{bar:30}{r_bar}", colour="blue") as pbar:
        for image, target in pbar:
            image, target = image.to(device), target.to(device)
            logits = model(image)
            loss = criterion(logits, target) + config["weight_decay_factor"]*weight_decay(model, device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["clip_grad_max_norm"])
            optimizer.step()

            epoch_loss.update(loss.item())
            epoch_accuracy.update(logits, target)
        
            pbar_stats = {
                "loss": f"{epoch_loss.compute():.6f}",
                "accuracy": f"{epoch_accuracy.compute():.2%}"
            }
            pbar.set_postfix(pbar_stats)
    
    stats = {
        "loss": epoch_loss.compute(),
        "accuracy": epoch_accuracy.compute()
    }
    return stats



def evaluate(model, criterion, data_loader, device, tqdm_desc):
    model.eval()
    val_loss = Mean()
    val_accuracy = Accuracy()
    tqdm_desc += " - Validation"

    with torch.inference_mode():
        with tqdm(data_loader, desc=tqdm_desc, unit="batch", bar_format="{l_bar}{bar:30}{r_bar}", colour="yellow") as pbar:
            for image, target in pbar:
                image, target = image.to(device), target.to(device)
                logits = model(image)

                loss = criterion(logits, target)

                val_loss.update(loss.item())
                val_accuracy.update(logits, target)
            
                pbar_stats = {
                    "loss": f"{val_loss.compute():.6f}",
                    "accuracy": f"{val_accuracy.compute():.2%}"
                }
                pbar.set_postfix(pbar_stats)

    stats = {
        "val_loss": val_loss.compute(),
        "val_accuracy": val_accuracy.compute()
    }      
    return stats
    


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_test = get_data()
    model = SeparableResNet(
        num_classes=config["num_classes"],
        kernel_size=config["kernel_size"],
        width_factor=config["net_width_factor"],
        depth_factor=config["net_depth_factor"]
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr_max"], momentum=config["momentum"])
    scheduler1 = lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=config["warmup_epochs"])
    scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["lr_base_period"],
        T_mult=config["lr_period_factor"],
        eta_min=config["lr_min"]
    )
    scheduler = lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[config["warmup_epochs"]])

    EPOCHS = config["epochs"]
    epoch_fmt = f">{len(str(EPOCHS))}"

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_parameters:,}")

    if torch.cuda.is_available():
        print("Training on GPU")
    else:
        print("Training on CPU")

    for epoch in range(1, 1+EPOCHS):
        tqdm_desc = f"Epoch {epoch:{epoch_fmt}}/{EPOCHS}"
        train_stats = train_one_epoch(model, criterion, optimizer, data_loader_train, device, config, tqdm_desc)
        scheduler.step()
        val_stats = evaluate(model, criterion, data_loader_test, device, tqdm_desc)

        if TRACK:
            wandb.log(train_stats, step=epoch)
            wandb.log(val_stats, step=epoch)
    
    if SAVE:
        weights_path = os.path.join(save_path, "weights.pth")
        torch.save(model.state_dict(), weights_path)



if __name__ == "__main__":
    main(config)