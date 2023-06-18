import os
import json
import time
from datetime import timedelta
from argparse import ArgumentParser, Namespace
import wandb

import torch
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

from separable_resnet2 import SeparableResNet
from resnet import resnet32
from utils import Mean, Accuracy, LossFn


def parse_args():
    parser = ArgumentParser(description="PyTorch training script")
    parser.add_argument(
        "--model_name",
        choices=["separable_resnet", "resnet"],
        default="separable_resnet",
    )
    parser.add_argument("--resume_run_id", type=str, help="WandB run id to resume")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--net_width_factor", type=int, default=4)
    parser.add_argument("--net_depth_factor", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_max", type=float, default=1e-1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--lr_base_period", type=int, default=15)
    parser.add_argument("--lr_period_factor", type=int, default=2)
    parser.add_argument("--weight_decay_factor", type=float, default=1e-2)
    parser.add_argument("--use_entropy_weights", action="store_true", default=False)
    parser.add_argument("--entropy_with_grads", action="store_true", default=False)
    parser.add_argument("--clip_grad_max_norm", type=float, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")

    args = parser.parse_args()
    return args


def get_data(dataset_name, batch_size):
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.RandomErasing(),
        ]
    )

    train_loader = DataLoader(
        dataset(root=".datasets", train=True, transform=transforms, download=True),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset(root=".datasets", train=False, transform=T.ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def train_one_epoch(
    model,
    criterion,
    optimizer,
    scaler,
    data_loader,
    device,
    clip_grad_max_norm,
    temperature,
):
    model.train()
    epoch_loss = Mean()
    epoch_accuracy = Accuracy()

    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(image) / temperature
            loss = criterion(logits, target)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss.update(loss.item())
        epoch_accuracy.update(logits, target)

    stats = {"loss": epoch_loss.compute(), "accuracy": epoch_accuracy.compute()}
    return stats


def evaluate(model, criterion, data_loader, device):
    model.eval()
    val_loss = Mean()
    val_accuracy = Accuracy()

    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            logits = model(image)

            loss = criterion(logits, target)

            val_loss.update(loss.item())
            val_accuracy.update(logits, target)

    stats = {"val_loss": val_loss.compute(), "val_accuracy": val_accuracy.compute()}
    return stats


def main(args):
    config = {
        "model_name": args.model_name,
        "warmup_epochs": args.warmup_epochs,
        "lr_max": args.lr_max,
        "momentum": args.momentum,
        "lr_base_period": args.lr_base_period,
        "lr_period_factor": args.lr_period_factor,
        "batch_size": args.batch_size,
        "weight_decay_factor": args.weight_decay_factor,
        "clip_grad_max_norm": args.clip_grad_max_norm,
        "label_smoothing": args.label_smoothing,
        "dataset": args.dataset,
    }

    if args.model_name == "separable_resnet":
        config.update(
            {
                "net_width_factor": args.net_width_factor,
                "net_depth_factor": args.net_depth_factor,
                "kernel_size": args.kernel_size,
            }
        )

    if args.resume_run_id:
        wandb.init(project="separable-resnet", resume="must", id=args.resume_run_id)
    else:
        wandb.init(project="separable-resnet", config=config)

    config_path = os.path.join(wandb.run.dir, "config.json")
    checkpoint_path = os.path.join(wandb.run.dir, "checkpoint.pth")

    if args.resume_run_id:  # load resuming run config
        wandb.restore("config.json")
        with open(config_path) as config_file:
            config = json.load(config_file)
        args_dict = vars(args)
        args_dict.update(config)
        args = Namespace(**args_dict)
    else:  # save run config
        with open(config_path, "w") as config_file:
            json.dump(config, config_file)
        wandb.save(config_path, base_path=wandb.run.dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_test = get_data(args.dataset, args.batch_size)
    if args.model_name == "separable_resnet":
        model = SeparableResNet(
            num_classes=len(data_loader_train.dataset.classes),
            kernel_size=args.kernel_size,
            width_factor=args.net_width_factor,
            depth_factor=args.net_depth_factor,
        )
    elif args.model_name == "resnet":
        model = resnet32()
    model = torch.compile(model)
    model = model.to(device)
    scaler = GradScaler()
    criterion = LossFn(
        model,
        device,
        label_smoothing=args.label_smoothing,
        weight_decay_factor=args.weight_decay_factor,
        use_entropy_weights=args.use_entropy_weights,
        entropy_with_grads=args.entropy_with_grads,
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr_max, momentum=args.momentum
    )
    scheduler1 = lr_scheduler.ConstantLR(
        optimizer, factor=1, total_iters=args.warmup_epochs
    )
    scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.lr_base_period, T_mult=args.lr_period_factor
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler1, scheduler2], milestones=[args.warmup_epochs]
    )

    if wandb.run.resumed:
        wandb.restore(
            "checkpoint.pth"
        )  # returns io object with wrong encoding on windows
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_parameters:,}")

    if torch.cuda.is_available():
        print("Training on GPU ...")
    else:
        print("Training on CPU ...")

    tot_epochs = start_epoch + args.epochs
    epoch_fmt = f">{len(str(tot_epochs))}"
    start_time_training = time.time()

    for epoch in range(1 + start_epoch, 1 + tot_epochs):
        start_time_epoch = time.time()
        train_stats = train_one_epoch(
            model,
            criterion,
            optimizer,
            scaler,
            data_loader_train,
            device,
            args.clip_grad_max_norm,
            args.temperature,
        )
        scheduler.step()
        val_stats = evaluate(model, criterion, data_loader_test, device)
        epoch_time = int(time.time() - start_time_epoch)
        training_time = int(time.time() - start_time_training)

        wandb.log(train_stats, step=epoch)
        wandb.log(val_stats, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        # torch.save(checkpoint, checkpoint_path)
        # wandb.save(checkpoint_path, base_path=wandb.run.dir)

        print(
            f"Epoch: {epoch:{epoch_fmt}}/{tot_epochs} - "
            f"Time: {timedelta(seconds=epoch_time)}/{timedelta(seconds=training_time)} - "
            f"Train Loss: {train_stats['loss']:.6f} - Train Accuracy: {train_stats['accuracy']:.2%} - "
            f"Test Loss: {val_stats['val_loss']:.6f} - Test Accuracy: {val_stats['val_accuracy']:.2%}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
