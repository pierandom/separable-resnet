import time
from datetime import timedelta

import mlflow
import torch
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import Config
from data_loaders import get_data_loaders
from resnet import resnet32
from separable_resnet import SeparableResNet
from utils import Accuracy, LossFn, Mean


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

    for image, target in tqdm(data_loader, leave=False, ncols=100):
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


def main(args: Config):
    mlflow.set_tracking_uri("http://0.0.0.0:8888")
    mlflow.set_experiment("separable-resnet")

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
        "temperature": args.temperature,
        "use_entropy_weights": args.use_entropy_weights,
        "entropy_with_grads": args.entropy_with_grads,
    }

    if args.model_name == "separable_resnet":
        config.update(
            {
                "net_width_factor": args.net_width_factor,
                "net_depth_factor": args.net_depth_factor,
                "kernel_size": args.kernel_size,
            }
        )

    mlflow.log_params(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader_train, data_loader_test = get_data_loaders(args.dataset, args.batch_size)
    if args.model_name == "separable_resnet":
        model = SeparableResNet(
            num_classes=len(data_loader_train.dataset.classes),
            kernel_size=args.kernel_size,
            width_factor=args.net_width_factor,
            depth_factor=args.net_depth_factor,
        )
    elif args.model_name == "resnet":
        model = resnet32()

    compiled_model = torch.compile(model)
    compiled_model = compiled_model.to(device)

    scaler = GradScaler()
    criterion = LossFn(
        compiled_model,
        device,
        label_smoothing=args.label_smoothing,
        weight_decay_factor=args.weight_decay_factor,
        use_entropy_weights=args.use_entropy_weights,
        entropy_with_grads=args.entropy_with_grads,
    )
    optimizer = torch.optim.SGD(
        compiled_model.parameters(), lr=args.lr_max, momentum=args.momentum
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

    start_epoch = 0

    num_parameters = sum(p.numel() for p in compiled_model.parameters())
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
            compiled_model,
            criterion,
            optimizer,
            scaler,
            data_loader_train,
            device,
            args.clip_grad_max_norm,
            args.temperature,
        )
        scheduler.step()
        val_stats = evaluate(compiled_model, criterion, data_loader_test, device)
        epoch_time = int(time.time() - start_time_epoch)
        training_time = int(time.time() - start_time_training)

        mlflow.log_metrics(train_stats, step=epoch)
        mlflow.log_metrics(val_stats, step=epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": compiled_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        mlflow.pytorch.log_state_dict(checkpoint, artifact_path="checkpoint")

        print(
            f"Epoch: {epoch:{epoch_fmt}}/{tot_epochs} - "
            f"Time: {timedelta(seconds=epoch_time)}/{timedelta(seconds=training_time)} - "
            f"ETA: {timedelta(seconds=epoch_time * (tot_epochs - epoch))} - "
            f"Train Loss: {train_stats['loss']:.6f} - Train Accuracy: {train_stats['accuracy']:.2%} - "
            f"Test Loss: {val_stats['val_loss']:.6f} - Test Accuracy: {val_stats['val_accuracy']:.2%}"
        )

    state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in compiled_model.state_dict().items()
    }
    model.load_state_dict(state_dict)
    mlflow.pytorch.log_model(
        model, artifact_path="model", registered_model_name="CIFAR10-Classifier"
    )


if __name__ == "__main__":
    args = Config().parse_args()
    main(args)
