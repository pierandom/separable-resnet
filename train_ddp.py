import os

import mlflow
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler

from config import Config
from data_loaders import get_data_loaders
from separable_resnet import SeparableResNet
from trainer_ddp import Trainer


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, config):
    setup(rank, world_size)
    try:
        _main(rank, config)
    except KeyboardInterrupt:
        print(f"Process {rank} interrupted!")
    finally:
        cleanup()


def _main(rank: int, args: Config):
    if rank == 0:
        mlflow.set_tracking_uri("http://0.0.0.0:8888")
        mlflow.set_experiment("separable-resnet")
        mlflow.log_params(args.as_dict())

    device = torch.device(rank)

    train_data_loader, val_data_loader = get_data_loaders(
        args.dataset, args.batch_size, is_distributed=True
    )
    model = SeparableResNet(
        num_classes=len(train_data_loader.dataset.classes),
        kernel_size=args.kernel_size,
        width_factor=args.net_width_factor,
        depth_factor=args.net_depth_factor,
    )

    model = model.to(device)
    compiled_model = torch.compile(model)
    ddp_model = DDP(compiled_model, device_ids=[rank])

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        ddp_model.parameters(), lr=args.lr_max, momentum=args.momentum
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

    num_parameters = sum(p.numel() for p in ddp_model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_parameters:,}")

    trainer = Trainer(
        rank,
        ddp_model,
        loss_fn,
        optimizer,
        scheduler,
        train_data_loader,
        val_data_loader,
        args,
    )
    trainer.train()


if __name__ == "__main__":
    config = Config().parse_args()
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(main, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
