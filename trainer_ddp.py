import time
from datetime import timedelta
from typing import Callable, Dict

import mlflow
import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from utils import Accuracy, Mean, weight_decay


class Trainer:
    def __init__(
        self,
        rank: int,
        model: DDP,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        config: Config,
    ):
        self.rank = rank
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.config = config

    def train(self):
        epoch_fmt = f">{len(str(self.config.epochs))}"
        start_time_training = time.time()

        for epoch in range(1, 1 + self.config.epochs):
            start_time_epoch = time.time()

            stats = self.train_step(epoch)
            stats |= self.evaluate()

            epoch_time = int(time.time() - start_time_epoch)
            training_time = int(time.time() - start_time_training)

            if self.rank == 0:
                mlflow.log_metrics(stats, step=epoch)

                print(
                    f"Epoch: {epoch:{epoch_fmt}}/{self.config.epochs} - "
                    f"Time: {timedelta(seconds=epoch_time)}/{timedelta(seconds=training_time)} - "
                    f"ETA: {timedelta(seconds=epoch_time * (self.config.epochs - epoch))} - "
                    f"Train Loss: {stats['loss']:.6f} - Train Accuracy: {stats['accuracy']:.2%} - "
                    f"Test Loss: {stats['val_loss']:.6f} - Test Accuracy: {stats['val_accuracy']:.2%}"
                )

    def train_step(self, epoch) -> Dict:
        self.model.train()

        train_loss = Mean()
        accuracy = Accuracy()

        self.train_data_loader.sampler.set_epoch(epoch)
        train_data_loader = self.train_data_loader
        if self.rank == 0:
            train_data_loader = tqdm(train_data_loader, leave=False, ncols=100)

        scaler = GradScaler()
        idx = 0
        for image, target in train_data_loader:
            idx += 1
            image, target = image.to(self.rank), target.to(self.rank)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(image)
                loss = self.loss_fn(
                    logits, target
                ) + self.config.weight_decay_factor * weight_decay(
                    self.model, self.rank
                )
                scaler.scale(loss).backward()

            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_grad_max_norm
            )

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            train_loss.update(loss.item())
            accuracy.update(logits, target)

        self.lr_scheduler.step()
        return {"loss": train_loss.compute(), "accuracy": accuracy.compute()}

    @torch.no_grad()
    def evaluate(self) -> Dict:
        self.model.eval()
        eval_loss = Mean()
        accuracy = Accuracy()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for image, target in self.val_data_loader:
                image, target = image.to(self.rank), target.to(self.rank)
                logits = self.model(image)

                loss = self.loss_fn(logits, target)
                eval_loss.update(loss.item())
                accuracy.update(logits, target)

        return {"val_loss": eval_loss.compute(), "val_accuracy": accuracy.compute()}
