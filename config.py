from typing import Literal
from tap import Tap

class Config(Tap):
    model_name: Literal["separable_resnet", "resnet"] = "separable_resnet"
    net_width_factor: int = 4
    net_depth_factor: int = 3
    kernel_size: int = 5
    epochs: int = 60
    warmup_epochs: int = 15
    lr_max: float = 0.1
    momentum: float = 0.9
    lr_base_period: int = 15
    lr_period_factor: int = 2
    batch_size: int = 64
    weight_decay_factor: float = 0.05
    clip_grad_max_norm: float = 1
    label_smoothing: float = 0.1
    dataset: Literal["cifar10"] = "cifar10"
    temperature: float = 1
    use_entropy_weights: bool = False
    entropy_with_grads: bool = False
    grad_accumulation_steps: int = 16

