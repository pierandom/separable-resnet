import torch
from torch import nn
from torch import Tensor


class Mean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, val: float):
        self.total += val
        self.count += 1

    def compute(self):
        return self.total / self.count


class Accuracy:
    def __init__(self) -> None:
        self.correct = 0
        self.count = 0

    def update(self, logits: Tensor, labels: Tensor):
        predicted_labels = logits.argmax(dim=1)
        self.correct += (predicted_labels == labels).sum().item()
        self.count += labels.size(0)

    def compute(self):
        return self.correct / self.count


def weight_decay(model, device):
    loss = torch.tensor(0, dtype=torch.float32, device=device)
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            loss = loss + m.weight.norm(2)
            n += 1
    return loss / n


def entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    eps = torch.finfo(logits.dtype).eps
    entropy = -torch.sum(
        probs * torch.log(torch.clamp(probs, min=eps)), dim=-1
    ) / torch.log(torch.tensor(logits.shape[-1]))
    preds = torch.argmax(logits, dim=-1)
    entropy = torch.where(preds == targets, entropy, 1.0)
    return entropy


class LossFn:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        label_smoothing: float,
        weight_decay_factor: float,
        use_entropy_weights: bool,
        entropy_with_grads: bool,
    ):
        self.model = model
        self.device = device
        self.label_smoothing = label_smoothing
        self.weight_decay_factor = weight_decay_factor
        self.use_entropy_weights = use_entropy_weights
        self.entropy_with_grads = entropy_with_grads
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction="none" if use_entropy_weights else "mean",
        )

    def __call__(self, logits, targets):
        if self.use_entropy_weights:
            if self.entropy_with_grads:
                weights = entropy(logits, targets)
            else:
                with torch.no_grad():
                    weights = entropy(logits, targets)
            loss = torch.mean(weights * self.criterion(logits, targets))
        else:
            loss = self.criterion(logits, targets)
        if self.model.training:
            return loss + self.weight_decay_factor * weight_decay(
                self.model, self.device
            )
        else:
            return loss
