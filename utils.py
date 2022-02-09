import torch
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
