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