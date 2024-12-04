from typing import List
import torch

def accuracy(output: torch.Tensor, target: torch.Tensor, top_k=(1,)) -> List[float]:
    max_k = max(top_k)
    batch_size = target.size(0)

    _, predict = output.topk(max_k, 1, True, True)
    predict = predict.t()
    correct = predict.eq(target.view(1, -1).expand_as(predict))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

class Metrics:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float):
        self.sum += val
        self.count += 1

    def update_with_weight(self, val: float, count: int):
        self.sum += val*count
        self.count += count

    def reset(self):
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        if self.count == 0:
            return -1
        return self.sum / self.count
