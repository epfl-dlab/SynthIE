import torch
from typing import Union

from src.metrics.abstract import IEAbstractTorchMetric


class TSF1(IEAbstractTorchMetric):
    name = "triplet_set_f1"

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    @staticmethod
    def _compute(correct, predicted, target, use_tensor=False) -> Union[float, torch.Tensor]:
        if correct == 0 or predicted == 0 or target == 0:
            return torch.tensor(0).float() if use_tensor else 0.0

        correct = correct.float() if use_tensor else float(correct)
        precision = correct / predicted
        recall = correct / target
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute(self):
        if self.total_predicted == 0 or self.total_target == 0 or self.total_correct == 0:
            return torch.tensor(0).float()

        precision = self.total_correct.float() / self.total_predicted
        recall = self.total_correct.float() / self.total_target
        f1 = 2 * precision * recall / (precision + recall)

        return f1
