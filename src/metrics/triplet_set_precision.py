import torch
from typing import Union

from src.metrics.abstract import IEAbstractTorchMetric


class TSPrecision(IEAbstractTorchMetric):
    name = "triplet_set_precision"

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    @staticmethod
    def _compute(correct, predicted, target, use_tensor=False) -> Union[float, torch.Tensor]:
        if predicted == 0:
            return torch.tensor(0).float() if use_tensor else 0.0

        correct = correct.float() if use_tensor else float(correct)
        precision = correct / predicted

        return precision

    def compute(self):
        if self.total_predicted == 0:
            return torch.tensor(0).float()

        return self.total_correct.float() / self.total_predicted
