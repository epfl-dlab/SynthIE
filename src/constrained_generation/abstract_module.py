from abc import ABC
from typing import Optional, Callable, Iterable

import torch


class ConstrainedGenerationModule(ABC):
    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        raise NotImplementedError()
