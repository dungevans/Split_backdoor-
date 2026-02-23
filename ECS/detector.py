from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from Detector import ZPerLabelRuleDetectorOnline


@dataclass
class DetectorConfig:
    n_classes: int = 4
    per_client_limit: int = 200
    store_per_batch: int = 64
    baseline_cap: int = 512
    min_client_points: int = 64
    min_baseline_points: int = 128
    thr_energy: float = 2.5
    thr_centroid: float = 2.3
    consecutive: int = 2
    seed: int = 1


class ECSDetector:
    """Wrapper around the online ECS z-rule detector for text classification logits."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.impl = ZPerLabelRuleDetectorOnline(
            n_classes=config.n_classes,
            per_client_limit=config.per_client_limit,
            store_per_batch=config.store_per_batch,
            baseline_cap=config.baseline_cap,
            min_client_points=config.min_client_points,
            min_baseline_points=config.min_baseline_points,
            thr_energy=config.thr_energy,
            thr_centroid=config.thr_centroid,
            consecutive=config.consecutive,
            seed=config.seed,
        )

    def update(self, logits: torch.Tensor, labels: torch.Tensor, client_id: str) -> Tuple[Dict, Dict]:
        return self.impl.update_batch(z=logits, y=labels, client_id=client_id)
