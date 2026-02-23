import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


@dataclass
class PoisonConfig:
    poison_rate: float = 0.1
    trigger_text: str = "cf"
    label_mapping: Optional[Dict[int, int]] = None
    max_length: int = 128
    seed: int = 1


def parse_label_mapping(mapping_str: str) -> Dict[int, int]:
    """Parse string like '0:1,1:2' into {0: 1, 1: 2}."""
    mapping: Dict[int, int] = {}
    if not mapping_str:
        return mapping
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src, tgt = pair.split(":")
        mapping[int(src)] = int(tgt)
    return mapping


def _inject_trigger(text: str, trigger_text: str) -> str:
    return f"{trigger_text} {text}" if trigger_text else text


def _poison_indices(labels: np.ndarray, mapping: Dict[int, int], poison_rate: float, seed: int) -> set:
    if not mapping:
        return set()
    src_labels = np.array(list(mapping.keys()))
    candidates = np.where(np.isin(labels, src_labels))[0]
    if len(candidates) == 0:
        return set()

    rng = np.random.default_rng(seed)
    num_poison = int(round(poison_rate * len(candidates)))
    if num_poison <= 0:
        return set()

    chosen = rng.choice(candidates, size=min(num_poison, len(candidates)), replace=False)
    return set(map(int, chosen))


class PoisonedEmotionDataset(Dataset):
    """
    AG News-based text dataset with trigger poisoning.

    This repository names it EMOTION, but data source is AG News (4 classes).
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizer,
        poison_config: PoisonConfig,
        poison_index_set: Optional[set] = None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.cfg = poison_config
        self.poison_index_set = poison_index_set or set()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        original_label = int(self.labels[idx])
        is_poisoned = idx in self.poison_index_set

        if is_poisoned:
            text = _inject_trigger(text, self.cfg.trigger_text)
            label = int(self.cfg.label_mapping.get(original_label, original_label))
        else:
            label = original_label

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
            "original_labels": torch.tensor(original_label, dtype=torch.long),
            "is_poisoned": torch.tensor(1 if is_poisoned else 0, dtype=torch.long),
        }


def load_emotion_raw(split: str, cache_dir: str = "./hf_cache") -> Tuple[List[str], List[int]]:
    dataset = load_dataset("ag_news", cache_dir=cache_dir, download_mode="reuse_dataset_if_exists")
    subset = dataset[split]
    return list(subset["text"]), list(subset["label"])


def build_poisoned_emotion_dataset(
    split: str = "train",
    tokenizer_name: str = "bert-base-cased",
    poison_rate: float = 0.1,
    trigger_text: str = "cf",
    label_mapping: Optional[Dict[int, int]] = None,
    max_length: int = 128,
    seed: int = 1,
    cache_dir: str = "./hf_cache",
) -> PoisonedEmotionDataset:
    random.seed(seed)
    np.random.seed(seed)

    mapping = label_mapping or {0: 1}
    cfg = PoisonConfig(
        poison_rate=poison_rate,
        trigger_text=trigger_text,
        label_mapping=mapping,
        max_length=max_length,
        seed=seed,
    )

    texts, labels = load_emotion_raw(split=split, cache_dir=cache_dir)
    labels_np = np.asarray(labels, dtype=np.int64)
    poisoned_set = _poison_indices(labels_np, mapping, poison_rate, seed)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    return PoisonedEmotionDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        poison_config=cfg,
        poison_index_set=poisoned_set,
    )


def create_poisoned_dataloader(
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
    tokenizer_name: str = "bert-base-cased",
    poison_rate: float = 0.1,
    trigger_text: str = "cf",
    label_mapping: Optional[Dict[int, int]] = None,
    max_length: int = 128,
    seed: int = 1,
    cache_dir: str = "./hf_cache",
) -> DataLoader:
    ds = build_poisoned_emotion_dataset(
        split=split,
        tokenizer_name=tokenizer_name,
        poison_rate=poison_rate,
        trigger_text=trigger_text,
        label_mapping=label_mapping,
        max_length=max_length,
        seed=seed,
        cache_dir=cache_dir,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
