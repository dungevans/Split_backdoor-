import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from data_poision import build_poisoned_emotion_dataset, parse_label_mapping
from detector import ECSDetector, DetectorConfig


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint format is not supported")

    if any(not isinstance(k, str) for k in checkpoint.keys()):
        raise TypeError("Checkpoint keys must be strings")

    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in checkpoint.items()
        }

    return checkpoint


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate_clean_accuracy(model: BertForSequenceClassification, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = _move_to_device(batch, device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        pred = logits.argmax(dim=1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return 0.0 if total == 0 else correct / total


@torch.no_grad()
def evaluate_asr(
    model: BertForSequenceClassification,
    loader: DataLoader,
    label_mapping: Dict[int, int],
    device: torch.device,
) -> float:
    """ASR over poisoned-and-flipped samples only."""
    model.eval()
    success = 0
    total = 0

    for batch in loader:
        batch = _move_to_device(batch, device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        pred = logits.argmax(dim=1)

        original = batch["original_labels"]
        poisoned = batch["is_poisoned"].bool()
        flipped = torch.zeros_like(poisoned)
        for src, tgt in label_mapping.items():
            flipped = flipped | ((original == src) & (tgt != src))

        mask = poisoned & flipped
        if mask.any():
            success += (pred[mask] == batch["labels"][mask]).sum().item()
            total += mask.sum().item()

    return 0.0 if total == 0 else success / total


def _next_batch(iterator: Iterable, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def run_detector_simulation(
    model: BertForSequenceClassification,
    benign_loader: DataLoader,
    poisoned_loader: DataLoader,
    detector: ECSDetector,
    num_clients: int,
    attacker_client: int,
    rounds: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    benign_iters = {cid: iter(benign_loader) for cid in range(num_clients) if cid != attacker_client}
    attack_iter = iter(poisoned_loader)

    flagged_count = defaultdict(int)
    flagged_labels = defaultdict(set)

    for _ in range(rounds):
        for cid in range(num_clients):
            if cid == attacker_client:
                batch, attack_iter = _next_batch(attack_iter, poisoned_loader)
            else:
                batch, benign_iters[cid] = _next_batch(benign_iters[cid], benign_loader)

            batch = _move_to_device(batch, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

            summary, _ = detector.update(logits=logits, labels=batch["labels"], client_id=f"client_{cid}")
            if summary["any_label_flagged"]:
                flagged_count[cid] += 1
                for lb in summary["flagged_labels"]:
                    flagged_labels[cid].add(int(lb))

    detected_clients = sorted([cid for cid, n in flagged_count.items() if n > 0])
    return {
        "detected_clients": detected_clients,
        "attacker_client": attacker_client,
        "attacker_detected": attacker_client in detected_clients,
        "flagged_rounds": {f"client_{cid}": int(n) for cid, n in flagged_count.items()},
        "flagged_labels": {f"client_{cid}": sorted(list(v)) for cid, v in flagged_labels.items()},
    }


def _build_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    parser = argparse.ArgumentParser(description="ECS validation for poisoned Emotion data")
    #parser.add_argument("--checkpoint", type=str, default="bert_pretrain.pth")
    parser.add_argument ("--checkpoint", type = str, default ="bert_pretrain.pth" )
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased")
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--poison-rate", type=float, default=0.1)
    parser.add_argument("--trigger-text", type=str, default="cf")
    parser.add_argument("--label-mapping", type=str, default="0:1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-clients", type=int, default=8)
    parser.add_argument("--attacker-client", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--cache-dir", type=str, default="./hf_cache")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    label_mapping = parse_label_mapping(args.label_mapping)
    if not label_mapping:
        raise ValueError("label_mapping is empty. Example: --label-mapping '0:1'")

    device = torch.device(args.device)

    model = BertForSequenceClassification.from_pretrained(args.tokenizer, num_labels=args.num_labels)
    ckpt = _safe_torch_load(args.checkpoint)
    state_dict = _as_state_dict(ckpt)
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint: {args.checkpoint} "
        f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})"
    )
    model.to(device)

    clean_test_ds = build_poisoned_emotion_dataset(
        split="test",
        tokenizer_name=args.tokenizer,
        poison_rate=0.0,
        trigger_text=args.trigger_text,
        label_mapping=label_mapping,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    poisoned_test_ds = build_poisoned_emotion_dataset(
        split="test",
        tokenizer_name=args.tokenizer,
        poison_rate=1.0,
        trigger_text=args.trigger_text,
        label_mapping=label_mapping,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    benign_train_ds = build_poisoned_emotion_dataset(
        split="train",
        tokenizer_name=args.tokenizer,
        poison_rate=0.0,
        trigger_text=args.trigger_text,
        label_mapping=label_mapping,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    attack_train_ds = build_poisoned_emotion_dataset(
        split="train",
        tokenizer_name=args.tokenizer,
        poison_rate=args.poison_rate,
        trigger_text=args.trigger_text,
        label_mapping=label_mapping,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    clean_loader = _build_loader(clean_test_ds, args.batch_size, shuffle=False)
    poisoned_loader = _build_loader(poisoned_test_ds, args.batch_size, shuffle=False)
    benign_train_loader = _build_loader(benign_train_ds, args.batch_size, shuffle=True)
    attack_train_loader = _build_loader(attack_train_ds, args.batch_size, shuffle=True)

    detector = ECSDetector(
        DetectorConfig(
            n_classes=args.num_labels,
            seed=args.seed,
        )
    )

    clean_acc = evaluate_clean_accuracy(model, clean_loader, device)
    asr = evaluate_asr(model, poisoned_loader, label_mapping, device)
    detector_report = run_detector_simulation(
        model=model,
        benign_loader=benign_train_loader,
        poisoned_loader=attack_train_loader,
        detector=detector,
        num_clients=args.num_clients,
        attacker_client=args.attacker_client,
        rounds=args.rounds,
        device=device,
    )

    print("=== ECS Validation Report ===")
    print(f"clean_accuracy: {clean_acc * 100:.2f}%")
    print(f"attack_success_rate: {asr * 100:.2f}%")
    print(f"detector_report: {detector_report}")


if __name__ == "__main__":
    main()
