import argparse
import os
import random
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from transformers import BertForSequenceClassification
from tqdm import tqdm

import src.Log
from src.dataset.dataloader import dataloader
from src.model.Bert import Bert


def _extract_logits(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def _safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


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


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device, logger) -> None:
    checkpoint = _safe_torch_load(checkpoint_path, device)
    state_dict = _as_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = len(incompatible.missing_keys)
    unexpected = len(incompatible.unexpected_keys)
    logger.log_info(
        f"Loaded checkpoint: {checkpoint_path} (missing={missing}, unexpected={unexpected})"
    )


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = _extract_logits(model(input_ids=input_ids, attention_mask=attention_mask))
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train(config):
    training = config["training"]
    learning = config["learning"]

    model_name = training["model-name"]
    if model_name != "Bert":
        raise ValueError(f"Bert-only mode: unsupported model-name '{model_name}'")

    data_name = training["data-name"]
    epochs = training["epochs"]
    n_block = training.get("n-block", 12)
    num_sample = training["num-sample"]
    eval_every_epoch = training.get("eval-every-epoch", True)
    save_path = training.get("save-path", "bert_pretrain.pth")
    load_path = training.get("load-path")
    use_pretrained = training.get("use-pretrained", True)
    pretrained_model_name = training.get("pretrained-model-name", "bert-base-cased")

    batch_size = learning["batch-size"]
    lr = learning["learning-rate"]
    weight_decay = learning["weight-decay"]
    clip_grad_norm = learning.get("clip-grad-norm", 0.0)

    set_seed(training.get("random-seed"))

    log_path = config.get("log_path", ".")
    os.makedirs(log_path, exist_ok=True)
    logger = src.Log.Logger(os.path.join(log_path, "app.log"), config.get("debug_mode", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_info(f"Using device: {device}")

    if not load_path and save_path and os.path.exists(save_path):
        load_path = save_path
        logger.log_info(f"Using existing checkpoint as load-path: {load_path}")

    train_loader = dataloader(
        model_name=model_name,
        data_name=data_name,
        batch_size=batch_size,
        distribution=num_sample,
        train=True,
    )
    test_loader = dataloader(model_name=model_name, data_name=data_name, train=False)

    if use_pretrained:
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=4).to(device)
        logger.log_info(f"Loaded pretrained model: {pretrained_model_name}")
        if load_path:
            if os.path.exists(load_path):
                _load_checkpoint(model, load_path, device, logger)
            else:
                logger.log_warning(f"Checkpoint not found: {load_path}")
    else:
        model = Bert(layer_id=0, n_block=n_block).to(device)
        if load_path:
            if os.path.exists(load_path):
                _load_checkpoint(model, load_path, device, logger)
            else:
                logger.log_warning(f"Checkpoint not found: {load_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.log_info(f"Starting training for {epochs} epoch(s)")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = _extract_logits(model(input_ids=input_ids, attention_mask=attention_mask))
            loss = criterion(logits, labels)
            loss.backward()

            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            optimizer.step()

            running_loss += loss.item()
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_samples += labels.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = running_correct / max(running_samples, 1)
        logger.log_info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

        if eval_every_epoch:
            val_loss, val_acc = evaluate(model, test_loader, criterion, device)
            logger.log_info(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        logger.log_info(f"Saved model to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Single-model LLM training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    train(config)


if __name__ == "__main__":
    main()
