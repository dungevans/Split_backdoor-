# LLM Training (No Split Learning)

This project now runs as a single-process LLM training pipeline using a full BERT model.
RabbitMQ, server-client split, and split learning orchestration are no longer required for the main run path.

## Setup
```commandline
source sl/bin/activate
pip install -r requirements.txt
```

## Configuration
Main config is in `config.yaml`:

```yaml
name: LLM

training:
  model-name: Bert
  data-name: EMOTION
  n-block: 12
  epochs: 1
  num-sample: 500
  random-seed: 1
  eval-every-epoch: true
  load-path: null
  save-path: bert_pretrain.pth

learning:
  learning-rate: 0.00001
  weight-decay: 0.01
  batch-size: 2
  clip-grad-norm: 0.0

log_path: .
debug_mode: true
```

## Run
```commandline
python server.py --config config.yaml
```

Or:
```commandline
bash run.sh
```

## Output
- Training/evaluation logs are written to `app.log`.
- Model checkpoint is saved to `training.save-path` (default: `bert_pretrain.pth`).
