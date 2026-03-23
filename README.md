# My GPT Language Model

This repository contains a simple GPT-style language model implemented in PyTorch.

## Files

- `train.py` – train and evaluate the model
- `finetune.py` – optional finetuning on additional datasets
- `util.py` – helper dataset and tokenizer utilities
- `loss.py` – custom loss functions
- `metrics.py` – custom metrics
- `input.txt` – dataset file (replace with your own text)
- `requirements.txt` – Python dependencies

> Model checkpoint files (`.pth`) are **not included**. You can train your own model using the instructions below.

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt