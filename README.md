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

## Pretrained Model (Recommended)

A trained model checkpoint is available here:

👉 https://drive.google.com/file/d/1wj--vjtkpUkQB63r7a4SM-hTVuZnVV0i/view?usp=sharing

### To use it:

1. Download the file  
2. Place it in the root directory of this project  
3. Run:

```bash
python train.py eval --prompt "Once upon a time" --load model_best.pth --token-count 200

---

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt