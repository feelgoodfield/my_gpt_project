#!/usr/bin/env python3
import argparse

import torch
import numpy as np

from transformers import AutoTokenizer
from util import Dataset

from gpt import GPTLanguageModel
from loss import estimate_loss
from metrics import Metrics

def train(data, model, tokenizer, steps, report_frequency, lr):
  device = next(model.parameters()).device
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  metrics = Metrics()

  for step in range(steps):
    xb, yb = data.get_batch('train', device)
    # style tensor: single style = 0
    style = torch.zeros(xb.shape[0], dtype=torch.long, device=device)

    _, loss = model(xb, style, targets=yb) # forward pass with loss
    optimizer.zero_grad(set_to_none=True) # zero gradients
    loss.backward() # backpropagate to compute gradients
    optimizer.step() # update parameters with computed gradients
    if step % report_frequency == 0 or step == steps - 1:
      losses = estimate_loss(data, model, style=0)
      print(f"Step {step}, train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

      metrics_dict = metrics(data, model, style=0, tokenizer=tokenizer)
      print("Metrics:", metrics_dict)

      print()
# -----Argument parser----- 
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="input.txt")
parser.add_argument("--seed", type=int, default=None)

parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.2)

subparsers = parser.add_subparsers(dest="command", required=True)

train_parser = subparsers.add_parser("train")
train_parser.add_argument("--input", type=str, default="input.txt")  # <--- ADD THIS LINE
train_parser.add_argument("--seed", type=int, default=1337)
train_parser.add_argument("--save", type=str, default="model_subword.pth")
train_parser.add_argument("--steps", type=int, default=5000)
train_parser.add_argument("--report", type=int, default=500)
train_parser.add_argument("--lr", type=float, default=1e-3)

eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--load", type=str, default="model_subword.pth")
eval_parser.add_argument("--prompt", type=str)
eval_parser.add_argument("--token-count", type=int, default=300)

args = parser.parse_args()
# -----Seed-----
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
else:
    torch.manual_seed(1337)  # default seed
    np.random.seed(1337)

#-----Device-----
device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if device == "cpu":
  print("WARNING: Running on cpu!")

#-----Tokenizer(GPT2 Subwords)-----
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

#-----Load Data-----
with open(args.input, "r", encoding="utf-8") as f:
  content = f.read()

data_tensor = torch.tensor(tokenizer.encode(content, add_special_tokens=False), dtype=torch.long)
dataset = Dataset(data_tensor, args.context_size, args.batch_size)
#-----Model-----
model = GPTLanguageModel(
  vocab_size=len(tokenizer),
  n_embd=args.n_embd,
  context_size=args.context_size,
  n_head=args.n_head,
  n_layer=args.n_layer,
  n_styles=1,  # single style
)
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6}")
print(f"Using device: {device}")
print()
#-----Train or Eval-----
if args.command == "eval":
  print("=" * 20, "INFERENCE", "=" * 20)
  model.load_state_dict(torch.load(args.load))
  model.eval()
elif args.command == "train":
  print("=" * 20, "TRAINING", "=" * 20)
  train(dataset, model, tokenizer, args.steps, args.report, args.lr)
  torch.save(model.state_dict(), args.save)
  print("=" * 50)

# ----- Generation -----
context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_tokens = 300

if args.command == "eval" and args.prompt is not None:
    # encode prompt with GPT2 tokenizer (subwords)
    context = torch.tensor([tokenizer.encode(args.prompt, add_special_tokens=False)], dtype=torch.long, device=device)
    max_tokens = args.token_count

# style tensor: single style (0) for now
style_tensor = torch.zeros(context.size(0), dtype=torch.long, device=device)

# generate tokens
generated_tokens = model.generate(
    start_idx=context,
    style=style_tensor,
    number_of_tokens=max_tokens,
    use_cache=True
)

generated_text = tokenizer.decode(generated_tokens[0].tolist())
print("\n--- GENERATED TEXT ---\n")
generated_text = generated_text.replace("â€œ", '"').replace("â€\x9d", '"')
print(generated_text)