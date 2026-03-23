#!/usr/bin/env python3
import argparse
import string

import torch
import numpy as np

from util import CharacterTokenizer, Dataset, MultiStyleDataset
from gpt import GPTLanguageModel
from metrics import Metrics
from loss import estimate_loss


def train(multi_style_data, model, tokenizer, steps, report_frequency, lr):
  device = next(model.parameters()).device
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  metrics = Metrics()

  for step in range(steps):
      xb, yb, style = multi_style_data.get_batch('train', device)

      _, loss = model(xb, style, targets=yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      if step % report_frequency == 0 or step == steps - 1:
          for style_index, dataset in enumerate(multi_style_data.datasets):
            losses_dict = estimate_loss(dataset, model, style_index)
            metrics_dict = metrics(dataset, model, style_index, tokenizer)

            report_str = ", ".join([f"{k} loss: {v:.4f}" for k, v in losses_dict.items()] + [f"{k} metric: {v:.4f}" for k, v in metrics_dict.items()])
            print(f"Step {step}, style {style_index}: {report_str}")

          print()


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="input.txt")
parser.add_argument("--finetune-input", type=str, default="finetune_input.txt")
parser.add_argument("--seed", type=int, default=1337)

parser.add_argument("--context-size", type=int, default=256)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--n-embd", type=int, default=384)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.2)

subparsers = parser.add_subparsers(dest="command", required=True)

finetune_parser = subparsers.add_parser("finetune")
finetune_parser.add_argument("--load", type=str, default="model.pth")
finetune_parser.add_argument("--save", type=str, default="model_finetuned.pth")
finetune_parser.add_argument("--steps", type=int, default=5000)
finetune_parser.add_argument("--lr", type=float, default=5e-5)
finetune_parser.add_argument("--report", type=int, default=500)

eval_parser = subparsers.add_parser("eval")
eval_parser.add_argument("--load", type=str, default="model_finetuned.pth")
eval_parser.add_argument("--prompt", type=str)
eval_parser.add_argument("--token-count", type=int, default=300)
eval_parser.add_argument("--style", type=int, default=0, help="Style index for style embedding")

args = parser.parse_args()

torch.manual_seed(args.seed)
batch_size = args.batch_size
context_size = args.context_size
n_embd = args.n_embd
n_head = args.n_head
n_layer = args.n_layer
dropout = args.dropout

# replace this with ur hw backend if needed
device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if device == "cpu":
  print("WARNING: Running on cpu!")

with open(args.input, "r") as f:
  content = f.read()

tokenizer = CharacterTokenizer(content)
data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
dataset = Dataset(data, context_size, batch_size)

with open(args.finetune_input, "r") as f:
  finetune_content = f.read()

# reuse the same tokenizer!
finetune_data = torch.tensor(tokenizer.encode(finetune_content), dtype=torch.long)
finetune_dataset = Dataset(finetune_data, context_size, batch_size)

multi_style_dataset = MultiStyleDataset([dataset, finetune_dataset], [0.2, 0.8])

model = GPTLanguageModel(len(tokenizer.vocab), n_embd, context_size, n_head, n_layer, n_styles=len(multi_style_dataset.datasets))
model = model.to(device)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6}")
print(f"Using device: {device}")
print()

# non strict load
model.load_state_dict(torch.load(args.load), strict=False)

if args.command == "eval":
  print("=" * 20, "INFERENCE", "=" * 20)
  model.eval()
elif args.command == "finetune":
  print("=" * 20, "TRAINING", "=" * 20)
  model.train()
  train(multi_style_dataset, model, tokenizer, args.steps, args.report, args.lr)
  torch.save(model.state_dict(), args.save)
  print("=" * 50)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
max_tokens = 300
if args.command == "eval":
  if args.prompt is not None:
    context = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
  max_tokens = args.token_count

style_tensor = torch.tensor([args.style], dtype=torch.long, device=device)
print(
  tokenizer.decode(
    model.generate(start_idx=context, style=style_tensor, number_of_tokens=max_tokens, use_cache=True)[0].tolist()
  )
)
