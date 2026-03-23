import torch


@torch.no_grad()
def estimate_loss(data, model, style, eval_iters=100):
    device = next(model.parameters()).device

    style_tensor = torch.full((data.batch_size,), style, dtype=torch.long, device=device)

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data.get_batch(split, device)
            _, loss = model(X, style_tensor, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
