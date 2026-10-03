"""Train the torch TinyGPT. Usage: python -m torch_gpt.train [options]"""

import argparse
from pathlib import Path

import torch

from common.data import prepare
from torch_gpt.model import GPTConfig, TinyGPT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/train.txt", help="Path to a plain-text training corpus")
    p.add_argument("--checkpoint", default="checkpoints/torch_gpt.pt")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=6000)
    p.add_argument("--print-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    tokenizer, train_ids, val_ids = prepare(args.data, args.val_fraction)
    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)
    print(f"Loaded {len(train_ids) + len(val_ids)} characters, vocab size {tokenizer.vocab_size}")

    def get_batch(split):
        source = train_data if split == "train" else val_data
        ix = torch.randint(0, len(source) - args.block_size, (args.batch_size,))
        x = torch.stack([source[i:i + args.block_size] for i in ix])
        y = torch.stack([source[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nTraining...\n")
    model.train()
    best_val = float("inf")

    for step in range(1, args.max_steps + 1):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.print_every == 0 or step == 1:
            print(f"step {step:5d} | train loss {loss.item():.4f}")

        if step % args.eval_every == 0 or step == 1:
            losses = estimate_loss()
            print(f"          eval | train {losses['train']:.4f} | val {losses['val']:.4f}")

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": config,
                    "tokenizer": tokenizer,
                    "best_val": best_val,
                    "step": step,
                }, ckpt_path)
                print(f"          saved best checkpoint -> {ckpt_path} (val={best_val:.4f})")

    print(f"\nDone. Best checkpoint at {ckpt_path} (val={best_val:.4f})")


if __name__ == "__main__":
    main()
