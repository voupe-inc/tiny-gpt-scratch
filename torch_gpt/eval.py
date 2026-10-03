"""Plot the loss curve and print sample generations for a trained checkpoint.

Usage: python -m torch_gpt.eval --checkpoint checkpoints/torch_gpt.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from torch_gpt.model import TinyGPT

DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little robot",
    "In the forest",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/torch_gpt.pt")
    p.add_argument("--history", default="checkpoints/history.json")
    p.add_argument("--plot-output", default="checkpoints/loss_curve.png")
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    return p.parse_args()


def plot_history(history: list[dict], output_path: Path):
    steps = [h["step"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, train_loss, label="train")
    ax.plot(steps, val_loss, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("TinyGPT training loss")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_samples(model, tokenizer, device, prompts, max_new_tokens, temperature):
    samples = []
    for prompt in prompts:
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)[0].tolist()
        samples.append(tokenizer.decode(out))
    return samples


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    history_path = Path(args.history)
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plot_path = Path(args.plot_output)
        plot_history(history, plot_path)
        print(f"Saved loss curve -> {plot_path}")
    else:
        print(f"No history file at {history_path}, skipping loss plot")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded step {ckpt['step']} | best val {ckpt['best_val']:.4f}")

    samples = generate_samples(
        model, ckpt["tokenizer"], device, args.prompts, args.max_new_tokens, args.temperature
    )
    for prompt, sample in zip(args.prompts, samples):
        print(f"\n--- Prompt: {prompt!r} ---")
        print(sample)


if __name__ == "__main__":
    main()
