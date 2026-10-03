"""Generate text from a trained torch TinyGPT checkpoint, no retraining. Usage:
python -m torch_gpt.sample --checkpoint checkpoints/torch_gpt.pt --prompt "I fell in love"
"""

import argparse

import torch

from torch_gpt.model import TinyGPT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/torch_gpt.pt")
    p.add_argument("--prompt", default="I fell in love")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # weights_only=False: checkpoint bundles a GPTConfig/tokenizer alongside the tensors.
    # Only load checkpoints you trust, as with any pickle-based format.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyGPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded step {ckpt['step']} | best val {ckpt['best_val']:.4f}")

    tokenizer = ckpt["tokenizer"]
    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
