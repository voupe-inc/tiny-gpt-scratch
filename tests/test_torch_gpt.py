import torch

from torch_gpt.model import GPTConfig, TinyGPT


def _tiny_config(vocab_size=11):
    return GPTConfig(vocab_size=vocab_size, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def test_forward_output_shape():
    torch.manual_seed(0)
    config = _tiny_config()
    model = TinyGPT(config)
    idx = torch.randint(0, config.vocab_size, (3, config.block_size))
    logits, loss = model(idx)
    assert logits.shape == (3, config.block_size, config.vocab_size)
    assert loss is None


def test_loss_decreases_on_tiny_batch():
    """A model that can't overfit a single tiny batch has a wiring bug."""
    torch.manual_seed(0)
    config = _tiny_config()
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    losses = []
    for _ in range(50):
        _, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_generate_appends_requested_token_count():
    torch.manual_seed(0)
    config = _tiny_config()
    model = TinyGPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 3))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0)
    assert out.shape == (1, 8)
