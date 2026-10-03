from pathlib import Path

from torch_gpt.eval import plot_history


def test_plot_history_writes_a_png(tmp_path):
    history = [
        {"step": 1, "train_loss": 4.1, "val_loss": 4.1},
        {"step": 2, "train_loss": 3.2, "val_loss": 3.4},
        {"step": 3, "train_loss": 2.5, "val_loss": 2.9},
    ]
    output_path = tmp_path / "loss_curve.png"
    plot_history(history, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
