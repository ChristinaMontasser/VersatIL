import torch
from torch import nn
from torch.utils.data import DataLoader

from versatil.post_training_compression.pruning.iterativeprunner import (
    IterativeCalibrationPruner,
)


def test_iterative_calibration_pruner_streaming():
    """Basic smoke test: streaming calibration runs and produces sparsity.

    Constructs a tiny Conv/Linear model, runs one ICP iteration over a small
    synthetic calibration DataLoader and asserts that some parameters were
    zeroed by global unstructured pruning.
    """
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.fc = nn.Linear(4 * 8 * 8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = torch.relu(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = TinyModel()
    # Small synthetic calibration dataset: three batches
    batches = [torch.randn(2, 3, 8, 8) for _ in range(3)]
    loader = DataLoader(batches, batch_size=1)

    pruner = IterativeCalibrationPruner(T=1, initial_sparsity=0.2, final_sparsity=0.2)
    total, zeroed = pruner.prune(module=model, calibration_loader=loader)

    assert isinstance(total, int) and isinstance(zeroed, int)
    # Expect at least one parameter zeroed for this tiny model and amount=0.2
    assert zeroed > 0
