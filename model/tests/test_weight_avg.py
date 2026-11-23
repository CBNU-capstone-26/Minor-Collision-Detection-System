from pathlib import Path

import pytest
import torch

from weight_avg import average, main, parse_epochs


def test_average_checkpoint_values_and_preserve_metadata(tmp_path):
    run = tmp_path / 'run'
    epochs = run / 'epochs'
    epochs.mkdir(parents=True)
    for epoch, value in [(25, 2.0), (26, 4.0)]:
        torch.save({'weight': torch.tensor([value]), 'num_batches_tracked': torch.tensor(epoch)},
                   epochs / f'hitandrun_x3d_261121_1915_ptY_bal_lqw_top3_ep{epoch:03d}.pth')
    result = main(['--run', str(run), '--epochs', '25-26'])
    assert result.parent == run / 'avg'
    assert 'avg25-26' in result.name and '_top3' in result.name
    averaged = torch.load(result, map_location='cpu', weights_only=True)
    assert averaged['weight'].item() == 3.0
    assert averaged['num_batches_tracked'].item() == 26


def test_parse_epochs_removes_duplicates_and_supports_descending_ranges():
    assert parse_epochs('3-1,2,5') == [3, 2, 1, 5]


def test_average_rejects_incompatible_checkpoints(tmp_path):
    first, second = tmp_path / 'a.pth', tmp_path / 'b.pth'
    torch.save({'a': torch.tensor([1.0])}, first)
    torch.save({'b': torch.tensor([2.0])}, second)
    with pytest.raises(SystemExit, match='키가 다름'):
        average([first, second])
