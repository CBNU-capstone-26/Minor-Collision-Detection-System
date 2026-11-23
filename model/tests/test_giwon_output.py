import contextlib
import csv
import io
from pathlib import Path
from unittest.mock import patch

import config
import torch
from epoch_suggest import suggest
from inference import predict_events_and_clips


def test_coarse_threshold_warning_only_when_unsafe(tmp_path):
    model = torch.nn.Linear(1, 1)
    for enabled, probability, expected in [(True, config.A_THRESHOLD, True), (True, 0.1, False), (False, 0.9, False)]:
        output = io.StringIO()
        with patch('inference.get_runner', return_value=object()), contextlib.redirect_stdout(output):
            try:
                predict_events_and_clips(model, tmp_path / 'missing.mp4', (0, 0, 1, 1), tmp_path,
                                         use_coarse_scan=enabled, coarse_prob=probability, render_clips=False)
            except ValueError:
                pass
        assert ('약한 충돌 신호' in output.getvalue()) is expected


def test_suggest_outputs_runnable_logged_commands(tmp_path):
    run = tmp_path / 'run with spaces'
    (run / 'epochs').mkdir(parents=True)
    (run / 'epochs' / 'model_top3_ep001.pth').touch()
    with (run / 'trainlog_test.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['epoch', 'monitor', 'val_real_bal', 'val_real_recallA', 'val_real_precA'])
        writer.writeheader()
        writer.writerow(dict(epoch=1, monitor=0.3, val_real_bal=0.3, val_real_recallA=0.9, val_real_precA=0.9))
    _, lines = suggest(run)
    output = '\n'.join(lines)
    assert 'outputs/console/eval_' in output
    assert 'mkdir -p outputs/console' in output
    assert 'HEAD_TEMPORAL_POOL=top3' in output
    assert "'" + str(run) + "'" in output
    assert 'model.epoch_rank' in output
