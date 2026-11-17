import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import config
import speed_check
from dataset import HitAndRunDataset


class CropAugmentationTests(unittest.TestCase):
    def test_invalid_settings_fail_on_import(self):
        for key, value in [('TRAIN_R_AUGMENT_PROB', '-0.1'), ('TRAIN_R_AUGMENT_PROB', '1.1'),
                           ('TRAIN_R_AUGMENT_PROB', 'nan'), ('TRAIN_R_AUGMENT_PROB', 'inf'), ('TRAIN_R_AUGMENT_VALUES', '0'),
                           ('TRAIN_R_AUGMENT_VALUES', '-2'), ('TRAIN_R_AUGMENT_VALUES', 'inf'),
                           ('TRAIN_R_AUGMENT_VALUES', 'nan')]:
            with self.subTest(key=key, value=value):
                env = os.environ.copy()
                env.update(PYTHONPATH='model', TRAIN_R_AUGMENT_PROB='0', TRAIN_R_AUGMENT_VALUES='1.5,2')
                env[key] = value
                p = subprocess.run([sys.executable, '-c', 'import config'], env=env, capture_output=True, text=True)
                self.assertNotEqual(p.returncode, 0)
                self.assertIn(key, p.stderr)

    def test_validation_and_disabled_augmentation_keep_base_r(self):
        ds = HitAndRunDataset.__new__(HitAndRunDataset)
        ds.r_value = 1.0
        with patch.object(config, 'TRAIN_R_AUGMENT_VALUES', (2.0,)):
            ds.augment = False
            with patch.object(config, 'TRAIN_R_AUGMENT_PROB', 1.0):
                self.assertEqual(ds._sample_crop_r(), 1.0)
            ds.augment = True
            with patch.object(config, 'TRAIN_R_AUGMENT_PROB', 0.0):
                self.assertEqual(ds._sample_crop_r(), 1.0)

    def test_enabled_augmentation_requires_values(self):
        env = os.environ.copy()
        env.update(PYTHONPATH='model', TRAIN_R_AUGMENT_PROB='0.5', TRAIN_R_AUGMENT_VALUES='')
        p = subprocess.run([sys.executable, '-c', 'import config'], env=env, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('TRAIN_R_AUGMENT_VALUES', p.stderr)

    def test_speed_check_uses_augmented_r_for_every_frame(self):
        ds = HitAndRunDataset.__new__(HitAndRunDataset)
        ds.samples = [{'mp4_path': 'test', 'start_f': 0, 'bbox': [0, 0, 2, 2]}]
        ds.clip_length = 2
        ds.r_value = 1.0
        ds.augment = True
        used = []
        ds._crop_and_pad = lambda frame, bbox, r: (used.append(r) or frame)
        ds._apply_augmentation = lambda frames: frames
        ds._frames_to_tensor = lambda frames: None
        class Capture:
            def set(self, *args): pass
            def read(self): return True, np.zeros((4, 4, 3), dtype=np.uint8)
            def release(self): pass
        with patch.object(config, 'TRAIN_R_AUGMENT_PROB', 1.0), \
             patch.object(config, 'TRAIN_R_AUGMENT_VALUES', (2.0,)), \
             patch.object(speed_check.cv2, 'VideoCapture', return_value=Capture()):
            speed_check.part_clip(ds, [0])
        self.assertEqual(used, [2.0, 2.0])
