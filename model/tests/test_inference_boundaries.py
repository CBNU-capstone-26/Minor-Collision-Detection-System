import unittest
import tempfile
import numpy as np
import cv2
import config
import torch
from pathlib import Path
from unittest.mock import patch
from dataset import HitAndRunDataset
from inference import _crop_square_and_pad, fps_scaled_event_params, predict_events_and_clips, predict_full_video


class CropBoundaryTests(unittest.TestCase):
    def test_full_video_output_does_not_extend_short_source(self):
        class NormalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cam_layer = torch.nn.Identity()
                self.head_conv = torch.nn.Conv3d(1, 2, 1)
            def forward(self, clips):
                self.cam_layer(clips.mean(dim=1, keepdim=True))
                return torch.tensor([[1.0, 0.0]] * len(clips))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.mp4'
            annotation = Path(directory) / 'short.txt'
            annotation.write_text('car,0,1,1,14,14\n')
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
            writer.release()
            output, _ = predict_full_video(NormalModel(), video_path=path, txt_path=annotation,
                                           target_id=0, output_dir=Path(directory) / 'output',
                                           resize=(16, 16), clip_length=4, display_offset=3)
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            finally:
                capture.release()

    def test_coarse_and_full_scans_agree_for_all_positive_windows(self):
        class PositiveRunner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[0.0, 5.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            for frames in (4, 5, 8, 9, 10, 13, 15):
                with self.subTest(frames=frames):
                    path = Path(directory) / f'coarse-{frames}.mp4'
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (16, 16))
                    self.assertTrue(writer.isOpened())
                    for _ in range(frames):
                        writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
                    writer.release()
                    results = []
                    with patch('inference.get_runner', return_value=PositiveRunner()), \
                         patch.object(config, 'PREDICT_MIN_EVENT_SPAN_FRAMES', 0), \
                         patch.object(config, 'PREDICT_FPS_SCALED_POSTPROC', False):
                        for coarse in (False, True):
                            stats = {}
                            events = predict_events_and_clips(
                                model, path, (1, 1, 10, 10), directory, resize=(16, 16),
                                clip_length=4, render_clips=False, use_coarse_scan=coarse,
                                coarse_step=5, coarse_pad=4, stats=stats,
                            )
                            self.assertEqual(stats['windows_eval'], stats['windows_total'])
                            results.append(events)
                    self.assertEqual(results[0], results[1])

    def test_full_video_output_preserves_source_fps(self):
        class NormalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cam_layer = torch.nn.Identity()
                self.head_conv = torch.nn.Conv3d(1, 2, 1)
            def forward(self, clips):
                self.cam_layer(clips.mean(dim=1, keepdim=True))
                return torch.tensor([[1.0, 0.0]] * len(clips))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.mp4'
            annotation = Path(directory) / 'video.txt'
            annotation.write_text('car,0,1,1,14,14\n')
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
            writer.release()
            output, _ = predict_full_video(NormalModel(), video_path=path, txt_path=annotation,
                                           target_id=0, output_dir=Path(directory) / 'output',
                                           resize=(16, 16), clip_length=4, display_offset=1)
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 15, places=1)
            finally:
                capture.release()

    def test_training_crop_matches_inference_for_tiny_and_normal_boxes(self):
        dataset = HitAndRunDataset.__new__(HitAndRunDataset)
        dataset.resize = (8, 8)
        frame = np.arange(10 * 10 * 3, dtype=np.uint16).reshape(10, 10, 3).astype(np.uint8)
        for bbox in ((3, 3, 4, 4), (0, 0, 5, 7), (8, 8, 10, 10)):
            with self.subTest(bbox=bbox):
                service, _ = _crop_square_and_pad(frame, bbox, 1.0, (8, 8))
                training = dataset._crop_and_pad(frame, bbox, 1.0)
                np.testing.assert_array_equal(service, training)

    def test_invalid_fps_does_not_crash_postprocessing(self):
        for fps in (float('nan'), float('inf'), float('-inf'), 0, -1):
            with self.subTest(fps=fps):
                self.assertEqual(fps_scaled_event_params(fps, coarse_step=10, min_span=35,
                                                         clip_length=30, enabled=True), (10, 35))
    def test_one_pixel_box_is_resized_without_failure(self):
        frame = np.full((10, 10, 3), 127, dtype=np.uint8)
        for bbox in ((3, 3, 4, 4), (0, 0, 1, 1), (9, 9, 10, 10)):
            with self.subTest(bbox=bbox):
                crop, bounds = _crop_square_and_pad(frame, bbox, 1.0, (8, 8))
                self.assertEqual(crop.shape, (8, 8, 3))
                self.assertGreater(bounds[2], bounds[0])
                self.assertGreater(bounds[3], bounds[1])

    def test_unreadable_video_is_not_reported_as_successful_empty_analysis(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory, \
             patch('inference.get_runner', return_value=object()):
            with self.assertRaises(ValueError):
                predict_events_and_clips(model, Path(directory) / 'missing.mp4',
                                         (0, 0, 1, 1), directory, render_clips=False)

    def test_short_video_is_padded_and_actually_evaluated(self):
        class Runner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[1.0, 0.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
            writer.release()
            stats = {}
            with patch('inference.get_runner', return_value=Runner()), \
                 patch.object(config, 'PREDICT_MIN_EVENT_SPAN_FRAMES', 0):
                result = predict_events_and_clips(model, path, (0, 0, 1, 1), directory,
                                                  render_clips=False, use_coarse_scan=False, stats=stats)
            self.assertEqual(result, [])
            self.assertEqual(stats['frames'], 3)
            self.assertEqual(stats['windows_total'], 1)
            self.assertEqual(stats['windows_eval'], 1)

    def test_invalid_video_metadata_does_not_stop_inference(self):
        class Runner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[1.0, 0.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
            writer.release()
            original_capture = cv2.VideoCapture

            class BadMetadataCapture:
                def __init__(self, filename):
                    self.capture = original_capture(filename)
                def get(self, prop):
                    if prop == cv2.CAP_PROP_FPS:
                        return float('nan')
                    if prop == cv2.CAP_PROP_FRAME_COUNT:
                        return float('inf')
                    return self.capture.get(prop)
                def __getattr__(self, name):
                    return getattr(self.capture, name)

            stats = {}
            with patch('inference.get_runner', return_value=Runner()), \
                 patch.object(config, 'PREDICT_MIN_EVENT_SPAN_FRAMES', 0), \
                 patch('inference.cv2.VideoCapture', side_effect=BadMetadataCapture):
                result = predict_events_and_clips(model, path, (0, 0, 1, 1), directory,
                                                  render_clips=False, use_coarse_scan=False, stats=stats)
            self.assertEqual(result, [])
            self.assertEqual(stats['windows_eval'], 1)

    def test_video_too_short_for_event_filter_is_not_reported_as_safe(self):
        class PositiveRunner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[0.0, 5.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'too-short.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 30, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(35):
                writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
            writer.release()
            with patch('inference.get_runner', return_value=PositiveRunner()):
                with self.assertRaisesRegex(ValueError, '짧'):
                    predict_events_and_clips(model, path, (0, 0, 1, 1), directory,
                                             render_clips=False, use_coarse_scan=False)

    def test_minimum_length_tracks_fps_scaled_event_filter(self):
        class PositiveRunner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[0.0, 5.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            for fps in (15, 30, 60):
                _, min_span = fps_scaled_event_params(fps)
                for frame_count in (min_span, min_span + 1):
                    with self.subTest(fps=fps, frames=frame_count):
                        path = Path(directory) / f'{fps}-{frame_count}.mp4'
                        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (16, 16))
                        self.assertTrue(writer.isOpened())
                        for _ in range(frame_count):
                            writer.write(np.full((16, 16, 3), 127, dtype=np.uint8))
                        writer.release()
                        with patch('inference.get_runner', return_value=PositiveRunner()):
                            if frame_count == min_span:
                                with self.assertRaisesRegex(ValueError, '짧'):
                                    predict_events_and_clips(model, path, (0, 0, 2, 2), directory,
                                                             resize=(16, 16), render_clips=False,
                                                             use_coarse_scan=False)
                            else:
                                events = predict_events_and_clips(model, path, (0, 0, 2, 2), directory,
                                                                  resize=(16, 16), render_clips=False,
                                                                  use_coarse_scan=False)
                                self.assertEqual(len(events), 1)

    def test_positive_predictions_create_readable_result_clips(self):
        class AccidentRunner:
            name = 'test'
            def __call__(self, batch):
                return torch.tensor([[0.0, 5.0]] * len(batch))

        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(45):
                frame = np.full((48, 64, 3), i, dtype=np.uint8)
                writer.write(frame)
            writer.release()
            with patch('inference.get_runner', return_value=AccidentRunner()):
                events = predict_events_and_clips(
                    model, path, (10, 10, 40, 40), directory, clip_length=4,
                    resize=(32, 32), display_offset=1, cam_overlay=False,
                    use_coarse_scan=False, clip_pad_frames=0,
                )
            self.assertEqual(len(events), 1)
            self.assertLessEqual(events[0]['start_frame'], events[0]['end_frame'])
            for name in ('clip_path', 'raw_clip_path'):
                capture = cv2.VideoCapture(str(events[0][name]))
                try:
                    self.assertTrue(capture.isOpened(), name)
                    self.assertTrue(capture.read()[0], name)
                finally:
                    capture.release()
