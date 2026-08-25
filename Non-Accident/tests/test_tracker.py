import unittest

from src.tracker import Observation, choose_reference_tracks
from src.nonaccident_pipeline import representative_boxes


class TrackerSelectionTests(unittest.TestCase):
    def test_selects_two_stationary_tracks_near_the_moving_vehicle(self):
        observations = [
            Observation(1, 0, [100, 100, 180, 180], 0.9),
            Observation(1, 10, [160, 100, 240, 180], 0.9),
            Observation(2, 0, [20, 100, 100, 180], 0.9),
            Observation(2, 10, [20, 100, 100, 180], 0.9),
            Observation(3, 0, [240, 100, 320, 180], 0.9),
            Observation(3, 10, [240, 100, 320, 180], 0.9),
            Observation(4, 0, [700, 700, 760, 760], 0.9),
            Observation(4, 10, [700, 700, 760, 760], 0.9),
        ]

        moving_id, references = choose_reference_tracks(observations, 10)

        self.assertEqual(moving_id, 1)
        self.assertEqual(references, [2, 3])

    def test_returns_no_references_when_fewer_than_two_stationary_candidates_exist(self):
        observations = [
            Observation(1, 0, [100, 100, 180, 180], 0.9),
            Observation(1, 10, [160, 100, 240, 180], 0.9),
            Observation(2, 0, [20, 100, 100, 180], 0.9),
            Observation(2, 10, [20, 100, 100, 180], 0.9),
        ]

        moving_id, references = choose_reference_tracks(observations, 10)

        self.assertEqual(moving_id, 1)
        self.assertEqual(references, [])

    def test_representative_boxes_avoid_heavy_occlusion(self):
        observations = [
            Observation(1, 0, [100, 100, 200, 200], 0.8),
            Observation(1, 10, [100, 100, 200, 200], 0.95),
            Observation(2, 0, [300, 100, 400, 200], 0.8),
            Observation(2, 10, [300, 100, 400, 200], 0.95),
            Observation(9, 0, [0, 0, 20, 20], 0.9),
            Observation(9, 10, [120, 100, 320, 200], 0.9),
        ]
        frame, boxes = representative_boxes(observations, [1, 2], moving_track_id=9)
        self.assertEqual(frame, 0)
        self.assertEqual(boxes, {1: [100, 100, 200, 200], 2: [300, 100, 400, 200]})


if __name__ == "__main__":
    unittest.main()
