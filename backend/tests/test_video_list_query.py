"""Regression tests for efficient video-list relationship loading."""

import unittest

from app import db_models
from app.routers import videos


class CapturingQuery:
    def __init__(self):
        self.loader_options = ()

    def options(self, *options):
        self.loader_options = options
        return self

    def filter(self, *_conditions):
        return self

    def order_by(self, *_columns):
        return self


class CapturingSession:
    def __init__(self):
        self.query_object = CapturingQuery()

    def query(self, model):
        self.model = model
        return self.query_object


class VideoListQueryTests(unittest.TestCase):
    def test_video_list_eagerly_loads_events_and_analysis_tasks(self):
        db = CapturingSession()

        videos.video_list_query(db, user_id=7)

        self.assertIs(db.model, db_models.Video)
        option_paths = " ".join(str(option.path) for option in db.query_object.loader_options)
        self.assertIn("Video.crash_events", option_paths)
        self.assertIn("Video.analysis_tasks", option_paths)


if __name__ == "__main__":
    unittest.main()
