"""Keep the foreign-key indexes used by Supabase in SQLAlchemy metadata."""

import unittest

from app import db_models


class ForeignKeyIndexTests(unittest.TestCase):
    def test_foreign_key_indexes_are_declared_in_model_metadata(self):
        expected = {
            "idx_analysis_tasks_video_id_fk": ("analysis_tasks", "video_id"),
            "idx_crash_events_task_id_fk": ("crash_events", "task_id"),
            "idx_crash_events_video_id_fk": ("crash_events", "video_id"),
            "idx_vehicle_detection_tasks_video_id_fk": (
                "vehicle_detection_tasks", "video_id"
            ),
            "idx_videos_user_id_fk": ("videos", "user_id"),
        }

        indexes = {
            index.name: (table.name, tuple(column.name for column in index.columns))
            for table in db_models.Base.metadata.tables.values()
            for index in table.indexes
        }

        for name, (table, column) in expected.items():
            with self.subTest(index=name):
                self.assertEqual(indexes[name], (table, (column,)))


if __name__ == "__main__":
    unittest.main()
