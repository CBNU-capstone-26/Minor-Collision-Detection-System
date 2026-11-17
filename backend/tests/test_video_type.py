"""Video type changes must persist and respect video ownership."""
import unittest
from datetime import datetime

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import api_schemas, db_models
from app.db_connection import Base
from app.routers.videos import update_video_type


class VideoTypeTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.owner = db_models.User(username="owner", password_hash="x")
        self.other = db_models.User(username="other", password_hash="x")
        self.db.add_all([self.owner, self.other])
        self.db.flush()
        self.video = db_models.Video(user_id=self.owner.id, video_name="a.mp4", video_path="a.mp4", width=100, height=100, fps=25, total_frames=100, created_at=datetime.utcnow())
        self.db.add(self.video)
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_type_is_saved_and_returned(self):
        result = update_video_type(self.video.id, api_schemas.VideoTypeUpdate(video_type="render_3d"), self.db, self.owner)
        self.db.expire_all()
        self.assertEqual(result.video_type, "render_3d")
        self.assertEqual(self.db.get(db_models.Video, self.video.id).video_type, "render_3d")

    def test_other_user_cannot_change_type(self):
        with self.assertRaises(HTTPException) as raised:
            update_video_type(self.video.id, api_schemas.VideoTypeUpdate(video_type="game"), self.db, self.other)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertIsNone(self.video.video_type)

    def test_invalid_type_is_rejected(self):
        with self.assertRaises(ValidationError):
            api_schemas.VideoTypeUpdate(video_type="unknown")


if __name__ == "__main__":
    unittest.main()
