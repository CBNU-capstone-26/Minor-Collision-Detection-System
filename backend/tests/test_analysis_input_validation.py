import unittest
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import HTTPException
from app.api_schemas import AnalyzeRequest
from app.routers import analysis, videos


class AnalysisInputValidationTests(unittest.TestCase):
    def test_valid_bounding_box_is_enqueued(self):
        video=SimpleNamespace(id=1,width=100,height=80)
        req=AnalyzeRequest(bbox_xmin=0,bbox_ymin=0,bbox_xmax=100,bbox_ymax=80)
        with patch.object(analysis,'_get_owned_video',return_value=video), \
             patch.object(analysis,'_enqueue',return_value='queued') as enqueue:
            self.assertEqual(analysis.analyze_video(1,req,None,None),'queued')
            enqueue.assert_called_once()

    def test_extremely_large_finite_timestamp_is_clamped(self):
        video=SimpleNamespace(id=1,width=100,height=80,fps=25,total_frames=100)
        with patch.object(videos,'_get_owned_video',return_value=video), \
             patch.object(videos,'_enqueue_vehicle_detection',return_value='queued') as enqueue:
            self.assertEqual(videos.enqueue_vehicle_detection(1,1e308,None,None),'queued')
            self.assertEqual(enqueue.call_args.args[1],99/25)

    def test_invalid_bounding_boxes_do_not_enqueue(self):
        video=SimpleNamespace(id=1,width=100,height=80)
        for coords in [(-1,0,50,50),(10,10,10,50),(50,10,20,50),(0,0,101,80),(0,0,100,81)]:
            with self.subTest(coords=coords), patch.object(analysis,'_get_owned_video',return_value=video), \
                 patch.object(analysis,'_enqueue') as enqueue:
                req=AnalyzeRequest(bbox_xmin=coords[0],bbox_ymin=coords[1],bbox_xmax=coords[2],bbox_ymax=coords[3])
                with self.assertRaises(HTTPException) as error:
                    analysis.analyze_video(1,req,None,None)
                self.assertEqual(error.exception.status_code,400)
                enqueue.assert_not_called()

    def test_invalid_detection_timestamp_is_rejected_before_enqueue(self):
        video=SimpleNamespace(id=1,width=100,height=80,fps=25,total_frames=100)
        for timestamp in [float('nan'),float('inf'),float('-inf')]:
            with self.subTest(timestamp=timestamp), patch.object(videos,'_get_owned_video',return_value=video), \
                 patch.object(videos,'_enqueue_vehicle_detection') as enqueue:
                with self.assertRaises(HTTPException) as error:
                    videos.enqueue_vehicle_detection(1,timestamp,None,None)
                self.assertEqual(error.exception.status_code,400)
                enqueue.assert_not_called()

    def test_sync_detection_also_rejects_nonfinite_timestamp(self):
        video=SimpleNamespace(id=1,width=100,height=80,fps=25,total_frames=100)
        with patch.object(videos,'_get_owned_video',return_value=video), \
             patch.object(videos,'get_storage') as storage:
            with self.assertRaises(HTTPException) as error:
                videos.detect_vehicles_in_video(1,float('inf'),None,None)
            self.assertEqual(error.exception.status_code,400)
            storage.assert_not_called()
