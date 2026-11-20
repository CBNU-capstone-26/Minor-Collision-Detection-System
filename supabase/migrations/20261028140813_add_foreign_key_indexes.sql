CREATE INDEX IF NOT EXISTS idx_analysis_tasks_video_id_fk
  ON public.analysis_tasks (video_id);

CREATE INDEX IF NOT EXISTS idx_crash_events_task_id_fk
  ON public.crash_events (task_id);

CREATE INDEX IF NOT EXISTS idx_crash_events_video_id_fk
  ON public.crash_events (video_id);

CREATE INDEX IF NOT EXISTS idx_vehicle_detection_tasks_video_id_fk
  ON public.vehicle_detection_tasks (video_id);

CREATE INDEX IF NOT EXISTS idx_videos_user_id_fk
  ON public.videos (user_id);;
