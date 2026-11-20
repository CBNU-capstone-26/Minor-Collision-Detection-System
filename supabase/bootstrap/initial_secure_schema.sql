-- Bootstrap for an EMPTY Supabase database. Run before migration history.
-- Refuses any existing application table; never use it to overwrite an existing database.
BEGIN;

DO $$ BEGIN
  IF to_regclass('public.users') IS NOT NULL OR to_regclass('public.videos') IS NOT NULL OR to_regclass('public.analysis_tasks') IS NOT NULL OR to_regclass('public.vehicle_detection_tasks') IS NOT NULL OR to_regclass('public.crash_events') IS NOT NULL OR to_regclass('public.file_cleanup_queue') IS NOT NULL THEN
    RAISE EXCEPTION 'Initial schema requires an empty application database';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') OR
     NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
    RAISE EXCEPTION 'Supabase API roles are required';
  END IF;
END $$;

CREATE TABLE public.users (
	id SERIAL NOT NULL,
	username VARCHAR(255) NOT NULL,
	password_hash VARCHAR(255) NOT NULL,
	name VARCHAR(100),
	email VARCHAR(255),
	phone VARCHAR(50),
	role VARCHAR(50),
	created_at TIMESTAMP WITHOUT TIME ZONE,
	PRIMARY KEY (id),
	UNIQUE (email)
);

CREATE INDEX IF NOT EXISTS ix_users_id ON public.users (id);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_username ON public.users (username);

CREATE TABLE public.videos (
	id SERIAL NOT NULL,
	user_id INTEGER NOT NULL,
	video_name VARCHAR(255) NOT NULL,
	title VARCHAR(100),
	video_path VARCHAR(500) NOT NULL,
	recording_date DATE,
	camera_location VARCHAR(255) NOT NULL,
	width INTEGER NOT NULL,
	height INTEGER NOT NULL,
	fps FLOAT NOT NULL,
	total_frames INTEGER NOT NULL,
	detected_vehicles TEXT,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	PRIMARY KEY (id),
	FOREIGN KEY(user_id) REFERENCES public.users (id)
);

CREATE INDEX IF NOT EXISTS idx_videos_user_id_fk ON public.videos (user_id);

CREATE INDEX IF NOT EXISTS ix_videos_id ON public.videos (id);

CREATE TABLE public.analysis_tasks (
	id SERIAL NOT NULL,
	video_id INTEGER NOT NULL,
	celery_task_id VARCHAR(255),
	bbox_xmin INTEGER NOT NULL,
	bbox_ymin INTEGER NOT NULL,
	bbox_xmax INTEGER NOT NULL,
	bbox_ymax INTEGER NOT NULL,
	status VARCHAR(50),
	error_message TEXT,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	finished_at TIMESTAMP WITHOUT TIME ZONE,
	event_count INTEGER,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES public.videos (id)
);

CREATE INDEX IF NOT EXISTS idx_analysis_tasks_video_id_fk ON public.analysis_tasks (video_id);

CREATE UNIQUE INDEX IF NOT EXISTS ix_analysis_tasks_celery_task_id ON public.analysis_tasks (celery_task_id);

CREATE INDEX IF NOT EXISTS ix_analysis_tasks_id ON public.analysis_tasks (id);

CREATE TABLE public.vehicle_detection_tasks (
	id SERIAL NOT NULL,
	video_id INTEGER NOT NULL,
	celery_task_id VARCHAR(255),
	timestamp_sec FLOAT NOT NULL,
	status VARCHAR(50) NOT NULL,
	error_message TEXT,
	result_json TEXT,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	finished_at TIMESTAMP WITHOUT TIME ZONE,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES public.videos (id)
);

CREATE INDEX IF NOT EXISTS idx_vehicle_detection_tasks_video_id_fk ON public.vehicle_detection_tasks (video_id);

CREATE UNIQUE INDEX IF NOT EXISTS ix_vehicle_detection_tasks_celery_task_id ON public.vehicle_detection_tasks (celery_task_id);

CREATE INDEX IF NOT EXISTS ix_vehicle_detection_tasks_id ON public.vehicle_detection_tasks (id);

CREATE TABLE public.crash_events (
	id SERIAL NOT NULL,
	task_id INTEGER NOT NULL,
	video_id INTEGER NOT NULL,
	timestamp_sec FLOAT NOT NULL,
	frame_number INTEGER NOT NULL,
	end_timestamp_sec FLOAT,
	end_frame_number INTEGER,
	crash_prob FLOAT,
	cam_heatmap_path VARCHAR(500),
	clip_start_sec FLOAT,
	created_at TIMESTAMP WITHOUT TIME ZONE,
	PRIMARY KEY (id),
	FOREIGN KEY(video_id) REFERENCES public.videos (id),
	FOREIGN KEY(task_id) REFERENCES public.analysis_tasks (id)
);

CREATE INDEX IF NOT EXISTS idx_crash_events_task_id_fk ON public.crash_events (task_id);

CREATE INDEX IF NOT EXISTS idx_crash_events_video_id_fk ON public.crash_events (video_id);

CREATE INDEX IF NOT EXISTS ix_crash_events_id ON public.crash_events (id);

ALTER TABLE public."users" ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES ON TABLE public."users" FROM PUBLIC, anon, authenticated;

CREATE POLICY "Deny direct API access" ON public."users" FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

ALTER TABLE public."videos" ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES ON TABLE public."videos" FROM PUBLIC, anon, authenticated;

CREATE POLICY "Deny direct API access" ON public."videos" FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

ALTER TABLE public."analysis_tasks" ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES ON TABLE public."analysis_tasks" FROM PUBLIC, anon, authenticated;

CREATE POLICY "Deny direct API access" ON public."analysis_tasks" FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

ALTER TABLE public."vehicle_detection_tasks" ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES ON TABLE public."vehicle_detection_tasks" FROM PUBLIC, anon, authenticated;

ALTER TABLE public."crash_events" ENABLE ROW LEVEL SECURITY;

REVOKE ALL PRIVILEGES ON TABLE public."crash_events" FROM PUBLIC, anon, authenticated;

CREATE POLICY "Deny direct API access" ON public."crash_events" FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);

COMMIT;
