-- Retryable post-commit file cleanup. The application may have created the
-- table already; protect it in either case.
CREATE TABLE IF NOT EXISTS public.file_cleanup_queue (
    id serial PRIMARY KEY,
    object_key text NOT NULL,
    guard_key text NOT NULL,
    guard_kind varchar(16) NOT NULL,
    cleanup_kind varchar(16) NOT NULL,
    attempts integer NOT NULL DEFAULT 0
);
ALTER TABLE public.file_cleanup_queue ENABLE ROW LEVEL SECURITY;
REVOKE ALL PRIVILEGES ON TABLE public.file_cleanup_queue FROM PUBLIC, anon, authenticated;
REVOKE ALL PRIVILEGES ON SEQUENCE public.file_cleanup_queue_id_seq FROM PUBLIC, anon, authenticated;
