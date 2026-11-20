BEGIN;
ALTER TABLE public.vehicle_detection_tasks ENABLE ROW LEVEL SECURITY;
REVOKE ALL PRIVILEGES ON TABLE public.vehicle_detection_tasks FROM anon, authenticated;
CREATE POLICY "Deny direct API access" ON public.vehicle_detection_tasks FOR ALL TO anon, authenticated USING (false) WITH CHECK (false);
COMMIT;;
