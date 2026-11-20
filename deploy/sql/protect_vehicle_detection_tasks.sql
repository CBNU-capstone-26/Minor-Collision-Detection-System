-- 검토용: 다른 애플리케이션 테이블과 같이 Supabase Data API 직접 접근을 차단한다.
-- 실행 전 백엔드가 postgres/service_role 등 RLS를 우회하는 계정을 쓰는지 확인한다.
BEGIN;

ALTER TABLE public.vehicle_detection_tasks ENABLE ROW LEVEL SECURITY;
REVOKE ALL PRIVILEGES ON TABLE public.vehicle_detection_tasks FROM anon, authenticated;
CREATE POLICY "Deny direct API access"
    ON public.vehicle_detection_tasks
    FOR ALL TO anon, authenticated
    USING (false)
    WITH CHECK (false);

COMMIT;
