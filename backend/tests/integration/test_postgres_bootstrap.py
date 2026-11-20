"""Opt-in test against a disposable local PostgreSQL database only."""

import os
import unittest
from urllib.parse import urlparse


DATABASE_URL = os.environ.get("SIOT_TEST_POSTGRES_URL", "")


@unittest.skipUnless(DATABASE_URL, "SIOT_TEST_POSTGRES_URL is not set")
class PostgresBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parsed = urlparse(DATABASE_URL.replace("postgresql+psycopg://", "postgresql://", 1))
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or not parsed.path.lstrip("/").startswith("siot_test_"):
            raise RuntimeError("Integration test requires a disposable local siot_test_ database")

        os.environ["DATABASE_URL"] = DATABASE_URL
        from app import main  # noqa: F401 - runs initialization on the disposable DB
        from app.db_connection import Base, engine
        from app.schema_init import create_tables_safely
        cls.engine = engine
        cls.metadata = Base.metadata
        cls.create_tables_safely = staticmethod(create_tables_safely)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def test_new_tables_are_private_and_existing_tables_are_not_modified(self):
        from sqlalchemy import text

        names = set(self.metadata.tables)
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT c.relname, c.relrowsecurity,
                       has_table_privilege('anon', c.oid, 'SELECT'),
                       has_table_privilege('authenticated', c.oid, 'UPDATE')
                FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname='public' AND c.relname = ANY(:names)
            """), {"names": list(names)}).all()
            self.assertEqual({row[0] for row in rows}, names)
            for name, rls, anon_read, authenticated_write in rows:
                self.assertTrue(rls, name)
                self.assertFalse(anon_read, name)
                self.assertFalse(authenticated_write, name)
            conn.execute(text("SELECT 1 FROM public.vehicle_detection_tasks LIMIT 1")).all()
            for role in ('anon', 'authenticated'):
                self.assertFalse(conn.execute(text(
                    "SELECT has_sequence_privilege(:role, 'public.file_cleanup_queue_id_seq', 'USAGE')"
                ), {'role': role}).scalar_one())

        # Existing tables are not silently given a new access policy on restart.
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE public.videos DISABLE ROW LEVEL SECURITY"))
        self.create_tables_safely(self.engine, self.metadata)
        with self.engine.connect() as conn:
            self.assertFalse(conn.execute(text(
                "SELECT relrowsecurity FROM pg_class WHERE oid='public.videos'::regclass"
            )).scalar_one())

    def test_failed_protection_rolls_back_table_creation(self):
        from sqlalchemy import Column, Integer, MetaData, Table, event, inspect

        metadata = MetaData()
        Table("siot_test_rollback", metadata, Column("id", Integer, primary_key=True))

        def fail_protection(_conn, _cursor, statement, _params, _context, _many):
            if "ENABLE ROW LEVEL SECURITY" in statement and "siot_test_rollback" in statement:
                raise RuntimeError("simulated protection failure")

        event.listen(self.engine, "before_cursor_execute", fail_protection)
        try:
            with self.assertRaisesRegex(RuntimeError, "simulated protection failure"):
                self.create_tables_safely(self.engine, metadata)
        finally:
            event.remove(self.engine, "before_cursor_execute", fail_protection)
        self.assertFalse(inspect(self.engine).has_table("siot_test_rollback"))


if __name__ == "__main__":
    unittest.main()
