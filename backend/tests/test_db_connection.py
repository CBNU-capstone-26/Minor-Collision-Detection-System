"""Regression tests for production-safe SQLAlchemy connection settings."""

import unittest
from unittest.mock import patch


class DatabaseEngineConfigTests(unittest.TestCase):
    def test_engine_uses_short_connect_timeout_and_recycles_connections(self):
        with patch("app.db_connection.create_engine") as create_engine:
            from app import db_connection

            db_connection.create_database_engine("postgresql+psycopg://example")

        create_engine.assert_called_once_with(
            "postgresql+psycopg://example",
            pool_pre_ping=True,
            pool_recycle=300,
            pool_timeout=5,
            connect_args={
                "connect_timeout": 5,
                "options": "-c statement_timeout=5000",
            },
        )

    def test_mysql_engine_does_not_receive_postgres_statement_option(self):
        with patch("app.db_connection.create_engine") as create_engine:
            from app import db_connection

            db_connection.create_database_engine("mysql+pymysql://example")

        create_engine.assert_called_once_with(
            "mysql+pymysql://example",
            pool_pre_ping=True,
            pool_recycle=300,
            pool_timeout=5,
            connect_args={"connect_timeout": 5},
        )


if __name__ == "__main__":
    unittest.main()
