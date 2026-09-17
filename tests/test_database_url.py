from backend.app.db_connection import normalize_database_url


def test_normalize_database_url_adds_psycopg_driver():
    assert normalize_database_url(
        "postgresql://user:password@db.example.com/app"
    ) == "postgresql+psycopg://user:password@db.example.com/app"


def test_normalize_database_url_keeps_existing_driver_and_mysql():
    assert normalize_database_url(
        "postgresql+psycopg://user:password@db.example.com/app"
    ) == "postgresql+psycopg://user:password@db.example.com/app"
    assert normalize_database_url(
        "mysql+pymysql://root:password@localhost/app"
    ) == "mysql+pymysql://root:password@localhost/app"
