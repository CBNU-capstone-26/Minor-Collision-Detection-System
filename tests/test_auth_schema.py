from sqlalchemy import inspect

from app.db_connection import engine


def test_users_table_has_phone_column():
    """기존 DB에서도 회원가입/로그인에 필요한 User 모델 컬럼이 존재해야 한다."""
    columns = {column["name"] for column in inspect(engine).get_columns("users")}

    assert "phone" in columns
