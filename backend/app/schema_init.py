"""Create new PostgreSQL tables and protect them before they are visible."""

from sqlalchemy import inspect, text


def create_tables_safely(engine, metadata):
    """Protect newly created PostgreSQL tables; leave existing permissions alone."""
    if engine.dialect.name != "postgresql":
        metadata.create_all(bind=engine)
        return

    with engine.begin() as connection:
        inspector = inspect(connection)
        existing = {
            table.key for table in metadata.sorted_tables
            if inspector.has_table(table.name, schema=table.schema)
        }
        metadata.create_all(bind=connection)

        roles = set(connection.execute(text(
            "SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated')"
        )).scalars())
        preparer = connection.dialect.identifier_preparer
        public_roles = [role for role in ("anon", "authenticated") if role in roles]
        for table in metadata.sorted_tables:
            if table.key in existing:
                continue
            table_name = preparer.format_table(table)
            connection.execute(text(f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY"))
            connection.execute(text(f"REVOKE ALL PRIVILEGES ON TABLE {table_name} FROM PUBLIC"))
            if public_roles:
                role_names = ", ".join(preparer.quote_identifier(role) for role in public_roles)
                connection.execute(text(f"REVOKE ALL PRIVILEGES ON TABLE {table_name} FROM {role_names}"))
            if table.name == "file_cleanup_queue":
                sequence = connection.execute(text(
                    "SELECT pg_get_serial_sequence(:table_name, 'id')"
                ), {"table_name": table_name}).scalar_one_or_none()
                if sequence:
                    connection.execute(text(f"REVOKE ALL PRIVILEGES ON SEQUENCE {sequence} FROM PUBLIC"))
                    if public_roles:
                        connection.execute(text(f"REVOKE ALL PRIVILEGES ON SEQUENCE {sequence} FROM {role_names}"))
