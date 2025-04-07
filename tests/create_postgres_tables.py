from asyncpg import Pool


async def create_entity_table(
    pool: Pool, app_id_field: str = "id", db_id_field: str = "_id"
):
    """
    Create a test table for entities with configurable ID fields.

    Args:
        pool: Database connection pool
        app_id_field: The application ID field name (default: "id")
        db_id_field: The database ID field name (default: "_id")
    """
    print(
        f"Creating test table for 'entities' with app_id_field={app_id_field}, db_id_field={db_id_field}"
    )

    # Determine which field is the primary key
    primary_key_field = db_id_field

    # Prepare columns list starting with the ID fields
    columns = []

    # Always add the database ID field as primary key
    columns.append(f"{db_id_field} TEXT PRIMARY KEY")

    # Add application ID field if it's different from db_id_field
    if app_id_field != db_id_field:
        columns.append(f"{app_id_field} TEXT")

    # Add the rest of the columns
    columns.extend(
        [
            "name TEXT NOT NULL",
            "value INTEGER NOT NULL DEFAULT 100",
            "tags TEXT[] NOT NULL DEFAULT ARRAY['test', 'sample']",
            "active BOOLEAN NOT NULL DEFAULT TRUE",
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
            "created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()",
            "updated_at TIMESTAMP WITHOUT TIME ZONE",
            "owner TEXT",
            "profile JSONB NOT NULL DEFAULT '{\"emails\": []}'::jsonb",
        ]
    )

    # Create unique index on app_id_field if it's different from db_id_field
    index_statement = ""
    if app_id_field != db_id_field:
        index_statement = (
            f"\nCREATE UNIQUE INDEX {app_id_field}_idx ON entities({app_id_field});"
        )

    # Construct the SQL statement
    sql = f"""
    DROP TABLE IF EXISTS entities;

    CREATE TABLE entities (
        {',\n        '.join(columns)}
    );{index_statement}
    """

    async with pool.acquire() as conn:
        await conn.execute(sql)


async def create_tables(pool: Pool):
    await create_entity_table(pool)
