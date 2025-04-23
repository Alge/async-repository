# tests/create_sqlite_tables.py
import logging
import aiosqlite
import json  # For default JSON values

logger = logging.getLogger(__name__)

# Constants from conftest
ENTITY_TABLE_NAME = "entities"
DEFAULT_APP_ID_FIELD = "id"
DEFAULT_SQLITE_DB_ID_FIELD = "db_id"


async def create_entity_table(
    db_path: str,
    app_id_field: str = DEFAULT_APP_ID_FIELD,
    db_id_field: str = DEFAULT_SQLITE_DB_ID_FIELD,
):
    """
    Creates the 'entities' table in SQLite for testing SQLiteRepository.

    Uses db_id_field (default 'db_id') as INTEGER PRIMARY KEY AUTOINCREMENT.
    Uses app_id_field (default 'id') as TEXT UNIQUE.
    Stores list and dict fields as TEXT (storing JSON strings).

    Args:
        db_path: Path to the SQLite database file.
        app_id_field: The name for the application-level ID column.
        db_id_field: The name for the database's primary key column.
    """
    logger.info(
        f"Ensuring SQLite table '{ENTITY_TABLE_NAME}' exists in '{db_path}' with "
        f"app_id={app_id_field} (TEXT UNIQUE), db_id={db_id_field} (INTEGER PK)"
    )

    # Quote identifiers for safety
    quoted_table = f'"{ENTITY_TABLE_NAME}"'
    quoted_db_id = f'"{db_id_field}"'
    quoted_app_id = f'"{app_id_field}"'

    # Define columns
    columns = [
        f"{quoted_db_id} INTEGER PRIMARY KEY AUTOINCREMENT",  # Standard SQLite PK
        f"{quoted_app_id} TEXT UNIQUE NOT NULL",  # Application ID must be unique and not null
        '"name" TEXT NOT NULL',
        '"value" INTEGER NOT NULL DEFAULT 100',
        # Store JSON as TEXT. Add default values as JSON strings.
        f'"tags" TEXT NOT NULL DEFAULT \'{json.dumps(["test", "sample"])}\'',
        '"active" INTEGER NOT NULL DEFAULT 1',  # Use INTEGER 0/1 for BOOLEAN
        f"\"metadata\" TEXT NOT NULL DEFAULT '{json.dumps({}) }'",
        # Store timestamps as TEXT in ISO format (recommended for SQLite)
        "\"created_at\" TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
        '"updated_at" TEXT',  # Allow NULL
        '"owner" TEXT',  # Allow NULL
        f'"profile" TEXT NOT NULL DEFAULT \'{json.dumps({"emails": []})}\'',
    ]

    # Construct the CREATE TABLE IF NOT EXISTS statement
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {quoted_table} (
        {',\n        '.join(columns)}
    );
    """
    # Separate index creation for clarity, although UNIQUE constraint does this
    # index_sql = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{ENTITY_TABLE_NAME}_{app_id_field} ON {quoted_table}({quoted_app_id});"

    try:
        # Use write-ahead logging for better concurrency
        async with aiosqlite.connect(
            db_path, isolation_level=None
        ) as conn:  # Use autocommit mode
            await conn.execute("PRAGMA journal_mode=WAL;")
            logger.debug(f"Executing: {create_sql}")
            await conn.execute(create_sql)
            # logger.debug(f"Executing: {index_sql}") # Index created by UNIQUE constraint
            # await conn.execute(index_sql)
            await conn.commit()  # Commit DDL changes
        logger.info(f"SQLite table '{ENTITY_TABLE_NAME}' ensured.")
    except Exception as e:
        logger.error(
            f"Error creating SQLite table '{ENTITY_TABLE_NAME}': {e}", exc_info=True
        )
        raise


async def create_tables(db_path: str):
    """Creates all necessary SQLite tables."""
    logger.info(f"Creating SQLite test tables in {db_path}...")
    # Use default field names matching SQLiteRepository defaults
    await create_entity_table(
        db_path,
        app_id_field=DEFAULT_APP_ID_FIELD,
        db_id_field=DEFAULT_SQLITE_DB_ID_FIELD,
    )
    logger.info("Finished creating SQLite test tables.")
