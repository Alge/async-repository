# tests/create_mysql_tables.py
import logging
import aiomysql
from aiomysql import Pool

# Configure logging for this module
logger = logging.getLogger(__name__)


async def create_entity_table(
    pool: Pool, app_id_field: str = "id", db_id_field: str = "db_id"
):
    """
    Creates the 'entities' table in MySQL for testing the MySQLRepository.

    Uses the specified field names for application ID and database ID (PK).
    Assumes the database ID field is an auto-incrementing integer primary key.
    Assumes the application ID field is a string and should be unique.
    Stores list and dict fields as JSON.

    Args:
        pool: An active aiomysql connection pool.
        app_id_field: The name for the application-level ID column.
        db_id_field: The name for the database's primary key column.
    """
    table_name = "entities"
    logger.info(
        f"Attempting to create MySQL table '{table_name}' with "
        f"app_id='{app_id_field}', db_id='{db_id_field}' (PK)"
    )

    # Use backticks for MySQL identifiers
    quoted_table = f"`{table_name}`"
    quoted_db_id = f"`{db_id_field}`"
    quoted_app_id = f"`{app_id_field}`"

    # Define columns based on the Entity dataclass
    # Primary Key (Database ID)
    columns = [f"{quoted_db_id} INT AUTO_INCREMENT PRIMARY KEY"]

    # Add application ID if different from DB ID
    if app_id_field != db_id_field:
        # VARCHAR(36) is suitable for UUIDs, 255 provides more flexibility
        columns.append(f"{quoted_app_id} VARCHAR(255) NULL")
        unique_index_sql = f"ALTER TABLE {quoted_table} ADD UNIQUE INDEX `idx_{app_id_field}` ({quoted_app_id});"
    else:
        # If app_id_field *is* the db_id_field, it's already defined as the PK.
        # This scenario might require adjusting the PK definition if app_id isn't INT AUTO_INCREMENT.
        # However, based on Entity and MySQLRepo defaults, they are different.
        unique_index_sql = ""
        logger.warning(
            f"app_id_field '{app_id_field}' is the same as db_id_field '{db_id_field}'. "
            f"Ensure the primary key definition matches the expected application ID type."
        )

    # Add remaining columns matching Entity fields
    columns.extend(
        [
            "`name` VARCHAR(255) NOT NULL",
            "`value` INT NOT NULL DEFAULT 100",
            # Store lists/dicts as JSON. Ensure default values are valid JSON strings.
            "`tags` JSON NOT NULL",  # DEFAULT ('["test", "sample"]') - Requires MySQL 8+ or MariaDB 10.2+
            "`active` BOOLEAN NOT NULL DEFAULT TRUE",  # TINYINT(1)
            "`metadata` JSON NOT NULL",  # DEFAULT ('{{}}')
            # DATETIME(6) supports microseconds
            "`created_at` DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)",
            "`updated_at` DATETIME(6) NULL",  # Initially NULL
            "`owner` VARCHAR(255) NULL",
            "`profile` JSON NOT NULL",  # DEFAULT ('{{"emails": []}}')
        ]
    )

    # Construct the CREATE TABLE statement
    # Note: Default values for JSON columns require newer MySQL versions.
    # If using older versions, remove DEFAULT clauses for JSON and handle defaults in application/repo code.
    # The example below includes defaults assuming a modern MySQL/MariaDB version.
    create_sql = f"""
    CREATE TABLE {quoted_table} (
        {',\n        '.join(columns)}
        -- Add JSON defaults here if needed and supported
        -- Example for older versions: remove DEFAULTs above, handle in code.
        -- Example for newer versions (syntax might vary slightly):
        -- , CHECK (JSON_VALID(`tags`)),
        -- , CHECK (JSON_VALID(`metadata`)),
        -- , CHECK (JSON_VALID(`profile`)) -- Optional JSON validation constraint
    );
    """
    # Add default JSON values separately for better compatibility/clarity
    # Using ALTER TABLE is generally safer for adding defaults after creation
    alter_defaults_sql = [
        f"ALTER TABLE {quoted_table} ALTER `tags` SET DEFAULT ('[]'), ALTER `metadata` SET DEFAULT ('{{}}'), ALTER `profile` SET DEFAULT ('{{}}');"
    ]

    drop_sql = f"DROP TABLE IF EXISTS {quoted_table};"

    conn = None
    try:
        conn = await pool.acquire()
        async with conn.cursor() as cursor:
            logger.debug(f"Executing: {drop_sql}")
            await cursor.execute(drop_sql)

            logger.debug(f"Executing: {create_sql}")
            await cursor.execute(create_sql)

            # Apply JSON defaults via ALTER TABLE
            # for alter_sql in alter_defaults_sql:
            #     try:
            #         logger.debug(f"Executing: {alter_sql}")
            #         await cursor.execute(alter_sql)
            #     except aiomysql.OperationalError as e:
            #          # Error 1060: Duplicate column name (means default already set somehow?)
            #          # Error related to JSON default syntax on older versions
            #          logger.warning(f"Could not set JSON default, possibly due to MySQL version or syntax: {e}")

            if unique_index_sql:
                logger.debug(f"Executing: {unique_index_sql}")
                try:
                    await cursor.execute(unique_index_sql)
                except aiomysql.OperationalError as e:
                    # Error 1061: Duplicate key name - Index likely exists from previous run/schema
                    if e.args[0] == 1061:
                        logger.warning(
                            f"Index idx_{app_id_field} likely already exists."
                        )
                    else:
                        raise e  # Re-raise other operational errors

        # No explicit commit needed if pool has autocommit=True (as configured in conftest)
        logger.info(f"MySQL table '{table_name}' created or replaced successfully.")

    except Exception as e:
        # Log the specific SQL that failed if possible
        logger.error(
            f"Error during MySQL table setup for '{table_name}': {e}", exc_info=True
        )
        # No explicit rollback needed if pool has autocommit=True
        raise  # Re-raise the exception to fail the test setup
    finally:
        if conn:
            pool.release(conn)


async def create_tables(pool: Pool):
    """
    Creates all necessary MySQL tables for the tests.

    Args:
        pool: An active aiomysql connection pool.
    """
    logger.info("Creating MySQL test tables...")
    # Call table creation functions for each required table
    # Currently, only the 'entities' table is defined
    await create_entity_table(pool)  # Uses default field names 'id' and 'db_id'
    logger.info("Finished creating MySQL test tables.")
