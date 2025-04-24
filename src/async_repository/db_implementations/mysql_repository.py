# src/async_repository/backends/mysql_repository.py


import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import is_dataclass
from datetime import date, datetime, timezone
from logging import LoggerAdapter
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import asyncio

# --- aiomysql Driver Import ---
import aiomysql

# --- Framework Imports ---
from async_repository.base.exceptions import (
    KeyAlreadyExistsException,
    ObjectNotFoundException,
)
from async_repository.base.interfaces import Repository
from async_repository.base.model_validator import (
    InvalidPathError,
    ModelValidator,
    _is_none_type,
)
from async_repository.base.query import (
    QueryExpression,
    QueryFilter,
    QueryLogical,
    QueryOperator,
    QueryOptions,
)
from async_repository.base.update import (
    IncrementOperation,
    MaxOperation,
    MinOperation,
    MultiplyOperation,
    PopOperation,
    PullOperation,
    PushOperation,
    SetOperation,
    UnsetOperation,
    Update,
    UpdateOperation,
)
from async_repository.base.utils import prepare_for_storage

# --- Type Variables ---
T = TypeVar("T")
# MySQL pool and cursor types
DB_POOL_TYPE = aiomysql.Pool
DB_CURSOR_TYPE = aiomysql.DictCursor

base_logger = logging.getLogger("async_repository.backends.mysql_repository")


class MySQLRepository(Repository[T], Generic[T]):
    """
    MySQL repository implementation using aiomysql.

    Requires an aiomysql.Pool during initialization and handles connection
    acquisition/release internally.

    Assumptions:
    - Table exists with the specified primary key field
    - Primary key values are strings (TEXT/VARCHAR)
    - Entity IDs are managed as defined in the constructor parameters
    - Complex types (nested objects, arrays) are stored as JSON
    """

    # --- Initialization ---
    def __init__(
            self,
            db_pool: DB_POOL_TYPE,
            table_name: str,
            entity_type: Type[T],
            app_id_field: str = "id",
            db_id_field: Optional[str] = None,
            db_name: str = None,  # MySQL uses 'database' instead of 'schema'
    ):
        """
        Initialize the MySQL repository with an existing connection pool.

        Args:
            db_pool: An active aiomysql.Pool object.
            table_name: The name of the database table.
            entity_type: The Python class representing the entity.
            app_id_field: Attribute name for the application ID.
            db_id_field: Database PK column name (defaults to app_id_field).
            db_name: The MySQL database name (if different from connection default).
        """
        if not isinstance(db_pool, aiomysql.Pool):
            raise TypeError("db_pool must be an instance of aiomysql.Pool")

        self._pool = db_pool
        self._table_name = table_name
        self._db_name = db_name
        self._entity_type = entity_type
        self._app_id_field = app_id_field
        self._db_id_field = (
            db_id_field if db_id_field is not None else app_id_field
        )

        # MySQL uses backticks for identifiers
        if self._db_name:
            self._qualified_table_name = f"`{self._db_name}`.`{self._table_name}`"
        else:
            self._qualified_table_name = f"`{self._table_name}`"

        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}[{entity_type.__name__}]"
        )
        try:
            self._validator = ModelValidator(entity_type)
        except Exception:
            self._validator = None
            self._logger.warning(
                f"Could not initialize ModelValidator for "
                f"{entity_type.__name__}. Proceeding without validation."
            )

        self._logger.info(
            f"Repository instance created (Pool) for "
            f"{entity_type.__name__} using table "
            f"'{self._qualified_table_name}' (App ID Field: "
            f"'{self._app_id_field}', DB PK Field: '{self._db_id_field}')."
        )

    # --- Abstract Property Implementations ---
    @property
    def entity_type(self) -> Type[T]:
        return self._entity_type

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    # --- Connection/Session Management ---
    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[
        Tuple[aiomysql.Connection, aiomysql.DictCursor], None]:
        """
        Acquire connection from the pool and create a DictCursor.
        Handles connection release. Uses DictCursor for easier record management.
        """
        conn = None
        cursor = None
        try:
            # Acquire connection from the pool
            conn = await self._pool.acquire()
            self._logger.debug(f"Acquired connection from pool.")

            # Create a DictCursor for easier mapping to/from entities
            cursor = await conn.cursor(aiomysql.DictCursor)

            yield conn, cursor  # Provide both connection and cursor

        except Exception as e:
            self._logger.error(
                f"Error during connection handling: {e}", exc_info=True
            )
            # Let specific db operation handlers wrap with _handle_db_error
            raise
        finally:
            # Close cursor and release connection back to the pool if acquired
            if cursor:
                await cursor.close()
            if conn:
                try:
                    self._pool.release(conn)
                    self._logger.debug(
                        f"Released connection back to pool."
                    )
                except Exception as release_error:
                    self._logger.error(
                        f"Error releasing connection: {release_error}",
                        exc_info=True,
                    )

    # --- Schema and Index Implementation ---
    async def check_schema(self, logger: LoggerAdapter) -> bool:
        """Check if the table exists in the specified database."""
        logger.info(f"Checking schema for '{self._qualified_table_name}'...")

        # Handle case with or without schema name
        if self._db_name:
            sql = """
                SELECT COUNT(*) as table_exists 
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            """
            params = (self._db_name, self._table_name)
        else:
            sql = """
                SELECT COUNT(*) as table_exists 
                FROM information_schema.tables 
                WHERE table_name = %s
            """
            params = (self._table_name,)

        try:
            async with self._get_session() as (conn, cursor):
                await cursor.execute(sql, params)
                result = await cursor.fetchone()
                exists = result['table_exists'] > 0

            if exists:
                logger.info(
                    f"Schema check PASSED for '{self._qualified_table_name}'."
                )
                return True
            else:
                logger.warning(
                    "Schema check FAILED: Table "
                    f"'{self._qualified_table_name}' not found."
                )
                return False

        except Exception as e:
            logger.error(
                "Error during schema check for "
                f"'{self._qualified_table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(
                e, f"checking schema for {self._qualified_table_name}"
            )
            return False  # pragma: no cover

    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """Check if essential indexes (PK on db_id_field) exist."""
        logger.info(
            f"Checking essential indexes (PK on '{self.db_id_field}') for "
            f"'{self._qualified_table_name}'..."
        )

        # Handle case with or without schema name
        if self._db_name:
            sql = """
                SELECT COUNT(*) as has_primary
                FROM information_schema.table_constraints 
                WHERE table_schema = %s 
                AND table_name = %s
                AND constraint_type = 'PRIMARY KEY'
            """
            params = (self._db_name, self._table_name)
        else:
            sql = """
                SELECT COUNT(*) as has_primary
                FROM information_schema.table_constraints 
                WHERE table_name = %s
                AND constraint_type = 'PRIMARY KEY'
            """
            params = (self._table_name,)

        try:
            async with self._get_session() as (conn, cursor):
                # First check if the table has a primary key
                await cursor.execute(sql, params)
                result = await cursor.fetchone()
                has_primary = result['has_primary'] > 0

                if not has_primary:
                    logger.warning(
                        f"Index check FAILED for '{self._qualified_table_name}': "
                        "No primary key defined."
                    )
                    return False

                # Then check if the primary key includes our db_id_field
                if self._db_name:
                    key_col_sql = """
                        SELECT column_name 
                        FROM information_schema.key_column_usage
                        WHERE table_schema = %s 
                        AND table_name = %s
                        AND constraint_name = 'PRIMARY'
                    """
                    key_params = (self._db_name, self._table_name)
                else:
                    key_col_sql = """
                        SELECT column_name 
                        FROM information_schema.key_column_usage
                        WHERE table_name = %s
                        AND constraint_name = 'PRIMARY'
                    """
                    key_params = (self._table_name,)

                await cursor.execute(key_col_sql, key_params)
                pk_cols = await cursor.fetchall()

                pk_col_found = False
                for record in pk_cols:
                    if record["column_name"] == self.db_id_field:
                        pk_col_found = True
                        break

            if pk_col_found:
                logger.info(
                    f"Index check PASSED for '{self._qualified_table_name}': "
                    f"PK includes '{self.db_id_field}'."
                )
                return True
            else:
                logger.warning(
                    f"Index check FAILED for '{self._qualified_table_name}': "
                    f"PK does not include '{self.db_id_field}'."
                )
                return False

        except Exception as e:
            logger.error(
                f"Error checking indexes for "
                f"'{self._qualified_table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(
                e, f"checking indexes for {self._qualified_table_name}"
            )
            return False  # pragma: no cover

    async def create_schema(self, logger: LoggerAdapter) -> None:
        """
        Create the table if it doesn't exist, inferring columns from type hints.
        Handles separate app_id_field and db_id_field configuration.
        """
        logger.info(
            "Attempting to create schema (table "
            f"'{self._qualified_table_name}')..."
        )

        pk_col = self.db_id_field
        # MySQL typical PK column type
        pk_col_type = "VARCHAR(255)"
        cols_def = [f"`{pk_col}` {pk_col_type} PRIMARY KEY NOT NULL"]
        processed_fields = {pk_col}

        try:
            hints = get_type_hints(self.entity_type)
        except Exception as e:
            logger.warning(f"Could not get type hints for schema creation: {e}")
            hints = {}

        # 1. Explicitly handle the app_id_field if it's different from PK
        if self.app_id_field != self.db_id_field:
            app_id_hint = hints.get(self.app_id_field)
            if app_id_hint:
                app_id_col_type = "VARCHAR(255)"  # Default assumption
                is_optional = False
                origin = get_origin(app_id_hint)
                if origin is Union and _is_none_type(get_args(app_id_hint)[-1]):
                    is_optional = True
                constraint = "" if is_optional else " NOT NULL"
                # Add UNIQUE constraint for application ID
                constraint += " UNIQUE"

                cols_def.append(
                    f"`{self.app_id_field}` {app_id_col_type}{constraint}"
                )
                processed_fields.add(self.app_id_field)
                logger.debug(
                    f"Added separate column for app_id_field: '{self.app_id_field}'")
            else:
                logger.warning(
                    f"app_id_field '{self.app_id_field}' differs from "
                    f"db_id_field '{self.db_id_field}' but not found in "
                    f"model hints. Column not created automatically."
                )

        # 2. Loop through remaining hints for other columns
        for name, hint in hints.items():
            if name in processed_fields:  # Skip PK and already added app_id
                continue
            if name.startswith("_"):
                continue

            # --- Type inference logic adapted for MySQL ---
            col_type = "VARCHAR(255)"  # Default type
            is_optional = False
            origin = get_origin(hint)
            actual_type = hint
            if origin is Union and hint and _is_none_type(get_args(hint)[-1]):
                is_optional = True
                actual_type = get_args(hint)[0]
                origin = get_origin(actual_type)

            if origin is Union:
                col_type = "JSON"  # MySQL uses JSON, not JSONB
            elif actual_type is int:
                col_type = "BIGINT"
            elif actual_type is float:
                col_type = "DOUBLE"  # MySQL uses DOUBLE, not DOUBLE PRECISION
            elif actual_type is bool:
                col_type = "TINYINT(1)"  # MySQL stores booleans as TINYINT(1)
            elif actual_type is bytes:
                col_type = "BLOB"
            elif actual_type is datetime:
                col_type = "DATETIME"  # MySQL doesn't include timezone by default, we'll handle it in deserialization
            elif actual_type is date:
                col_type = "DATE"
            elif actual_type in (str, Any):
                col_type = "TEXT"  # Or VARCHAR(n) if size is known
            elif origin in (list, List, dict, Dict, set, Set, tuple, Tuple, Mapping) or \
                    (isinstance(actual_type, type) and
                     (issubclass(actual_type, (list, dict, set, tuple)) or
                      is_dataclass(actual_type) or hasattr(actual_type, 'model_dump'))):
                col_type = "JSON"
            # --- End type inference ---

            constraint = "" if is_optional else " NOT NULL"
            cols_def.append(f"`{name}` {col_type}{constraint}")
            processed_fields.add(name)

        # Create schema if specified
        if self._db_name:
            create_schema_sql = f"CREATE DATABASE IF NOT EXISTS `{self._db_name}`;"
        else:
            create_schema_sql = None

        create_table_sql = (
            f"CREATE TABLE IF NOT EXISTS {self._qualified_table_name} "
            f"({', '.join(cols_def)})"
        )
        logger.debug(f"Table creation SQL: {create_table_sql}")

        try:
            async with self._get_session() as (conn, cursor):
                if create_schema_sql:
                    await cursor.execute(create_schema_sql)
                    logger.info(f"Ensured database '{self._db_name}' exists.")

                await cursor.execute(create_table_sql)
                await conn.commit()

            logger.info(
                "Schema creation/verification complete for "
                f"'{self._qualified_table_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to create schema for "
                f"'{self._qualified_table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(
                e, f"creating schema for {self._qualified_table_name}"
            )

    async def create_indexes(self, logger: LoggerAdapter) -> None:
        """Create potentially useful indexes (UNIQUE, Fulltext for JSON)."""
        logger.info(
            "Attempting to create indexes for "
            f"'{self._qualified_table_name}'..."
        )
        indexes_to_create: List[str] = []

        # Check for distinct app_id_field needing a unique index
        if self.app_id_field != self.db_id_field:
            field_exists_in_hints = False
            try:
                hints = get_type_hints(self.entity_type)
                field_exists_in_hints = self.app_id_field in hints
            except Exception:
                pass  # Ignore hint errors, proceed if field names differ

            if field_exists_in_hints:
                idx_name = f"idx_{self._table_name}_{self.app_id_field}_unique"
                sql = (
                    f"CREATE UNIQUE INDEX IF NOT EXISTS `{idx_name}` ON "
                    f"{self._qualified_table_name}(`{self.app_id_field}`);"
                )
                indexes_to_create.append(sql)
                logger.info(
                    f"Defining unique index on separate app_id_field "
                    f"'{self.app_id_field}'."
                )
            else:
                logger.info(
                    f"Skipping unique index on app_id_field '{self.app_id_field}' as it's not found in model hints.")

        # Identify JSON columns for potential indexing
        # Note: MySQL's JSON indexing differs from PostgreSQL's GIN
        json_cols_for_index: List[str] = []
        try:
            hints = get_type_hints(self.entity_type)
        except Exception as e:
            logger.warning(f"Could not get hints for JSON indexing: {e}")
            hints = {}

        for name, hint in hints.items():
            if name == self.db_id_field or name == self.app_id_field:
                continue  # Skip PK and app_id
            if name.startswith("_"):
                continue

            actual_type = hint
            origin = get_origin(hint)
            if origin is Union and hint and _is_none_type(get_args(hint)[-1]):
                actual_type = get_args(hint)[0]
                origin = get_origin(actual_type)

            if origin is Union or origin in (list, List, dict, Dict, set, Set, tuple,
                                             Tuple, Mapping) or \
                    (isinstance(actual_type, type) and (issubclass(actual_type,
                                                                   (list, dict, set,
                                                                    tuple)) or is_dataclass(
                        actual_type) or hasattr(actual_type, 'model_dump'))):
                json_cols_for_index.append(name)

        # MySQL doesn't have GIN indexes for JSON
        # Instead, we can create functional indexes on JSON fields
        # But this is more complex and specific to the use case
        # For each JSON column, we'll just log that we're skipping it
        for col_name in json_cols_for_index:
            logger.info(
                f"JSON column '{col_name}' identified. MySQL requires functional indexes for JSON fields.")
            # We don't automatically create indexes for JSON here as it requires specific paths

        if not indexes_to_create:
            logger.info(
                f"No explicit indexes defined for creation on "
                f"'{self._qualified_table_name}'."
            )
            return

        try:
            async with self._get_session() as (conn, cursor):
                for sql in indexes_to_create:
                    logger.debug(f"Executing index SQL: {sql}")
                    await cursor.execute(sql)
                await conn.commit()

            logger.info(
                "Explicit index creation/verification complete for "
                f"'{self._qualified_table_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to create indexes for "
                f"'{self._qualified_table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(
                e, f"creating indexes for {self._qualified_table_name}"
            )

    # --- Core CRUD Method Implementations ---
    async def get(
            self,
            id: str,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            use_db_id: bool = False,
    ) -> T:
        """Retrieve an entity by its application or database ID."""
        field_to_match = self.db_id_field if use_db_id else self.app_id_field
        id_type_str = (
            "database (PK)" if use_db_id else f"application ('{field_to_match}')"
        )
        logger.debug(
            f"Getting {self.entity_type.__name__} by {id_type_str}='{id}' "
            f"using column '{field_to_match}'"
        )

        query = (
            f"SELECT * FROM {self._qualified_table_name} "
            f"WHERE `{field_to_match}` = %s LIMIT 1"
        )
        params = (id,)

        try:
            async with self._get_session() as (conn, cursor):
                # Set timeout if provided
                if timeout:
                    await conn.ping(reconnect=True)  # Ensure connection is alive
                    # MySQL doesn't have a direct per-query timeout setting
                    # We could use SET SESSION MAX_EXECUTION_TIME but it's MySQL 5.7.8+ specific

                await cursor.execute(query, params)
                record_data = await cursor.fetchone()

            if record_data is None:
                logger.warning(f"{self.entity_type.__name__} '{id}' not found.")
                raise ObjectNotFoundException(
                    f"{self.entity_type.__name__} with ID '{id}' not found."
                )

            entity = self._deserialize_record(record_data)
            logger.info(f"Retrieved {self.entity_type.__name__} '{id}'.")
            return entity

        except ObjectNotFoundException:
            raise
        except Exception as e:
            logger.error(
                f"Error during get operation (id={id}, use_db_id={use_db_id}):"
                f" {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"getting entity ID {id}")
            raise  # pragma: no cover

    async def get_by_db_id(
            self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        """Retrieve an entity specifically by its database primary key ID."""
        logger.debug(
            f"Getting {self.entity_type.__name__} by DB ID '{db_id}' "
            f"(using PK column '{self.db_id_field}')"
        )
        return await self.get(
            id=str(db_id), logger=logger, timeout=timeout, use_db_id=True
        )

    async def store(
            self,
            entity: T,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            generate_app_id: bool = True,
            return_value: bool = False,
    ) -> Optional[T]:
        """Store a new entity, generating ID if needed."""
        self.validate_entity(entity)
        app_id = getattr(entity, self.app_id_field, None)

        if generate_app_id and app_id is None:
            app_id = self.id_generator()
            setattr(entity, self.app_id_field, app_id)
            logger.debug(f"Generated ID '{app_id}' for new {entity}.")
        elif app_id is None:
            raise ValueError("Entity missing ID field and generate_app_id=False.")

        db_id_value = getattr(entity, self.app_id_field)
        logger.debug(
            f"Storing new {self.entity_type.__name__} with "
            f"{self.app_id_field}='{app_id}' (PK '{self.db_id_field}')"
        )
        try:
            db_data = self._serialize_entity(entity)
            db_data[self.db_id_field] = db_id_value

            cols = list(db_data.keys())
            cols_clause = ", ".join(f"`{k}`" for k in cols)
            placeholders = ", ".join(["%s"] * len(cols))
            query = (
                f"INSERT INTO {self._qualified_table_name} ({cols_clause}) "
                f"VALUES ({placeholders})"
            )
            params = tuple(db_data[k] for k in cols)

            async with self._get_session() as (conn, cursor):
                if timeout:
                    await conn.ping(reconnect=True)  # Ensure connection is alive

                await cursor.execute(query, params)
                await conn.commit()

            logger.info(
                f"Stored new {self.entity_type.__name__} ID '{app_id}'."
            )
            return entity if return_value else None

        except Exception as e:
            logger.error(
                f"Error storing entity (app_id {app_id}): {e}", exc_info=True
            )
            # Check for duplicate key error
            if isinstance(e, aiomysql.IntegrityError) and "Duplicate entry" in str(e):
                logger.warning(
                    f"Failed to store {self.entity_type.__name__} PK "
                    f"'{db_id_value}': {e}"
                )
                raise KeyAlreadyExistsException(
                    f"Entity with ID '{app_id}' already exists."
                ) from e
            else:
                self._handle_db_error(e, f"storing entity app_id {app_id}")
            raise  # Should not be reached

    async def upsert(
            self,
            entity: T,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            generate_app_id: bool = True,
    ) -> None:
        """Insert or update an entity based on its ID."""
        self.validate_entity(entity)
        app_id = getattr(entity, self.app_id_field, None)

        if generate_app_id and app_id is None:
            app_id = self.id_generator()
            setattr(entity, self.app_id_field, app_id)
            logger.debug(f"Generated potential ID '{app_id}' for upsert.")
        elif app_id is None:
            raise ValueError("Entity for upsert must have ID.")

        db_id_value = getattr(entity, self.app_id_field)
        logger.debug(
            f"Upserting {self.entity_type.__name__} with "
            f"{self.app_id_field}='{app_id}' (PK '{self.db_id_field}')"
        )
        try:
            db_data = self._serialize_entity(entity)
            db_data[self.db_id_field] = db_id_value

            cols = list(db_data.keys())
            cols_clause = ", ".join(f"`{k}`" for k in cols)
            placeholders = ", ".join(["%s"] * len(cols))

            # Use MySQL's ON DUPLICATE KEY UPDATE
            update_cols = [k for k in cols if k != self.db_id_field]
            if not update_cols:
                # If no other columns to update, just try insert
                query = (
                    f"INSERT IGNORE INTO {self._qualified_table_name} ({cols_clause}) "
                    f"VALUES ({placeholders})"
                )
            else:
                # Use ON DUPLICATE KEY UPDATE with column=VALUES(column)
                set_clause = ", ".join(
                    f"`{k}` = VALUES(`{k}`)" for k in update_cols
                )
                query = (
                    f"INSERT INTO {self._qualified_table_name} ({cols_clause}) "
                    f"VALUES ({placeholders}) "
                    f"ON DUPLICATE KEY UPDATE {set_clause}"
                )

            params = tuple(db_data[k] for k in cols)

            async with self._get_session() as (conn, cursor):
                if timeout:
                    await conn.ping(reconnect=True)  # Ensure connection is alive

                await cursor.execute(query, params)
                await conn.commit()

            logger.info(
                f"Upserted {self.entity_type.__name__} ID '{app_id}'."
            )
        except Exception as e:
            logger.error(
                f"Error upserting entity (app_id {app_id}): {e}", exc_info=True
            )
            self._handle_db_error(e, f"upserting entity app_id {app_id}")
            raise  # Should not be reached


    async def update_one(
            self,
            options: QueryOptions,
            update: Update,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            return_value: bool = False,
    ) -> Optional[T]:
        """
        Update single entity matching criteria.

        In MySQL, we use a transaction to ensure only one row is updated.
        If return_value is True, we fetch the updated entity after update.
        """
        logger.debug(
            f"Updating one {self.entity_type.__name__} matching: {options!r} "
            f"with update: {update!r}"
        )
        if not update:
            logger.warning("update_one called with empty update operations.")
            if return_value:
                # Find and return the existing entity if requested
                return await self.find_one(logger=logger, options=options)
            return None
        if not options.expression:
            raise ValueError(
                "QueryOptions must include an 'expression' for update_one."
            )

        row_id = None  # Initialize row_id outside the transaction try block

        try:
            # Prepare options for finding the single target row ID
            find_options = options.copy()
            find_options.limit = 1
            find_options.offset = find_options.offset or 0

            # Translate query and update parts separately
            query_parts = self._translate_query_options(
                find_options, include_sorting_pagination=True
            )
            update_parts = self._translate_update(update)

            if not update_parts.get("set"):
                logger.warning("Update resulted in no SET clauses.")
                if return_value:
                    # Find and return the existing entity if requested
                    return await self.find_one(logger=logger, options=options)
                return None

            # MySQL uses %s placeholders for aiomysql
            where_clause = query_parts.get("where", "")
            where_params = query_parts.get("params", [])

            order_clause = (
                f"ORDER BY {query_parts['order']}" if query_parts.get("order") else ""
            )

            limit_clause = (
                f"LIMIT {query_parts['limit']}"
                if query_parts.get("limit", -1) >= 0 else ""
            )

            offset_clause = (
                f"OFFSET {query_parts['offset']}"
                if query_parts.get("offset", 0) > 0 else ""
            )

            # MySQL doesn't support CTE in the same way as PostgreSQL
            # We'll use a transaction with FOR UPDATE to lock the row
            set_clause = update_parts["set"]
            set_params = update_parts["params"]

            # Final SQL statements for the transaction
            select_for_update_sql = f"""
                SELECT `{self.db_id_field}`
                FROM {self._qualified_table_name}
                {where_clause}
                {order_clause}
                {limit_clause}
                {offset_clause}
                FOR UPDATE
            """

            update_sql = f"""
                UPDATE {self._qualified_table_name}
                SET {set_clause}
                WHERE `{self.db_id_field}` = %s
            """

            async with self._get_session() as (conn, cursor):
                # Start transaction
                await conn.begin()
                try:
                    # First, select and lock the row we want to update
                    await cursor.execute(select_for_update_sql, where_params)
                    row_to_update = await cursor.fetchone()

                    if not row_to_update:
                        # No rows found, rollback and raise exception
                        await conn.rollback()
                        logger.warning(
                            f"Update target matching criteria was not found: "
                            f"{options.expression!r}"
                        )
                        raise ObjectNotFoundException(
                            f"No {self.entity_type.__name__} found matching "
                            f"criteria for update: {options.expression!r}"
                        )

                    # Get the ID to use in the update - row_id is now guaranteed to be set
                    row_id = row_to_update[self.db_id_field]

                    # Now update the specific row we locked
                    update_params = set_params + [row_id]
                    await cursor.execute(update_sql, update_params)

                    updated_count = cursor.rowcount

                    # Check if the update resulted in changes.
                    # A rowcount of 0 here is *not* an error if the row was found by
                    # SELECT FOR UPDATE. It likely means a conditional update
                    # (like $min/$max) didn't meet its condition to change the value.
                    if updated_count == 0:
                        logger.info(
                            f"Update statement executed for ID '{row_id}' but resulted "
                            f"in 0 rows changed (e.g., conditional update like "
                            f"$min/$max might not have met criteria)."
                        )
                    elif updated_count > 1:
                        # This should ideally not happen when updating by specific PK
                        await conn.rollback()
                        logger.error(
                            f"Update unexpectedly affected {updated_count} rows when "
                            f"updating by specific ID '{row_id}'."
                        )
                        raise RuntimeError(
                            f"Update modified {updated_count} rows for ID '{row_id}'."
                        )
                    # else: updated_count == 1, which is the expected case for a successful change.

                    # If caller wants the updated entity, fetch it (regardless of updated_count)
                    updated_entity = None
                    if return_value:
                        # Fetch the potentially updated (or unchanged) row
                        fetch_sql = f"""
                            SELECT * FROM {self._qualified_table_name}
                            WHERE `{self.db_id_field}` = %s
                        """
                        await cursor.execute(fetch_sql, [row_id])
                        updated_record = await cursor.fetchone()

                        if updated_record:
                            updated_entity = self._deserialize_record(updated_record)
                        else:
                            # This *would* be a critical error if the row vanished after lock
                            await conn.rollback()
                            logger.error(
                                f"CRITICAL: Row with ID '{row_id}' vanished after "
                                f"successful SELECT FOR UPDATE and UPDATE execution."
                            )
                            raise RuntimeError(
                                f"Failed to re-fetch row with ID '{row_id}' after update."
                            )

                    # Commit the transaction
                    await conn.commit()

                    logger.info(
                        f"Updated one {self.entity_type.__name__} with ID '{row_id}' (changed rows: {updated_count})."
                    )

                    if return_value:
                        return updated_entity
                    return None

                except Exception as e:
                    # Rollback on any error within the transaction block
                    await conn.rollback()
                    # Use a conditional log message since row_id might be None if error occurred before assignment
                    log_id_str = f"for potential ID '{row_id}'" if row_id else "before ID was determined"
                    logger.error(
                        f"Error during transactional update {log_id_str} (rolled back): {e}",
                        exc_info=True,
                    )
                    # Handle specific errors or re-raise
                    if isinstance(e, ObjectNotFoundException):
                        raise  # Re-raise the expected exception
                    self._handle_db_error(e, f"updating entity during transaction {log_id_str}")
                    raise  # Should not be reached if _handle_db_error raises

        except ObjectNotFoundException:
            # Catch specific exception raised outside transaction or re-raised
            logger.warning(
                f"ObjectNotFoundException during update_one for filter: "
                f"{options.expression!r}"
            )
            raise  # Propagate ObjectNotFoundException
        except Exception as e:
            # Catch other DB errors or translation errors outside transaction
            logger.error(
                f"Error setting up update_one (filter: {options.expression!r}): {e}",
                exc_info=True
            )
            self._handle_db_error(e, f"setting up update_one for filter {options.expression!r}")
            raise  # Should not be reached if _handle_db_error raises


    async def update_many(
            self,
            options: QueryOptions,
            update: Update,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
    ) -> int:
        """Update multiple entities matching the criteria."""
        logger.debug(
            f"Updating many {self.entity_type.__name__} matching: {options!r} "
            f"with update: {update!r}"
        )
        if not update:
            logger.warning("update_many called with empty update.")
            return 0
        if not options.expression:
            raise ValueError(
                "QueryOptions must include an 'expression' for update_many."
            )

        if options.limit is not None or options.offset is not None or options.sort_by:
            logger.warning("limit/offset/sort_by ignored for update_many.")

        try:
            find_parts = self._translate_query_options(options, False)
            update_parts = self._translate_update(update)

            where_clause = find_parts.get("where", "")
            where_params = find_parts.get("params", [])
            set_clause = update_parts.get("set", "")
            set_params = update_parts.get("params", [])

            # Safety check for empty where clause
            if not where_clause or where_clause == "WHERE 1=1":
                raise ValueError(
                    "Cannot update_many without a specific filter expression "
                    "(safety check)."
                )
            if not set_clause:
                logger.warning("Update resulted in no SET clauses.")
                return 0

            sql = (
                f"UPDATE {self._qualified_table_name} "
                f"SET {set_clause} {where_clause}"
            )
            all_params = set_params + where_params

            logger.debug(f"Executing update_many: SQL='{sql}', Params={all_params}")
            async with self._get_session() as (conn, cursor):
                await cursor.execute(sql, all_params)
                affected_count = cursor.rowcount
                await conn.commit()

            logger.info(
                f"Updated {affected_count} {self.entity_type.__name__}(s)."
            )
            return affected_count

        except ValueError as e:
            # Handle the specific ValueError raised for missing expression
            logger.error(f"ValueError during update_many: {e}")
            raise  # Re-raise ValueError after logging
        except Exception as e:
            logger.error(f"Error updating many entities: {e}", exc_info=True)
            self._handle_db_error(e, "updating many entities")
            raise  # pragma: no cover

    async def delete_many(
            self,
            options: QueryOptions,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
    ) -> int:
        """Delete multiple entities matching the criteria."""
        logger.debug(
            f"Deleting many {self.entity_type.__name__}(s) matching: {options!r}"
        )
        if not options.expression:
            raise ValueError(
                "QueryOptions must include an 'expression' for delete_many."
            )

        try:
            find_parts = self._translate_query_options(options, False)
            where_clause = find_parts.get("where", "")
            where_params = find_parts.get("params", [])

            # Safety check for empty where clause
            if not where_clause or where_clause == "WHERE 1=1":
                raise ValueError(
                    "Cannot delete_many without a specific filter expression "
                    "(safety check)."
                )

            sql = f"DELETE FROM {self._qualified_table_name} {where_clause}"

            logger.debug(f"Executing delete_many: SQL='{sql}', Params={where_params}")
            async with self._get_session() as (conn, cursor):
                await cursor.execute(sql, where_params)
                affected_count = cursor.rowcount
                await conn.commit()

            logger.info(
                f"Deleted {affected_count} {self.entity_type.__name__}(s)."
            )
            return affected_count

        except ValueError as e:
            # Handle the specific ValueError raised for missing expression
            logger.error(f"ValueError during delete_many: {e}")
            raise  # Re-raise ValueError after logging
        except Exception as e:
            logger.error(f"Error deleting many entities: {e}", exc_info=True)
            self._handle_db_error(e, "deleting many entities")
            raise  # pragma: no cover

    async def list(
            self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        """List entities matching criteria, with sorting and pagination."""
        effective_options = options or QueryOptions()
        logger.debug(f"Listing {self.entity_type.__name__}(s): {options!r}")

        try:
            query_parts = self._translate_query_options(effective_options, True)
            where_clause = query_parts.get("where", "")
            where_params = query_parts.get("params", [])

            order_clause = (
                f"ORDER BY {query_parts['order']}" if query_parts.get("order") else ""
            )

            limit_clause = (
                f"LIMIT {query_parts['limit']}"
                if query_parts.get("limit", -1) >= 0 else ""
            )

            offset_clause = (
                f"OFFSET {query_parts['offset']}"
                if query_parts.get("offset", 0) > 0 else ""
            )

            sql = (
                f"SELECT * FROM {self._qualified_table_name} {where_clause} "
                f"{order_clause} {limit_clause} {offset_clause}"
            )

            logger.debug(f"Executing list query: SQL='{sql}', Params={where_params}")
            async with self._get_session() as (conn, cursor):
                await cursor.execute(sql, where_params)

                # Fetch and yield rows one at a time
                async for record_data in cursor:
                    try:
                        yield self._deserialize_record(record_data)
                    except Exception as deserialization_error:
                        logger.error(
                            f"Failed to deserialize record during list: "
                            f"{deserialization_error}. Record: {record_data}",
                            exc_info=True,
                        )
                        continue

        except Exception as e:
            logger.error(f"Error listing entities: {e}", exc_info=True)
            self._handle_db_error(e, "listing entities")
            raise  # pragma: no cover

    async def count(
            self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        """Count entities matching the criteria."""
        effective_options = options or QueryOptions()
        logger.debug(f"Counting {self.entity_type.__name__}(s): {options!r}")

        try:
            query_parts = self._translate_query_options(effective_options, False)
            where_clause = query_parts.get("where", "")
            where_params = query_parts.get("params", [])

            sql = f"SELECT COUNT(*) AS count FROM {self._qualified_table_name} {where_clause}"

            logger.debug(f"Executing count query: SQL='{sql}', Params={where_params}")
            async with self._get_session() as (conn, cursor):
                await cursor.execute(sql, where_params)
                result = await cursor.fetchone()
                count_val = result['count']

            logger.info(f"Counted {count_val} matching entities.")
            return int(count_val or 0)

        except Exception as e:
            logger.error(f"Error counting entities: {e}", exc_info=True)
            self._handle_db_error(e, "counting entities")
            raise  # pragma: no cover

    # --- Helper Method Implementations ---
    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dict suitable for MySQL storage."""
        data = prepare_for_storage(entity)
        if not isinstance(data, dict):
            raise TypeError(
                f"prepare_for_storage did not return dict for {entity}"
            )

        serialized_data = {
            k: v for k, v in data.items() if not k.startswith("_")
        }

        # Process data for MySQL compatibility
        for key, value in serialized_data.items():
            # Handle JSON data
            if isinstance(value, (dict, list, tuple, set)):
                serialized_data[key] = json.dumps(value)

            # Handle boolean values explicitly to ensure consistency
            # MySQL uses TINYINT(1) with 0/1 values for booleans
            elif isinstance(value, bool):
                serialized_data[key] = int(value)  # Convert True/False to 1/0

            # Note: We intentionally don't strip timezone information from datetime objects
            # This is because MySQL DATETIME doesn't store timezone information
            # MySQL silently drops timezone info when storing, and we'll add it back on retrieval

        app_id_value = data.get(self.app_id_field)
        if app_id_value is not None:
            serialized_data[self.db_id_field] = app_id_value
        else:
            # Try getattr as fallback before raising error
            db_id_value_fallback = getattr(entity, self.app_id_field, None)
            if db_id_value_fallback is not None:
                serialized_data[self.db_id_field] = db_id_value_fallback
            else:
                raise ValueError(
                    f"Cannot determine PK value for '{self.db_id_field}'."
                )

        # Remove app_id_field if it's different and not a real column
        if self.app_id_field != self.db_id_field:
            try:
                hints = get_type_hints(self.entity_type)
                if self.app_id_field not in hints:
                    serialized_data.pop(self.app_id_field, None)
            except Exception:
                pass  # Ignore errors getting hints

        return serialized_data

    def _deserialize_record(self, record_data: Dict[str, Any]) -> T:
        """Convert MySQL record dict into an entity object T."""
        if record_data is None:
            raise ValueError("Cannot deserialize None record data.")

        entity_dict = dict(record_data)
        processed_dict = {}

        # Get field type information from entity hints
        field_types = {}
        try:
            field_types = get_type_hints(self.entity_type)
        except Exception:
            # Continue without type hints if we can't get them
            pass

        # Process values from MySQL format to Python objects
        for key, value in entity_dict.items():
            # Skip processing None values
            if value is None:
                processed_dict[key] = None
                continue

            # Handle boolean values - MySQL typically returns integers (0/1) for booleans
            if key in field_types and field_types.get(key) is bool:
                if isinstance(value, int):
                    processed_dict[key] = bool(value)
                    continue
                elif isinstance(value, bool):
                    processed_dict[key] = value
                    continue

            # Handle datetime fields - attach timezone info if missing
            if isinstance(value, datetime) and key in field_types:
                field_type = field_types.get(key)
                if field_type is datetime or (
                        get_origin(field_type) is Union and
                        datetime in get_args(field_type)
                ):
                    # If the datetime has no timezone and the original has timezone
                    if value.tzinfo is None:
                        # Attach UTC timezone to make it comparable with timezone-aware datetimes
                        value = value.replace(tzinfo=timezone.utc)

            # Handle JSON strings and arrays - fix string value handling
            if isinstance(value, str) and value.startswith(
                    ("{", "[")) and value.endswith(("]", "}")):
                try:
                    # Parse JSON string to Python object
                    json_value = json.loads(value)

                    # Fix string values in arrays that might have double quotes
                    if key in field_types:
                        field_type = field_types.get(key)
                        # Check if this is a list/array of strings
                        if (get_origin(field_type) in (list, List) and
                                str in get_args(field_type)):
                            # Process each item in the array to handle potential double quotes
                            if isinstance(json_value, list):
                                for i, item in enumerate(json_value):
                                    if isinstance(item, str) and item.startswith(
                                            '"') and item.endswith('"'):
                                        # Remove extra quotes if the string itself is quoted
                                        try:
                                            unquoted = json.loads(item)
                                            if isinstance(unquoted, str):
                                                json_value[i] = unquoted
                                        except json.JSONDecodeError:
                                            # Keep as is if it's not valid JSON
                                            pass

                    processed_dict[key] = json_value
                    continue
                except (json.JSONDecodeError, ValueError):
                    pass  # Not valid JSON, use as is

            processed_dict[key] = value

        db_id_value = processed_dict.get(self.db_id_field)

        if db_id_value is not None:
            # If app_id field exists in the record (meaning it's a distinct column)
            # use its value for the entity's app_id attribute. Otherwise, map
            # the db_id value back to the entity's app_id attribute.
            if self.app_id_field in processed_dict:
                processed_dict[self.app_id_field] = str(
                    processed_dict[self.app_id_field])
            else:
                processed_dict[self.app_id_field] = str(db_id_value)

            # Remove db_id_field if it's different and not an actual entity field
            if self.db_id_field != self.app_id_field:
                try:
                    hints = get_type_hints(self.entity_type)
                    if self.db_id_field not in hints:
                        processed_dict.pop(self.db_id_field, None)
                except Exception:
                    pass  # Ignore hint errors
        else:
            self._logger.error(f"PK '{self.db_id_field}' is NULL in record.")
            # Ensure app_id field is set to None if it exists in the model
            try:
                hints = get_type_hints(self.entity_type)
                if self.app_id_field in hints:
                    processed_dict[self.app_id_field] = None
            except Exception:
                pass

        try:
            return self.entity_type(**processed_dict)
        except Exception as e:
            self._logger.error(
                f"Failed to instantiate {self.entity_type.__name__} from DB "
                f"data: {e}. Data: {processed_dict!r}",
                exc_info=True,
            )
            raise ValueError(
                f"Failed to create {self.entity_type.__name__} from record"
            ) from e

    def _translate_query_options(
            self, options: QueryOptions, include_sorting_pagination: bool = False
    ) -> Dict[str, Any]:
        """Translate QueryOptions into MySQL WHERE/ORDER BY parts."""
        sql_parts: Dict[str, Any] = {"where": "WHERE 1=1", "params": []}

        if options.expression:
            self._logger.debug(
                f"Translating expression: {options.expression!r}")
            try:
                (
                    where_clause,
                    params,
                ) = self._translate_expression_recursive(options.expression)

                # Check if the recursive call actually produced a clause
                if where_clause and where_clause != "1=1" and where_clause != "TRUE":
                    sql_parts["where"] = f"WHERE {where_clause}"
                    sql_parts["params"] = params
                    self._logger.debug(
                        f"Translated WHERE: '{where_clause}', PARAMS: {params}")
                elif where_clause == "FALSE":  # Handle definite false conditions
                    sql_parts["where"] = "WHERE 1=0"  # More standard way than FALSE
                    sql_parts["params"] = []
                    self._logger.debug(
                        "Translated WHERE: '1=0' (Expression evaluated to FALSE)")
                else:
                    self._logger.debug(
                        "Expression translated to trivial/empty WHERE clause.")

            except Exception as e:
                # Log the error source during translation attempt
                self._logger.error(
                    f"Failed to translate query expression: {options.expression!r}",
                    exc_info=True,
                )
                # Re-raise to stop further processing
                raise

        # --- Sorting and Pagination ---
        if include_sorting_pagination:
            if options.random_order:
                sql_parts["order"] = "RAND()"
                self._logger.debug("Added ORDER BY RAND()")
            elif options.sort_by:
                direction = "DESC" if options.sort_desc else "ASC"
                if "." in options.sort_by:
                    base_col, *nested_parts = options.sort_by.split(".")
                    # Use JSON_EXTRACT for sorting nested JSON fields in MySQL
                    path_str = "$." + ".".join(nested_parts)
                    sql_parts["order"] = (
                        f"JSON_EXTRACT(`{base_col}`, '{path_str}') {direction}"
                    )
                else:
                    # Direct column sort
                    sql_parts["order"] = f"`{options.sort_by}` {direction}"
                self._logger.debug(f"Added ORDER BY: {sql_parts['order']}")
            else:
                sql_parts["order"] = None

            # Store limit/offset values for the SQL generation
            sql_parts["limit"] = (
                options.limit
                if (options.limit is not None and options.limit >= 0)
                else -1
            )
            sql_parts["offset"] = (
                options.offset
                if (options.offset is not None and options.offset > 0)
                else 0
            )
            if sql_parts["limit"] != -1:
                self._logger.debug(f"Query Limit: {sql_parts['limit']}")
            if sql_parts["offset"] != 0:
                self._logger.debug(f"Query Offset: {sql_parts['offset']}")

        self._logger.debug(f"Finished translating query options: {sql_parts}")
        return sql_parts

    def _translate_update(self, update: Update) -> Dict[str, Any]:
        """Translate Update object into MySQL SET clause and params."""
        set_clauses: List[str] = []
        params: List[Any] = []

        operations = update.build()
        if not operations:
            return {"set": "", "params": []}

        for op in operations:
            field = op.field_path
            is_nested = "." in field
            base_column_name = field.split(".", 1)[0] if is_nested else field
            quoted_base_column = f"`{base_column_name}`"

            # Prepare JSON path for nested fields
            json_path = None
            if is_nested:
                path_parts = field.split(".")[1:]
                json_path = "$." + ".".join(path_parts)

            # --- Handle SetOperation ---
            if isinstance(op, SetOperation):
                if is_nested and json_path:
                    value = op.value
                    param_value = json.dumps(value)
                    params.append(param_value)
                    set_clauses.append(
                        f"{quoted_base_column} = JSON_SET("
                        f"COALESCE({quoted_base_column}, JSON_OBJECT()), "
                        f"'{json_path}', CAST(%s AS JSON))"
                    )
                else:
                    value = op.value
                    if isinstance(value, (dict, list, tuple, set)):
                        params.append(json.dumps(value))
                    elif isinstance(value, bool):
                        params.append(int(value))
                    else:
                        params.append(value)
                    set_clauses.append(f"{quoted_base_column} = %s")

            # --- Handle UnsetOperation ---
            elif isinstance(op, UnsetOperation):
                if is_nested and json_path:
                    set_clauses.append(
                        f"{quoted_base_column} = JSON_REMOVE("
                        f"COALESCE({quoted_base_column}, JSON_OBJECT()), "
                        f"'{json_path}')"
                    )
                else:
                    set_clauses.append(f"{quoted_base_column} = NULL")

            # --- Handle Increment/Multiply ---
            elif isinstance(op, (IncrementOperation, MultiplyOperation)):
                if is_nested:
                    raise NotImplementedError(
                        "Nested numeric ops (inc/mul) not directly supported in MySQL JSON updates."
                    )
                op_sql = "+" if isinstance(op, IncrementOperation) else "*"
                val = op.amount if isinstance(op, IncrementOperation) else op.factor
                params.append(val)
                set_clauses.append(
                    f"{quoted_base_column} = "
                    f"COALESCE({quoted_base_column}, 0) {op_sql} %s"
                )

            # --- Handle Min/Max ---
            elif isinstance(op, (MinOperation, MaxOperation)):
                if is_nested:
                    raise NotImplementedError(
                        "Nested min/max ops not directly supported in MySQL JSON updates."
                    )
                comparison_op = ">" if isinstance(op, MinOperation) else "<"
                params.append(op.value)
                params.append(op.value)
                set_clauses.append(
                    f"{quoted_base_column} = "
                    f"IF({quoted_base_column} {comparison_op} %s OR {quoted_base_column} IS NULL, "
                    f"%s, {quoted_base_column})"
                )

            # --- Handle PushOperation ---
            elif isinstance(op, PushOperation):
                if not op.items:
                    continue

                if is_nested and json_path:
                    # Nested push (Keep previous working logic - seems okay for now)
                    init_expr = f"COALESCE({quoted_base_column}, JSON_OBJECT())"
                    ensure_array_expr = f"JSON_MERGE_PRESERVE({init_expr}, JSON_OBJECT('{json_path}', JSON_ARRAY()))"
                    current_col_expr = ensure_array_expr
                    for item in op.items:
                        param_value = json.dumps(item)
                        params.append(param_value)
                        current_col_expr = f"JSON_ARRAY_APPEND({current_col_expr}, '{json_path}', CAST(%s AS JSON))"
                    set_clauses.append(f"{quoted_base_column} = {current_col_expr}")
                else:
                    # --- Top-Level Push - Simplest Form ---
                    # Generate one SET assignment per item using the column itself
                    # Use string literal '[]' for COALESCE default
                    for item in op.items:
                        params.append(item)
                        set_clauses.append(
                            f"{quoted_base_column} = JSON_ARRAY_APPEND("
                            f"COALESCE({quoted_base_column}, '[]'), "  # Use string literal
                            f"'$', %s)"  # Append raw value parameter
                        )
                    # --- End Simplest Form ---

            # --- Handle PopOperation ---
            elif isinstance(op, PopOperation):
                if is_nested and json_path:
                    index_str = "last" if op.position == 1 else "0"
                    set_clauses.append(
                        f"{quoted_base_column} = JSON_REMOVE("
                        f"COALESCE({quoted_base_column}, JSON_OBJECT()), "
                        f"'{json_path}[{index_str}]')"
                    )
                else:
                    index_str = "last" if op.position == 1 else "0"
                    set_clauses.append(
                        f"{quoted_base_column} = "
                        f"JSON_REMOVE(COALESCE({quoted_base_column}, JSON_ARRAY()), '$[{index_str}]')"
                    )

            # --- Handle PullOperation ---
            elif isinstance(op, PullOperation):
                value_to_pull = op.value_or_condition
                if isinstance(value_to_pull, dict):
                    raise NotImplementedError(
                        "PullOperation with dictionary criteria not supported in MySQL JSON updates."
                    )
                if is_nested and json_path:
                    raise NotImplementedError(
                        "PullOperation for nested arrays not directly supported in MySQL JSON updates."
                    )
                else:
                    json_val_param = json.dumps(value_to_pull)
                    params.append(json_val_param)
                    set_clauses.append(
                        f"{quoted_base_column} = COALESCE("
                        f"  (SELECT JSON_ARRAYAGG(j.value) FROM "
                        f"   JSON_TABLE(COALESCE({quoted_base_column}, JSON_ARRAY()), '$[*]' COLUMNS("
                        f"     value JSON PATH '$'"
                        f"   )) AS j "
                        f"   WHERE j.value != CAST(%s AS JSON)),"
                        f"  JSON_ARRAY()"
                        f")"
                    )

            else:
                raise TypeError(f"Unsupported UpdateOperation: {type(op)}")

        set_clause_str = ", ".join(set_clauses)
        return {"set": set_clause_str, "params": params}

    def _translate_expression_recursive(
            self, expression: QueryExpression
    ) -> Tuple[str, List[Any]]:
        """
        Recursively translate QueryExpression nodes into MySQL WHERE clause.

        Args:
            expression: The QueryExpression node to translate.

        Returns:
            A tuple containing:
                - The generated SQL WHERE clause fragment (or "TRUE"/"FALSE").
                - A list of parameters for this fragment.
        """
        current_params: List[Any] = []

        if isinstance(expression, QueryFilter):
            field = expression.field_path
            op = expression.operator
            val = expression.value

            self._logger.debug(
                f"Translating Filter: field='{field}', op='{op}' "
                f"(type: {type(op)}), val='{val}' (type: {type(val)})"
            )

            sql_op_map = {
                QueryOperator.EQ: "=",
                QueryOperator.NE: "!=",
                QueryOperator.GT: ">",
                QueryOperator.GTE: ">=",
                QueryOperator.LT: "<",
                QueryOperator.LTE: "<=",
                QueryOperator.IN: "IN",
                QueryOperator.NIN: "NOT IN",
                QueryOperator.LIKE: "LIKE",
                QueryOperator.STARTSWITH: "LIKE",
                QueryOperator.ENDSWITH: "LIKE",
                # MySQL doesn't use @> for contains
            }

            # Handle case where op is already a QueryOperator enum member
            op_key = op if isinstance(op, QueryOperator) else str(op)
            sql_op = sql_op_map.get(op_key)

            is_exists_op = op == QueryOperator.EXISTS or str(op) == 'exists'
            is_contains_op = op == QueryOperator.CONTAINS or str(op) == 'contains'

            # --- Raise error for unsupported operators ---
            if not sql_op and not is_exists_op and not is_contains_op:
                self._logger.error(
                    f"Unsupported operator '{op}' detected, raising ValueError."
                )
                raise ValueError(
                    f"Unsupported query operator for MySQL translation: {op}"
                )

            # --- Path and Field Quoting ---
            is_json_path = "." in field
            quoted_field: str

            if is_json_path:
                base_column_name, *nested_parts = field.split(".")
                quoted_base_column = f"`{base_column_name}`"
                # Build JSON path for MySQL JSON_EXTRACT
                json_path = "$." + ".".join(nested_parts)
                quoted_field = f"JSON_EXTRACT({quoted_base_column}, '{json_path}')"
            else:
                quoted_field = f"`{field}`"  # MySQL uses backticks

            # --- Operator specific logic ---
            if is_exists_op:
                if not is_json_path:  # Top-level field existence
                    exists_op_sql = "IS NOT NULL" if val else "IS NULL"
                    return f"{quoted_field} {exists_op_sql}", []
                else:
                    # Nested JSON path existence using JSON_CONTAINS_PATH
                    exists_sql = (
                        f"JSON_CONTAINS_PATH({quoted_base_column}, 'one', '{json_path}')"
                    )
                    if not val:
                        exists_sql = f"NOT {exists_sql}"
                    return exists_sql, []

            elif is_contains_op:
                # Check if the target field is likely an array based on hints
                is_target_array = False
                if self._validator:
                    try:
                        field_type = self._validator.get_field_type(field)
                        origin = get_origin(field_type)
                        if origin in (list, List, tuple, Tuple, set, Set) or \
                                isinstance(field_type, (list, tuple, set)):
                            is_target_array = True
                    except InvalidPathError:
                        pass  # Field not found in validator

                # MySQL has JSON_CONTAINS for this purpose
                if is_json_path or is_target_array:
                    # For arrays/objects in JSON
                    prepared_val = prepare_for_storage(val)
                    if isinstance(val, str) and is_target_array:
                        # String value in JSON array
                        current_params.append(val)

                        if is_json_path:
                            sql = f"JSON_CONTAINS({quoted_base_column}, %s, '{json_path}')"
                        else:
                            sql = f"JSON_CONTAINS({quoted_field}, JSON_QUOTE(%s))"
                    else:
                        # For objects/values, use JSON_CONTAINS
                        param_json_string = json.dumps(prepared_val)
                        current_params.append(param_json_string)

                        if is_json_path:
                            sql = f"JSON_CONTAINS({quoted_base_column}, %s, '{json_path}')"
                        else:
                            sql = f"JSON_CONTAINS({quoted_field}, %s)"

                    return sql, current_params
                else:
                    # Fallback to LIKE for non-JSON/non-array TEXT fields
                    if not isinstance(val, str):
                        raise ValueError(
                            f"Cannot use non-string value '{val}' with "
                            f"'contains' on TEXT field '{field}'"
                        )
                    current_params.append(f"%{val}%")
                    return f"{quoted_field} LIKE %s", current_params

            # --- Handle remaining standard operators using sql_op ---
            elif sql_op:
                if op_key in (QueryOperator.IN, QueryOperator.NIN):
                    if not isinstance(val, (list, tuple)):
                        raise ValueError(f"Value for {op} must be list/tuple.")
                    if not val:
                        sql_frag = "FALSE" if op_key == QueryOperator.IN else "TRUE"
                        return sql_frag, []
                    placeholders = ", ".join(["%s"] * len(val))
                    current_params.extend(val)
                    sql = f"{quoted_field} {sql_op} ({placeholders})"
                    return sql, current_params

                elif op_key == QueryOperator.STARTSWITH:
                    if not isinstance(val, str):
                        raise ValueError("Value must be string.")
                    current_params.append(f"{val}%")
                    sql = f"{quoted_field} {sql_op} %s"
                    return sql, current_params

                elif op_key == QueryOperator.ENDSWITH:
                    if not isinstance(val, str):
                        raise ValueError("Value must be string.")
                    current_params.append(f"%{val}")
                    sql = f"{quoted_field} {sql_op} %s"
                    return sql, current_params

                elif op_key == QueryOperator.LIKE:
                    if not isinstance(val, str):
                        raise ValueError("Value must be string.")
                    current_params.append(val)
                    sql = f"{quoted_field} {sql_op} %s"
                    return sql, current_params

                else:  # EQ, NE, GT, GTE, LT, LTE
                    current_params.append(val)
                    sql = f"{quoted_field} {sql_op} %s"
                    return sql, current_params
            else:
                # Defensive: Should be unreachable due to initial check
                self._logger.error(
                    f"Internal Translation Error: Operator '{op}' fell through."
                )
                raise RuntimeError(
                    f"Internal error: Operator '{op}' failed."
                )

        elif isinstance(expression, QueryLogical):
            if not expression.conditions:
                return "TRUE", []

            sql_fragments = []
            all_params: List[Any] = []

            for cond in expression.conditions:
                fragment, params = self._translate_expression_recursive(cond)
                if fragment and fragment not in ("TRUE", "FALSE"):
                    sql_fragments.append(f"({fragment})")
                    all_params.extend(params)
                elif fragment == "FALSE" and expression.operator.upper() == "AND":
                    return "FALSE", []
                elif fragment == "TRUE" and expression.operator.upper() == "OR":
                    return "TRUE", []

            if not sql_fragments:
                return "TRUE", []

            logical_op_sql = f" {expression.operator.upper()} "
            return logical_op_sql.join(sql_fragments), all_params

        else:
            raise TypeError(f"Unknown QueryExpression type: {type(expression)}")

    def _handle_db_error(self, error: Exception, context: str = "") -> None:
        """
        Map specific database errors to appropriate exceptions.

        Args:
            error: The exception caught.
            context: A string describing the operation context where the error occurred.

        Raises:
            ValueError: For constraint violations, data format errors,
                        or unsupported operations/values.
            KeyAlreadyExistsException: For primary key unique violations.
            ObjectNotFoundException: If specifically raised and caught.
            NotImplementedError: For features not implemented in this backend.
            RuntimeError: For unexpected database issues or other errors.
        """
        log_message = f"Error during {context}: {error}"

        # --- Handle aiomysql/pymysql specific errors ---
        if hasattr(error, 'args') and len(error.args) > 0:
            # MySQL errors often have the error code in the first argument
            errno = getattr(error, 'errno', None)
            sqlstate = getattr(error, 'sqlstate', None)

            # Log database errors with traceback for debugging
            self._logger.error(log_message, exc_info=True)

            # Duplicate key error (1062)
            if errno == 1062:  # "Duplicate entry"
                # Check if it's the primary key constraint
                if self.db_id_field in str(error):
                    # Extract app_id if possible from context for better message
                    app_id_match = re.search(r"app_id\s+([^\s\)]+)", context)
                    app_id_ctx = f" (app_id: {app_id_match.group(1)})" if app_id_match else ""
                    raise KeyAlreadyExistsException(
                        f"Entity with PK derived from App ID already exists{app_id_ctx}. "
                        f"Detail: {error}"
                    ) from error
                else:
                    # Another unique constraint violation
                    raise ValueError(
                        f"Unique constraint violated during "
                        f"{context}. Detail: {error}"
                    ) from error

            # NOT NULL constraint violation (1048)
            elif errno == 1048:  # "Column cannot be null"
                raise ValueError(
                    f"NOT NULL constraint violated during {context}. Detail: {error}"
                ) from error

            # Foreign key constraint violation (1451, 1452)
            elif errno in (1451, 1452):  # "Cannot delete/update" (parent/child)
                raise ValueError(
                    f"Foreign key constraint violated during {context}. Detail: {error}"
                ) from error

            # Data truncation (1406)
            elif errno == 1406:  # "Data too long"
                raise ValueError(
                    f"Data truncation error during {context}. Detail: {error}"
                ) from error

            # JSON format error
            elif errno == 3140:  # "Invalid JSON text"
                raise ValueError(
                    f"Invalid JSON format during {context}. Detail: {error}"
                ) from error

            # Authorization / Privilege Errors
            elif errno in (1044, 1045):  # "Access denied"
                raise RuntimeError(
                    f"DB authorization/privilege error during {context}. "
                    f"Detail: {error}"
                ) from error

            # Schema errors
            elif errno in (1146, 1054, 1305):  # Table/column/function doesn't exist
                raise RuntimeError(
                    f"DB schema mismatch or missing object during "
                    f"{context}. Detail: {error}"
                ) from error

            # Syntax errors
            elif errno == 1064:  # "SQL syntax error"
                raise RuntimeError(
                    f"Invalid SQL syntax generated during {context}. "
                    f"Detail: {error}"
                ) from error

            # Catch-all for other specific MySQL errors
            else:
                raise RuntimeError(
                    f"A specific database error occurred during {context}: {error}"
                ) from error

        # --- Handle specific non-DB errors from repository/framework logic ---
        elif isinstance(
                error,
                (
                        ValueError,
                        # Includes unsupported ops, bad values, translation errors
                        TypeError,  # Bad types passed to operations
                        NotImplementedError,  # Features not implemented
                        InvalidPathError,  # From ModelValidator
                        KeyAlreadyExistsException,  # Re-raise if passed through
                        ObjectNotFoundException,  # Re-raise if passed through
                )
        ):
            # Log potentially expected errors without traceback
            self._logger.error(log_message)
            raise error  # Re-raise the original specific exception

        # --- Handle truly unexpected errors ---
        else:
            # Log with traceback as it's not a recognized DB or framework error
            self._logger.error(log_message, exc_info=True)
            raise RuntimeError(
                f"An unexpected error occurred during {context}"
            ) from error