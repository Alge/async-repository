# src/async_repository/backends/postgres_repository.py


import json
import logging
import re
from contextlib import asynccontextmanager
from dataclasses import is_dataclass
from datetime import date, datetime
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

# --- asyncpg Driver Import ---
import asyncpg

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
# <<< Only accept Pool now >>>
DB_POOL_TYPE = asyncpg.Pool
DB_RECORD_TYPE = asyncpg.Record

base_logger = logging.getLogger("async_repository.backends.postgres_repository")


# --- Codec Setup ---
# Set to store IDs of connections where codecs have been set within the pool's lifetime (or repo lifetime)
_codec_set_conn_ids: Set[int] = set()
_codec_lock = asyncio.Lock() # Prevent race conditions during first setup across tasks


async def _ensure_postgres_codecs(conn: asyncpg.Connection, logger: logging.LoggerAdapter):
    """Registers JSONB codecs on a connection if not already tracked as set."""
    conn_id = conn.get_server_pid()
    # Fast check without lock
    if conn_id in _codec_set_conn_ids:
        return

    # Acquire lock for slower check and potential setup
    async with _codec_lock:
        # Double check after acquiring lock
        if conn_id in _codec_set_conn_ids:
            return

        logger.debug(
            f"Setting JSONB codec for connection {conn} (ID: {conn_id})"
        )
        try:
            await conn.set_type_codec(
                'jsonb',
                encoder=lambda v: json.dumps(v),
                decoder=lambda s: json.loads(s),
                schema='pg_catalog',
                format='text'
            )
            _codec_set_conn_ids.add(conn_id) # Mark as set
        except Exception as e:
            logger.error(
                f"Failed to set JSONB codec on {conn}: {e}", exc_info=True
            )
            raise RuntimeError("Failed to configure necessary PostgreSQL codecs.") from e


class PostgresRepository(Repository[T], Generic[T]):
    """
    PostgreSQL repository implementation using asyncpg.

    Requires an asyncpg.Pool during initialization and handles connection
    acquisition/release and JSONB codec setup internally.

    Assumptions: ... (rest of docstring remains the same)
    """

    # --- Initialization ---
    def __init__(
        self,
        # <<< Updated Type Hint and Parameter Name >>>
        db_pool: DB_POOL_TYPE,
        table_name: str,
        entity_type: Type[T],
        app_id_field: str = "id",
        db_id_field: Optional[str] = None,
        db_schema: str = "public",
    ):
        """
        Initialize the PostgreSQL repository with an existing connection pool.

        Args:
            db_pool: An active asyncpg.Pool object.
            table_name: The name of the database table.
            entity_type: The Python class representing the entity.
            app_id_field: Attribute name for the application ID.
            db_id_field: Database PK column name (defaults to app_id_field).
            db_schema: The PostgreSQL schema where the table resides.
        """
        # <<< Updated Validation >>>
        if not isinstance(db_pool, asyncpg.Pool):
            raise TypeError("db_pool must be an instance of asyncpg.Pool")
        # Cannot easily check if pool is closed/terminated, assume valid if passed

        self._pool = db_pool # Store the pool

        self._table_name = table_name
        self._db_schema = db_schema
        self._entity_type = entity_type
        self._app_id_field = app_id_field
        self._db_id_field = (
            db_id_field if db_id_field is not None else app_id_field
        )
        self._qualified_table_name = (
            f'"{self._db_schema}"."{self._table_name}"'
        )

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
    def entity_type(self) -> Type[T]: return self._entity_type
    @property
    def app_id_field(self) -> str: return self._app_id_field
    @property
    def db_id_field(self) -> str: return self._db_id_field

    # --- Connection/Session Management ---
    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Acquire connection from the pool, ensure codecs, and release.
        """
        conn: Optional[asyncpg.Connection] = None
        try:
            # Acquire connection from the pool stored during init
            conn = await self._pool.acquire()
            self._logger.debug(f"Acquired connection {conn} from pool.")

            # Ensure codecs are set for this connection
            await _ensure_postgres_codecs(conn, self._logger)

            yield conn # Provide the connection with codecs ensured

        except Exception as e:
            self._logger.error(
                f"Error during connection handling/codec setup: {e}", exc_info=True
            )
            # Let specific db operation handlers wrap with _handle_db_error
            raise
        finally:
            # Release connection back to the pool if acquired
            if conn:
                try:
                    await self._pool.release(conn)
                    self._logger.debug(
                        f"Released connection {conn} back to pool."
                    )
                except Exception as release_error:
                    self._logger.error(
                        f"Error releasing connection {conn}: {release_error}",
                        exc_info=True,
                    )


    # --- Schema and Index Implementation ---
    async def check_schema(self, logger: LoggerAdapter) -> bool:
        """Check if the table exists in the specified database schema."""
        logger.info(f"Checking schema for '{self._qualified_table_name}'...")
        sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = $2
            );
        """
        try:
            async with self._get_session() as conn:
                exists = await conn.fetchval(
                    sql, self._db_schema, self._table_name
                )
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
            # The following line should not be reached if _handle_db_error raises
            return False  # pragma: no cover

    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """Check if essential indexes (PK on db_id_field) seem to exist."""
        logger.info(
            f"Checking essential indexes (PK on '{self.db_id_field}') for "
            f"'{self._qualified_table_name}'..."
        )
        sql = """
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid
                               AND a.attnum = ANY(i.indkey)
            JOIN pg_class t ON t.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE i.indisprimary AND t.relname = $1 AND n.nspname = $2;
        """
        pk_col_found = False
        try:
            async with self._get_session() as conn:
                pk_cols = await conn.fetch(
                    sql, self._table_name, self._db_schema
                )
            for record in pk_cols:
                if record["attname"] == self.db_id_field:
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

        pk_col = self.db_id_field  # e.g., "_id"
        # Determine PK column type (e.g., TEXT or maybe UUID?)
        # For consistency, let's assume TEXT, matching the app_id default
        pk_col_type = "TEXT"
        cols_def = [f'"{pk_col}" {pk_col_type} PRIMARY KEY NOT NULL']
        # Track processed fields to avoid duplication
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
                # Determine type and constraints for the app_id column
                app_id_col_type = "TEXT"  # Default assumption, could check hint
                is_optional = False
                origin = get_origin(app_id_hint)
                if origin is Union and _is_none_type(get_args(app_id_hint)[-1]):
                    is_optional = True
                constraint = "" if is_optional else " NOT NULL"
                # Add UNIQUE constraint for application ID
                constraint += " UNIQUE"

                cols_def.append(
                    f'"{self.app_id_field}" {app_id_col_type}{constraint}'
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

            # --- Type inference logic (same as before) ---
            col_type = "TEXT"
            is_optional = False
            origin = get_origin(hint)
            actual_type = hint
            if origin is Union and hint and _is_none_type(get_args(hint)[-1]):
                is_optional = True
                actual_type = get_args(hint)[0]
                origin = get_origin(actual_type)

            if origin is Union:
                col_type = "JSONB"
            elif actual_type is int:
                col_type = "BIGINT"
            elif actual_type is float:
                col_type = "DOUBLE PRECISION"
            elif actual_type is bool:
                col_type = "BOOLEAN"
            elif actual_type is bytes:
                col_type = "BYTEA"
            elif actual_type is datetime:
                col_type = "TIMESTAMP WITH TIME ZONE"
            elif actual_type is date:
                col_type = "DATE"
            elif actual_type in (str, Any):
                col_type = "TEXT"
            elif origin in (list, List, dict, Dict, set, Set, tuple, Tuple, Mapping) or \
                    (isinstance(actual_type, type) and
                     (issubclass(actual_type, (list, dict, set, tuple)) or
                      is_dataclass(actual_type) or hasattr(actual_type, 'model_dump'))):
                col_type = "JSONB"
            # --- End type inference ---

            constraint = "" if is_optional else " NOT NULL"
            cols_def.append(f'"{name}" {col_type}{constraint}')
            processed_fields.add(name)  # Mark as processed

        create_sql = (
            f"CREATE TABLE IF NOT EXISTS {self._qualified_table_name} "
            f'({", ".join(cols_def)})'
        )
        logger.debug(f"Schema creation SQL: {create_sql}")

        try:
            async with self._get_session() as conn:
                await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{self._db_schema}";')
                logger.info(f"Ensured schema '{self._db_schema}' exists.")
                await conn.execute(create_sql)
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
        """Create potentially useful indexes (UNIQUE, GIN)."""
        logger.info(
            "Attempting to create indexes for "
            f"'{self._qualified_table_name}'..."
        )
        indexes_to_create: List[str] = []
        jsonb_cols_for_gin: List[str] = []

        # Check for distinct app_id_field needing a unique index
        # This check remains the same, but now the column should actually exist
        # because create_schema added it.
        if self.app_id_field != self.db_id_field:
            field_exists_in_hints = False
            try:
                hints = get_type_hints(self.entity_type)
                field_exists_in_hints = self.app_id_field in hints
            except Exception:
                pass  # Ignore hint errors, proceed if field names differ

            # We attempt index creation if names differ, assuming create_schema
            # added the column (or it exists manually). The CREATE INDEX command
            # will fail if the column truly doesn't exist.
            if field_exists_in_hints:
                idx_name = f"idx_{self._table_name}_{self.app_id_field}_unique"
                # Important: The column name here MUST be self.app_id_field ("id")
                sql = (
                    f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx_name}" ON '
                    f'{self._qualified_table_name}("{self.app_id_field}");'
                )
                indexes_to_create.append(sql)
                logger.info(
                    f"Defining unique index on separate app_id_field "
                    f"'{self.app_id_field}'."
                )
            else:
                logger.info(
                    f"Skipping unique index on app_id_field '{self.app_id_field}' as it's not found in model hints (column might not exist).")

        # --- (Rest of the GIN index identification logic remains the same) ---
        try:
            hints = get_type_hints(self.entity_type)
        except Exception as e:
            logger.warning(f"Could not get hints for GIN indexing: {e}")
            hints = {}
        for name, hint in hints.items():
            if name == self.db_id_field or (
                    name == self.app_id_field): continue  # Skip PK and app_id
            if name.startswith("_"): continue
            actual_type = hint;
            origin = get_origin(hint)
            if origin is Union and hint and _is_none_type(
                get_args(hint)[-1]): actual_type = get_args(hint)[
                0]; origin = get_origin(actual_type)
            if origin is Union or origin in (list, List, dict, Dict, set, Set, tuple,
                                             Tuple, Mapping) or \
                    (isinstance(actual_type, type) and (issubclass(actual_type,
                                                                   (list, dict, set,
                                                                    tuple)) or is_dataclass(
                        actual_type) or hasattr(actual_type, 'model_dump'))):
                jsonb_cols_for_gin.append(name)
        for col_name in jsonb_cols_for_gin:
            idx_name = f"idx_{self._table_name}_{col_name}_gin"
            sql = (
                f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON {self._qualified_table_name} USING GIN ("{col_name}");')
            indexes_to_create.append(sql)
            logger.info(f"Defining GIN index on JSONB column '{col_name}'.")
        # --- End GIN index logic ---

        if not indexes_to_create:
            logger.info(
                f"No explicit indexes defined for creation on "
                f"'{self._qualified_table_name}'."
            )
            return

        try:
            async with self._get_session() as conn:
                async with conn.transaction():
                    for sql in indexes_to_create:
                        logger.debug(f"Executing index SQL: {sql}")
                        await conn.execute(sql)
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
            # Let _handle_db_error raise the appropriate exception
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
            f'SELECT * FROM {self._qualified_table_name} '
            f'WHERE "{field_to_match}" = $1 LIMIT 1'
        )
        params = (id,)

        try:
            async with self._get_session() as conn:
                record_data = await conn.fetchrow(
                    query, *params, timeout=timeout
                )

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
            cols_clause = ", ".join(f'"{k}"' for k in cols)
            placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
            query = (
                f"INSERT INTO {self._qualified_table_name} ({cols_clause}) "
                f"VALUES ({placeholders})"
            )
            params = tuple(db_data[k] for k in cols)

            async with self._get_session() as conn:
                await conn.execute(query, *params, timeout=timeout)

            logger.info(
                f"Stored new {self.entity_type.__name__} ID '{app_id}'. "
                "(Commit handled externally)"
            )
            return entity if return_value else None

        except asyncpg.UniqueViolationError as e:
            # Basic check focusing on the PK column name
            if self.db_id_field in str(e.detail or "") or (
                e.constraint_name and f"{self._table_name}_pkey" in e.constraint_name
            ):
                logger.warning(
                    f"Failed to store {self.entity_type.__name__} PK "
                    f"'{db_id_value}': {e}"
                )
                raise KeyAlreadyExistsException(
                    f"Entity with ID '{app_id}' already exists."
                ) from e
            else:
                # Other unique constraint violation
                logger.error(
                    f"Unique constraint violation storing entity "
                    f"(app_id {app_id}): {e}",
                    exc_info=True,
                )
                self._handle_db_error(e, f"storing entity app_id {app_id}")
                raise  # Should not be reached

        except Exception as e:
            logger.error(
                f"Error storing entity (app_id {app_id}): {e}", exc_info=True
            )
            self._handle_db_error(e, f"storing entity app_id {app_id}")
            raise  # Should not be reached

    async def upsert(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
    ) -> None:
        """Insert or update an entity based on its ID (using ON CONFLICT)."""
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
            cols_clause = ", ".join(f'"{k}"' for k in cols)
            placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
            params = tuple(db_data[k] for k in cols)

            update_cols = [k for k in cols if k != self.db_id_field]
            if not update_cols:
                update_clause_str = "DO NOTHING"
            else:
                set_clause = ", ".join(
                    f'"{k}" = excluded."{k}"' for k in update_cols
                )
                update_clause_str = f"DO UPDATE SET {set_clause}"

            query = (
                f"INSERT INTO {self._qualified_table_name} ({cols_clause}) "
                f"VALUES ({placeholders}) "
                f'ON CONFLICT ("{self.db_id_field}") {update_clause_str}'
            )

            async with self._get_session() as conn:
                await conn.execute(query, *params, timeout=timeout)

            logger.info(
                f"Upserted {self.entity_type.__name__} ID '{app_id}'. "
                "(Commit handled externally)"
            )
        except Exception as e:
            logger.error(
                f"Error upserting entity (app_id {app_id}): {e}", exc_info=True
            )
            self._handle_db_error(e, f"upserting entity app_id {app_id}")
            raise  # Should not be reached

        # src/async_repository/backends/postgres_repository.py (within PostgresRepository class)

    async def update_one(
            self,
            options: QueryOptions,
            update: Update,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            return_value: bool = False,
    ) -> Optional[T]:
        """
        Update single entity matching criteria using CTE for safety.

        Handles potential sorting/offset before update and returns the
        updated entity if requested.
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

        try:
            # Prepare options for finding the single target row ID
            find_options = options.copy()
            find_options.limit = 1
            find_options.offset = find_options.offset or 0

            # Translate query and update parts separately
            find_parts = self._translate_query_options(
                find_options, include_sorting_pagination=True
            )
            update_parts = self._translate_update(update)

            if not update_parts.get("set"):
                logger.warning("Update resulted in no SET clauses.")
                if return_value:
                    # Find and return the existing entity if requested
                    return await self.find_one(logger=logger, options=options)
                return None

            # --- Parameter Re-indexing and Ordering ---
            param_idx = 1  # Start parameter indexing at $1

            # 1. Process WHERE clause first
            where_clause, where_params, param_idx = self._build_where_clause(
                find_parts, param_idx
            )

            # 2. Process LIMIT/OFFSET (if any) and add their params
            limit_offset_params = []
            limit_clause = ""
            offset_clause = ""
            if find_parts.get("limit", -1) >= 0:
                limit_clause = f"LIMIT ${param_idx}"
                limit_offset_params.append(find_parts["limit"])
                param_idx += 1
            if find_parts.get("offset", 0) > 0:
                offset_clause = f"OFFSET ${param_idx}"
                limit_offset_params.append(find_parts["offset"])
                param_idx += 1

            order_clause = (
                f"ORDER BY {find_parts['order']}" if find_parts.get("order") else ""
            )

            # 3. Process SET clause, re-indexing its params starting from current param_idx
            # _reindex_params returns the modified clause, original params list, and next index
            set_clause, set_params_ordered, next_param_idx = self._reindex_params(
                update_parts["set"], update_parts["params"], param_idx
            )
            # param_idx is now updated for subsequent clauses if needed

            # 4. Assemble the final parameter list IN ORDER
            all_params = (
                    where_params + limit_offset_params + set_params_ordered
            )
            # --- End Parameter Re-indexing ---

            returning_clause = "RETURNING *" if return_value else ""

            # Use CTE to select the PK, respecting order/limit/offset
            # Ensures only the intended single row is updated.
            sql = f"""
                WITH target AS (
                    SELECT "{self.db_id_field}"
                    FROM {self._qualified_table_name}
                    {where_clause}
                    {order_clause}
                    {limit_clause}
                    {offset_clause}
                )
                UPDATE {self._qualified_table_name} AS t
                SET {set_clause}
                WHERE t."{self.db_id_field}" =
                      (SELECT "{self.db_id_field}" FROM target)
                {returning_clause};
            """

            logger.debug(f"Executing update_one: SQL='{sql}', Params={all_params}")

            final_sql_debug = sql.replace('\n', ' ').replace('    ',
                                                             '')  # Clean up for logging
            logger.debug(
                f"!!! DEBUG update_one SQL: {final_sql_debug}")  # Use CRITICAL to ensure visibility
            logger.debug(f"!!! DEBUG update_one Params: {all_params}")

            async with self._get_session() as conn:
                if return_value:
                    # Use fetchrow when RETURNING * is expected
                    updated_record = await conn.fetchrow(
                        sql, *all_params, timeout=timeout
                    )
                    if updated_record is None:
                        # This implies the row selected by the CTE subquery
                        # didn't exist or couldn't be updated (e.g., deleted
                        # concurrently, though less likely with CTE).
                        logger.warning(
                            f"Update target matching criteria was not found "
                            f"during update execution (or disappeared): "
                            f"{options.expression!r}"
                        )
                        raise ObjectNotFoundException(
                            f"No {self.entity_type.__name__} found matching "
                            f"criteria for update (or disappeared): "
                            f"{options.expression!r}"
                        )
                    entity = self._deserialize_record(updated_record)
                    pk_val = getattr(entity, self.db_id_field, "?")
                    logger.info(
                        f"Updated and retrieved {self.entity_type.__name__} "
                        f"PK '{pk_val}'."
                    )
                    return entity
                else:
                    status = await conn.execute(
                        sql, *all_params, timeout=timeout
                    )
                    # Parse status string like 'UPDATE 1'
                    match = re.match(r"UPDATE\s+(\d+)", status or "")
                    updated_count = -1  # Default to unknown
                    if match:
                        try:
                            updated_count = int(match.group(1))
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Could not parse update count from status: {status}")
                    else:
                        logger.warning(
                            f"Update status string format unexpected: {status}")

                    if updated_count == 1:
                        logger.info(
                            "Updated one matching entity. "
                            "(Commit handled externally)"
                        )
                        return None
                    elif updated_count == 0:
                        # If the CTE found a row but UPDATE affected 0, it's problematic.
                        logger.warning(
                            f"Update target matching criteria was not found "
                            f"during update execution (possible race condition "
                            f"or CTE issue): {options.expression!r}"
                        )
                        # Raise ObjectNotFoundException because the intended target wasn't updated
                        raise ObjectNotFoundException(
                            f"No {self.entity_type.__name__} found matching "
                            f"criteria for update (or disappeared/failed): "
                            f"{options.expression!r}"
                        )
                    else:  # updated_count is -1 or some other number
                        # This case is ambiguous. Log a warning but assume okay if no error raised.
                        logger.warning(
                            f"Update operation status parsing resulted in count {updated_count} "
                            f"(Status: {status}). Assuming success as no error was raised."
                        )
                        return None

        except ObjectNotFoundException:
            # Catch specific exception raised within this method or find_one
            logger.warning(
                f"ObjectNotFoundException during update_one for filter: "
                f"{options.expression!r}"
            )
            raise  # Propagate ObjectNotFoundException
        except Exception as e:
            # Catch other DB errors or translation errors
            logger.error(
                f"Error updating one entity (filter: {options.expression!r}): {e}",
                exc_info=True
            )
            self._handle_db_error(e, "updating one entity")
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
        # <<< Corrected check for expression >>>
        # Original check was correct, but the test expects a specific message
        if not options.expression:
            # Raise the exact error message the test expects
            raise ValueError(
                "QueryOptions must include an 'expression' for update_many."
            )

        if options.limit is not None or options.offset is not None or options.sort_by:
            logger.warning("limit/offset/sort_by ignored for update_many.")

        try:
            param_idx = 1
            find_parts = self._translate_query_options(options, False)
            update_parts = self._translate_update(update)

            where_clause, where_params, param_idx = self._build_where_clause(
                find_parts, param_idx
            )

            # <<< Apply the same fix as in update_one >>>
            # Unpack all 3 return values from _reindex_params
            set_clause, set_params_ordered, next_param_idx = self._reindex_params(
                update_parts["set"], update_parts["params"], param_idx
            )
            # We don't use next_param_idx here, but need to unpack it

            # <<< Adjusted safety check >>>
            # Check where_clause specifically, not the whole find_parts dict
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
            # Use the correctly ordered parameters
            all_params = where_params + set_params_ordered

            logger.debug(f"Executing update_many: SQL='{sql}', Params={all_params}")
            async with self._get_session() as conn:
                 status = await conn.execute(sql, *all_params, timeout=timeout)
                 match = re.match(r"UPDATE (\d+)", status or "") # Add `or ""`
                 affected_count = int(match.group(1)) if match else 0

            logger.info(
                f"Updated {affected_count} {self.entity_type.__name__}(s). "
                "(Commit handled externally)"
            )
            return affected_count
        except ValueError as e:
            # Handle the specific ValueError raised for missing expression
            logger.error(f"ValueError during update_many: {e}")
            self._handle_db_error(e, "updating many entities (validation)")
            raise # Re-raise ValueError after logging/handling
        except Exception as e:
            logger.error(f"Error updating many entities: {e}", exc_info=True)
            self._handle_db_error(e, "updating many entities")
            raise # pragma: no cover


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
        # <<< Change the error message here >>>
        if not options.expression:
            raise ValueError(
                "QueryOptions must include an 'expression' for delete_many."
            )

        # Note: PostgreSQL DELETE doesn't directly support LIMIT/OFFSET/ORDER BY
        if options.limit is not None or options.offset is not None or options.sort_by:
            logger.warning("limit/offset/sort_by ignored for Postgres delete_many.")

        try:
            param_idx = 1
            find_parts = self._translate_query_options(options, False)
            where_clause, where_params, param_idx = self._build_where_clause(
                find_parts, param_idx
            )

            # <<< Adjusted safety check >>>
            if not where_clause or where_clause == "WHERE 1=1":
                raise ValueError(
                    "Cannot delete_many without a specific filter expression "
                    "(safety check)."
                 )

            sql = f"DELETE FROM {self._qualified_table_name} {where_clause}"
            all_params = where_params

            logger.debug(f"Executing delete_many: SQL='{sql}', Params={all_params}")
            async with self._get_session() as conn:
                 status = await conn.execute(sql, *all_params, timeout=timeout)
                 match = re.match(r"DELETE (\d+)", status or "") # Add or ""
                 affected_count = int(match.group(1)) if match else 0

            logger.info(
                f"Deleted {affected_count} {self.entity_type.__name__}(s). "
                "(Commit handled externally)"
            )
            return affected_count
        except ValueError as e:
            # Handle the specific ValueError raised for missing expression
            # or the safety check
            logger.error(f"ValueError during delete_many: {e}")
            self._handle_db_error(e, "deleting many entities (validation)")
            raise # Re-raise ValueError after logging/handling
        except Exception as e:
            logger.error(f"Error deleting many entities: {e}", exc_info=True)
            self._handle_db_error(e, "deleting many entities")
            raise # pragma: no cover

    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        """List entities matching criteria, with sorting and pagination."""
        effective_options = options or QueryOptions()
        logger.debug(f"Listing {self.entity_type.__name__}(s): {options!r}")

        try:
            param_idx = 1
            query_parts = self._translate_query_options(effective_options, True)
            where_clause, where_params, param_idx = self._build_where_clause(
                query_parts, param_idx
            )
            order_clause = (
                f"ORDER BY {query_parts['order']}" if query_parts.get("order") else ""
            )
            limit_clause = (
                f"LIMIT ${param_idx}" if query_parts.get("limit", -1) >= 0 else ""
            )
            offset_clause = (
                f"OFFSET ${param_idx + (1 if limit_clause else 0)}"
                if query_parts.get("offset", 0) > 0
                else ""
            )

            all_params = list(where_params)
            if limit_clause:
                all_params.append(query_parts["limit"])
            if offset_clause:
                all_params.append(query_parts["offset"])

            sql = (
                f"SELECT * FROM {self._qualified_table_name} {where_clause} "
                f"{order_clause} {limit_clause} {offset_clause}"
            )

            logger.debug(f"Executing list query: SQL='{sql}', Params={all_params}")
            async with self._get_session() as conn:
                # Cursor needs transaction in asyncpg > 0.24.0 ? Check docs.
                # Using fetch for simplicity unless cursor proven necessary.
                # async with conn.transaction(): # If using cursor
                #     async for record_data in conn.cursor(sql, *all_params, timeout=effective_options.timeout):
                #         # ... yield ...
                records = await conn.fetch(
                    sql, *all_params, timeout=effective_options.timeout
                )
                for record_data in records:
                    try:
                        yield self._deserialize_record(record_data)
                    except Exception as deserialization_error:
                        logger.error(
                            f"Failed to deserialize record during list: "
                            f"{deserialization_error}. Record: {dict(record_data)}",
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
            param_idx = 1
            query_parts = self._translate_query_options(effective_options, False)
            where_clause, where_params, param_idx = self._build_where_clause(
                query_parts, param_idx
            )

            sql = f"SELECT COUNT(*) FROM {self._qualified_table_name} {where_clause}"
            all_params = where_params

            logger.debug(f"Executing count query: SQL='{sql}', Params={all_params}")
            async with self._get_session() as conn:
                count_val = await conn.fetchval(
                    sql, *all_params, timeout=effective_options.timeout
                )

            logger.info(f"Counted {count_val} matching entities.")
            return int(count_val or 0)
        except Exception as e:
            logger.error(f"Error counting entities: {e}", exc_info=True)
            self._handle_db_error(e, "counting entities")
            raise  # pragma: no cover

    # --- Helper Method Implementations ---
    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Convert entity to dict suitable for asyncpg storage."""
        data = prepare_for_storage(entity)
        if not isinstance(data, dict):
            raise TypeError(
                f"prepare_for_storage did not return dict for {entity}"
            )

        serialized_data = {
            k: v for k, v in data.items() if not k.startswith("_")
        }

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

    def _deserialize_record(self, record_data: asyncpg.Record) -> T:
        """Convert asyncpg.Record (dict-like) into an entity object T."""
        if record_data is None:
            raise ValueError("Cannot deserialize None record data.")

        entity_dict = dict(record_data)
        processed_dict = entity_dict

        db_id_value = entity_dict.get(self.db_id_field) # e.g., value from "_id"

        if db_id_value is not None:
            # If app_id field exists in the record (meaning it's a distinct column)
            # use its value for the entity's app_id attribute. Otherwise, map
            # the db_id value back to the entity's app_id attribute.
            if self.app_id_field in entity_dict:
                 processed_dict[self.app_id_field] = str(entity_dict[self.app_id_field])
            else:
                 processed_dict[self.app_id_field] = str(db_id_value)

            # Remove db_id_field if it's different and not an actual entity field
            if self.db_id_field != self.app_id_field:
                 try:
                      hints = get_type_hints(self.entity_type)
                      if self.db_id_field not in hints:
                           processed_dict.pop(self.db_id_field, None)
                 except Exception: pass # Ignore hint errors
        else:
            self._logger.error(f"PK '{self.db_id_field}' is NULL in record.")
            # Ensure app_id field is set to None if it exists in the model
            try:
                hints = get_type_hints(self.entity_type)
                if self.app_id_field in hints:
                    processed_dict[self.app_id_field] = None
            except Exception: pass

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

    def _build_where_clause(
        self, query_parts: Dict[str, Any], start_index: int
    ) -> Tuple[str, List[Any], int]:
        """Build WHERE clause string and re-index $n parameters."""
        where_clause = ""
        where_params = []
        current_index = start_index
        original_where = query_parts.get("where")

        if original_where and original_where != "1=1":
            original_params = query_parts.get("params", [])
            param_count = original_where.count("$")
            # Basic sanity check - $n placeholders should match param list length
            # Note: Literal '$' could exist, but less likely in generated code
            if param_count != len(original_params):
                raise ValueError(
                    f"WHERE clause parameter count mismatch: "
                    f"{param_count} placeholders, {len(original_params)} params."
                )

            reindexed_where = original_where
            # Replace $1, $2, ... sequentially with $start_index, ...
            for i in range(param_count):
                placeholder = r"\$" + str(i + 1) + r"(\b|$)"
                replacement = f"${current_index + i}\\1"
                reindexed_where = re.sub(
                    placeholder, replacement, reindexed_where, count=1
                )

            where_clause = f"WHERE {reindexed_where}"
            where_params = original_params
            current_index += param_count

        return where_clause, where_params, current_index

        # src/async_repository/backends/postgres_repository.py


    def _reindex_params(
        self, clause_str: str, params: List[Any], start_index: int
    ) -> Tuple[str, List[Any], int]:
        """
        Re-index $n parameters in a SQL clause string sequentially, assuming
        input placeholders are $1, $2, ... corresponding to the params list.
        """
        if not clause_str or not params:
            return clause_str, params, start_index

        reindexed_clause = clause_str
        num_params = len(params)

        # Replace placeholders in reverse order to avoid partial replacements
        # (e.g., replacing $1 in $10 before replacing $10)
        for i in range(num_params - 1, -1, -1):
            original_index = i + 1
            new_index = start_index + i
            # Use word boundary \b to ensure whole number match
            placeholder_pattern = r"\$" + str(original_index) + r"\b"
            replacement = f"${new_index}"
            reindexed_clause = re.sub(
                placeholder_pattern, replacement, reindexed_clause
            )

        next_index = start_index + num_params
        return reindexed_clause, params, next_index

        # src/async_repository/backends/postgres_repository.py (within PostgresRepository class)

    def _translate_query_options(
            self, options: QueryOptions, include_sorting_pagination: bool = False
    ) -> Dict[str, Any]:
        """Translate QueryOptions into PostgreSQL WHERE/ORDER BY parts."""
        sql_parts: Dict[str, Any] = {"where": "1=1", "params": []}
        current_param_index = 1  # Start fresh for each query translation

        if options.expression:
            self._logger.debug(
                f"Translating expression: {options.expression!r}")  # Log start
            try:
                (
                    where_clause,
                    params,
                    current_param_index,  # Capture updated index
                ) = self._translate_expression_recursive(
                    options.expression, current_param_index
                )
                # Check if the recursive call actually produced a clause
                if where_clause and where_clause != "1=1" and where_clause != "TRUE":
                    sql_parts["where"] = where_clause
                    sql_parts["params"] = params
                    self._logger.debug(
                        f"Translated WHERE: '{where_clause}', PARAMS: {params}")
                elif where_clause == "FALSE":  # Handle definite false conditions
                    sql_parts["where"] = "1=0"  # More standard way than FALSE
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
                # Re-raise to stop further processing and allow test to catch
                raise

        # --- Sorting and Pagination ---
        if include_sorting_pagination:
            if options.random_order:
                sql_parts["order"] = "RANDOM()"
                self._logger.debug("Added ORDER BY RANDOM()")
            elif options.sort_by:
                direction = "DESC" if options.sort_desc else "ASC"
                if "." in options.sort_by:
                    base_col, *nested_parts = options.sort_by.split(".")
                    # Use text extraction for sorting nested JSONB fields
                    path_str = "{" + ",".join(nested_parts) + "}"
                    sql_parts["order"] = (
                        f'("{base_col}"#>>\'{path_str}\') {direction}'
                    )
                else:
                    # Direct column sort
                    sql_parts["order"] = f'"{options.sort_by}" {direction}'
                self._logger.debug(f"Added ORDER BY: {sql_parts['order']}")
            else:
                sql_parts["order"] = None

            # Store limit/offset values regardless of whether they are used
            # in the SQL generation (handled by the caller, e.g., list method)
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
            if sql_parts["limit"] != -1: self._logger.debug(
                f"Query Limit: {sql_parts['limit']}")
            if sql_parts["offset"] != 0: self._logger.debug(
                f"Query Offset: {sql_parts['offset']}")

        self._logger.debug(f"Finished translating query options: {sql_parts}")
        return sql_parts

        # src/async_repository/backends/postgres_repository.py

    def _translate_expression_recursive(
            self, expression: QueryExpression, param_start_index: int
    ) -> Tuple[str, List[Any], int]:
        """
        Recursively translate QueryExpression nodes into PostgreSQL WHERE clause.

        Args:
            expression: The QueryExpression node to translate.
            param_start_index: The starting index for $n placeholders.

        Returns:
            A tuple containing:
                - The generated SQL WHERE clause fragment (or "TRUE"/"FALSE").
                - A list of parameters for this fragment.
                - The next available parameter index ($n+1).
        """
        current_params: List[Any] = []
        current_index = param_start_index

        if isinstance(expression, QueryFilter):
            field = expression.field_path
            op = expression.operator  # op can be QueryOperator member or str
            val = expression.value

            self._logger.debug(
                f"Translating Filter: field='{field}', op='{op}' "
                f"(type: {type(op)}), val='{val}' (type: {type(val)})"
            )

            sql_op_map = {
                QueryOperator.EQ: "=", QueryOperator.NE: "!=",
                QueryOperator.GT: ">", QueryOperator.GTE: ">=",
                QueryOperator.LT: "<", QueryOperator.LTE: "<=",
                QueryOperator.IN: "IN", QueryOperator.NIN: "NOT IN",
                QueryOperator.LIKE: "LIKE", QueryOperator.STARTSWITH: "LIKE",
                QueryOperator.ENDSWITH: "LIKE", QueryOperator.CONTAINS: "@>",
                # EXISTS ('?') is handled separately below
            }
            # Handle case where op is already a QueryOperator enum member
            op_key = op if isinstance(op, QueryOperator) else str(op)
            sql_op = sql_op_map.get(op_key)  # Use op_key for lookup
            placeholder = f"${current_index}"  # Base placeholder for single value

            self._logger.debug(f"Mapped sql_op for '{op_key}': {sql_op}")
            is_exists_op = op == QueryOperator.EXISTS or str(op) == 'exists'
            is_contains_op = op == QueryOperator.CONTAINS or str(op) == 'contains'
            self._logger.debug(
                f"Checking condition: not sql_op ({not sql_op}) "
                f"and not is_exists_op ({not is_exists_op}) "
                f"and not is_contains_op ({not is_contains_op})"
            )

            # --- Raise error for unsupported operators ---
            if not sql_op and not is_exists_op and not is_contains_op:
                self._logger.error(
                    f"Unsupported operator '{op}' detected, raising ValueError."
                )
                raise ValueError(
                    f"Unsupported query operator for PostgreSQL translation: {op}"
                )

            # --- Path and Field Quoting ---
            is_jsonb_path = "." in field
            quoted_field: str
            path_list: Optional[List[str]] = None
            quoted_base_column: Optional[str] = None  # Only set if is_jsonb_path

            if is_jsonb_path:
                base_column_name, *nested_parts = field.split(".")
                quoted_base_column = f'"{base_column_name}"'
                path_list = nested_parts
                # Use #>> for default text comparison for most operators
                path_str_hashtags = "{" + ",".join(path_list) + "}"
                quoted_field = f"({quoted_base_column}#>>'{path_str_hashtags}')"
            else:
                quoted_base_column = f'"{field}"'  # Base column is the field itself
                quoted_field = quoted_base_column

            # --- Operator specific logic ---
            if is_exists_op:
                if not is_jsonb_path:  # Top-level field existence
                    exists_op_sql = "IS NOT NULL" if val else "IS NULL"
                    return f"{quoted_field} {exists_op_sql}", [], current_index
                else:
                    # Nested JSONB path existence using jsonb_path_exists
                    jsonpath_suffix = ".".join(
                        f'"{p}"' if not p.isdigit() else f'[{p}]'
                        for p in (path_list or [])
                    )
                    path_param = f"$.{jsonpath_suffix}"
                    current_params.append(path_param)
                    idx = current_index
                    current_index += 1
                    # Note: quoted_base_column is guaranteed non-None here
                    sql = f"jsonb_path_exists({quoted_base_column}, ${idx})"
                    if not val:
                        sql = f"NOT {sql}"
                    return sql, current_params, current_index

            # In _translate_expression_recursive method in PostgresRepository class
            # Replace the "is_contains_op" section with this improved version:

            elif is_contains_op:
                # Check if the target field is likely an array based on hints
                is_target_array = False
                if self._validator:
                    try:
                        field_type = self._validator.get_field_type(field)
                        # Check if origin is List/list or field_type itself is list
                        origin = get_origin(field_type)
                        # Allow List, list, tuple, Tuple, set, Set as array-like for contains
                        if origin in (list, List, tuple, Tuple, set, Set) or \
                                isinstance(field_type, (list, tuple, set)):
                            is_target_array = True
                    except InvalidPathError:
                        pass  # Field not found in validator

                # Use JSONB specialized operators
                if is_jsonb_path or is_target_array:
                    # For PostgreSQL JSONB arrays, we need a different approach:
                    # For strings in arrays, use the ? operator which checks string values
                    if isinstance(val, str) and is_target_array:
                        # For string value in JSONB array, use the ? operator
                        # This directly checks if the string is an array element
                        current_params.append(val)  # Just the string value
                        idx = current_index
                        current_index += 1

                        if path_list and quoted_base_column:
                            # Nested JSONB path contains check
                            path_str_curly = "{" + ",".join(path_list) + "}"
                            sql = f"({quoted_base_column}#>'{path_str_curly}') ? ${idx}"
                        else:
                            # Top-level JSONB column contains check for string value in array
                            sql = f"{quoted_field} ? ${idx}"

                        self._logger.debug(
                            f"Array contains string query: {sql} with string param: '{val}'"
                        )
                    else:
                        # Fallback to standard @> for non-string values or non-array fields
                        # This is useful for checking object containment or more complex values
                        prepared_val = prepare_for_storage(val)
                        # JSON array with the value
                        param_json_string = json.dumps([prepared_val])

                        current_params.append(param_json_string)
                        idx = current_index
                        current_index += 1

                        if path_list and quoted_base_column:
                            # Nested JSONB path contains check
                            path_str_curly = "{" + ",".join(path_list) + "}"
                            sql = f"({quoted_base_column}#>'{path_str_curly}') @> ${idx}::jsonb"
                        else:
                            # Top-level JSONB column contains check
                            sql = f"{quoted_field} @> ${idx}::jsonb"

                        self._logger.debug(
                            f"Array/Object containment query: {sql} with JSONB param: '{param_json_string}'"
                        )

                    return sql, current_params, current_index
                else:
                    # --- Fallback to LIKE for non-JSONB/non-array TEXT fields ---
                    self._logger.warning(
                        f"Operator 'contains' used on non-JSONB/non-array "
                        f"field '{field}'. Falling back to LIKE '%value%'."
                    )
                    if not isinstance(val, str):
                        raise ValueError(
                            f"Cannot use non-string value '{val}' with "
                            f"'contains' on TEXT field '{field}'"
                        )
                    # Use the mapped operator which should be LIKE here
                    sql_op_like = sql_op_map.get(QueryOperator.LIKE)
                    if not sql_op_like:  # Defensive check
                        raise RuntimeError(
                            "Internal error: LIKE operator mapping missing.")
                    current_params.append(f"%{val}%")
                    sql = f"{quoted_field} {sql_op_like} {placeholder}"
                    current_index += 1
                    return sql, current_params, current_index

            # --- Handle remaining standard operators using sql_op ---
            elif sql_op:  # Should only be reached if sql_op was found
                if op_key in (QueryOperator.IN, QueryOperator.NIN):
                    if not isinstance(val, (list, tuple)):
                        raise ValueError(f"Value for {op} must be list/tuple.")
                    if not val:
                        sql_frag = "FALSE" if op_key == QueryOperator.IN else "TRUE"
                        return sql_frag, [], param_start_index
                    placeholders = ", ".join(
                        f"${current_index + i}" for i in range(len(val))
                    )
                    current_params.extend(val)
                    sql = f"{quoted_field} {sql_op} ({placeholders})"
                    current_index += len(val)
                    return sql, current_params, current_index

                elif op_key == QueryOperator.STARTSWITH:
                    if not isinstance(val, str): raise ValueError(
                        "Value must be string.")  # noqa E701
                    current_params.append(f"{val}%")
                    sql = f"{quoted_field} {sql_op} {placeholder}"
                    current_index += 1
                    return sql, current_params, current_index

                elif op_key == QueryOperator.ENDSWITH:
                    if not isinstance(val, str): raise ValueError(
                        "Value must be string.")  # noqa E701
                    current_params.append(f"%{val}")
                    sql = f"{quoted_field} {sql_op} {placeholder}"
                    current_index += 1
                    return sql, current_params, current_index

                elif op_key == QueryOperator.LIKE:
                    if not isinstance(val, str): raise ValueError(
                        "Value must be string.")  # noqa E701
                    current_params.append(val)
                    sql = f"{quoted_field} {sql_op} {placeholder}"
                    current_index += 1
                    return sql, current_params, current_index

                else:  # EQ, NE, GT, GTE, LT, LTE
                    current_params.append(val)
                    sql = f"{quoted_field} {sql_op} {placeholder}"
                    current_index += 1
                    return sql, current_params, current_index
            else:
                # Defensive: Should be unreachable due to initial check
                self._logger.error(
                    f"Internal Translation Error: Operator '{op}' fell through."
                )  # pragma: no cover
                raise RuntimeError(
                    f"Internal error: Operator '{op}' failed."
                )  # pragma: no cover

        elif isinstance(expression, QueryLogical):
            if not expression.conditions:
                return ("TRUE", [], param_start_index)

            sql_fragments = []
            all_params: List[Any] = []
            current_idx_recursive = param_start_index

            for cond in expression.conditions:
                fragment, params, current_idx_recursive = (
                    self._translate_expression_recursive(
                        cond, current_idx_recursive
                    )
                )
                if fragment and fragment not in ("TRUE", "FALSE"):
                    sql_fragments.append(f"({fragment})")
                    all_params.extend(params)
                elif fragment == "FALSE" and expression.operator.upper() == "AND":
                    return ("FALSE", [], current_idx_recursive)
                elif fragment == "TRUE" and expression.operator.upper() == "OR":
                    return ("TRUE", [], current_idx_recursive)

            if not sql_fragments:
                return ("TRUE", [], current_idx_recursive)

            logical_op_sql = f" {expression.operator.upper()} "
            return (
                logical_op_sql.join(sql_fragments),
                all_params,
                current_idx_recursive
            )

        else:
            raise TypeError(f"Unknown QueryExpression type: {type(expression)}")

    def _translate_update(self, update: Update) -> Dict[str, Any]:
        """Translate Update object into PostgreSQL SET clause and params."""
        set_clauses: List[str] = []
        params: List[Any] = []
        current_param_index = 1

        operations = update.build()
        if not operations:
            return {"set": "", "params": []}

        for op in operations:
            field = op.field_path
            is_nested = "." in field
            base_column_name = field.split(".", 1)[0] if is_nested else field
            quoted_base_column = f'"{base_column_name}"'

            # Prepare path array literal for JSONB ops if nested
            path_array_sql: Optional[str] = None
            if is_nested:
                path_parts = field.split(".")[1:]
                # Ensure parts are quoted correctly for ARRAY literal
                quoted_parts = [f"'{p}'" for p in path_parts]
                path_array_sql = f"ARRAY[{','.join(quoted_parts)}]"

            idx = current_param_index # Capture current index for this op

            if isinstance(op, SetOperation):
                params.append(op.value)
                current_param_index += 1
                if is_nested and path_array_sql:
                    # Use jsonb_set. Handle base NULL. Cast value param.
                    set_clauses.append(
                        f'{quoted_base_column} = jsonb_set('
                        f"COALESCE({quoted_base_column}, '{{}}'::jsonb), "
                        f"{path_array_sql}, "
                        f"${idx}::jsonb, "
                        f"true)"  # create_missing = true
                    )
                else:
                    set_clauses.append(f"{quoted_base_column} = ${idx}")

            elif isinstance(op, UnsetOperation):
                if is_nested and path_array_sql:
                    # Use #- operator. Handle base NULL.
                    set_clauses.append(
                        f"{quoted_base_column} = "
                        f"COALESCE({quoted_base_column}, '{{}}'::jsonb) #- "
                        f"{path_array_sql}"
                    )
                    # No parameter needed for #- with literal path array
                else:
                    set_clauses.append(f"{quoted_base_column} = NULL")

            elif isinstance(op, (IncrementOperation, MultiplyOperation)):
                if is_nested: raise NotImplementedError("Nested numeric ops.")
                op_sql = "+" if isinstance(op, IncrementOperation) else "*"
                val = op.amount if isinstance(op, IncrementOperation) else op.factor
                params.append(val)
                current_param_index += 1
                set_clauses.append(
                    f"{quoted_base_column} = "
                    f"COALESCE({quoted_base_column}, 0) {op_sql} ${idx}"
                )

            elif isinstance(op, (MinOperation, MaxOperation)):
                if is_nested: raise NotImplementedError("Nested min/max ops.")
                func_sql = "LEAST" if isinstance(op, MinOperation) else "GREATEST"
                params.append(op.value)
                current_param_index += 1
                # Use COALESCE with the value itself for correct comparison if NULL
                set_clauses.append(
                    f"{quoted_base_column} = {func_sql}("
                    f"COALESCE({quoted_base_column}, ${idx}), ${idx})"
                )


            elif isinstance(op, PushOperation):
                if not op.items: continue

                concat_parts = []
                for item in op.items:
                    params.append(item)
                    item_idx = current_param_index; current_param_index += 1
                    concat_parts.append(f"${item_idx}::jsonb")
                concat_clause = " || ".join(concat_parts)

                if is_nested:
                    path_parts = field.split('.')[1:]
                    path_array_sql = "ARRAY[" + ",".join(f"'{p}'" for p in path_parts) + "]"
                    # Path for #> operator (e.g., '{emails}')
                    path_curly_braces = "{" + ",".join(path_parts) + "}"

                    # Extract existing array using #>, default to '[]' if path null/invalid
                    existing_array_expr = (
                        f"COALESCE({quoted_base_column} #> '{path_curly_braces}', '[]'::jsonb)"
                    )

                    # Ensure the extracted value IS an array before concatenating
                    # This adds complexity back, but might be necessary
                    safe_concat_expr = (
                        f"CASE "
                        f"WHEN jsonb_typeof({existing_array_expr}) = 'array' "
                        f"THEN {existing_array_expr} || {concat_clause} "
                        f"ELSE '[]'::jsonb || {concat_clause} " # Start new array if not array
                        f"END"
                    )

                    set_clauses.append(
                        f'{quoted_base_column} = jsonb_set('
                        f"COALESCE({quoted_base_column}, '{{}}'::jsonb), "
                        f"{path_array_sql}, "
                        # f"({existing_array_expr} || {concat_clause}), " # Old version
                        f"{safe_concat_expr}, " # New version with type check
                        f"true)"
                    )
                else:
                    # Top-Level Push (remains same)
                    set_clauses.append(
                        f"{quoted_base_column} = "
                        f"COALESCE({quoted_base_column}, '[]'::jsonb) || "
                        f"{concat_clause}"
                    )


            elif isinstance(op, PopOperation):
                index_str = "-1" if op.position == 1 else "0" # PG #- path index
                if is_nested:
                    path_parts = field.split(".")[1:] + [index_str]
                    quoted_parts = [f"'{p}'" for p in path_parts]
                    pop_path_array = f"ARRAY[{','.join(quoted_parts)}]"
                    # Handle base NULL
                    set_clauses.append(
                        f"{quoted_base_column} = "
                        f"COALESCE({quoted_base_column}, '{{}}'::jsonb) #- {pop_path_array}"
                    )
                else: # Top level
                    pop_path_array = f"ARRAY['{index_str}']"
                     # Handle base NULL, ensure it's treated as array if NULL
                    set_clauses.append(
                        f"{quoted_base_column} = "
                        f"COALESCE({quoted_base_column}, '[]'::jsonb) #- {pop_path_array}"
                    )
                # No parameters needed for #-

            elif isinstance(op, PullOperation):
                # ... (common setup: prepare value_to_pull, param_value, where_clause_subquery) ...
                if isinstance(op.value_or_condition, dict):
                    raise NotImplementedError("Dict criteria not supported.")
                value_to_pull = op.value_or_condition
                # ... (param_value preparation) ...
                params.append(value_to_pull)
                val_idx = current_param_index
                current_param_index += 1
                where_clause_subquery = f"elem #>> '{{}}' != ${val_idx}::text"
                if value_to_pull is None:
                    where_clause_subquery = "elem IS NOT NULL"

                if is_nested:
                    # --- Nested Pull Logic (using jsonb_set) ---
                    path_parts = field.split('.')[1:]
                    path_array_sql = "ARRAY[" + ",".join(
                        f"'{p}'" for p in path_parts) + "]"
                    jsonpath_suffix = '.'.join(
                        f'"{p}"' if not p.isdigit() else f'[{p}]' for p in path_parts)
                    jsonpath_expr = f"('$.{jsonpath_suffix}')::jsonpath"

                    filtered_array_subquery = f"""
                                ( SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                  FROM jsonb_array_elements(
                                      COALESCE(jsonb_path_query_first( -- Use path_query_first for safety
                                          COALESCE({quoted_base_column}, '{{}}'::jsonb),
                                          {jsonpath_expr}
                                      ), '[]'::jsonb)
                                  ) AS elem
                                  WHERE {where_clause_subquery}
                                )
                                """
                    # Correct: Use jsonb_set for nested paths
                    set_clauses.append(
                        f'{quoted_base_column} = jsonb_set('
                        f"COALESCE({quoted_base_column}, '{{}}'::jsonb), "
                        f"{path_array_sql}, "
                        f"{filtered_array_subquery}, "
                        f"false)"  # create_missing = false
                    )
                else:
                    # --- Top-Level Pull Logic (Direct Assignment) ---
                    quoted_field = quoted_base_column  # e.g., "tags"
                    jsonpath_expr = "'$'::jsonpath"  # Path for jsonb_path_query_first

                    # Subquery to rebuild array without the pulled value
                    filtered_array_subquery = f"""
                                ( SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                  FROM jsonb_array_elements(
                                       -- Get the top-level array, default to [] if null/not found
                                       COALESCE(jsonb_path_query_first(
                                           COALESCE({quoted_field}, '{{}}'::jsonb),
                                           {jsonpath_expr}
                                       ), '[]'::jsonb)
                                  ) AS elem
                                  WHERE {where_clause_subquery}
                                )
                                """
                    # <<< Correct: Directly assign the subquery result >>>
                    set_clauses.append(
                        f"{quoted_field} = {filtered_array_subquery}"
                    )


            else:
                raise TypeError(f"Unsupported UpdateOperation: {type(op)}")

        set_clause_str = ", ".join(set_clauses)
        return {"set": set_clause_str, "params": params}


# src/async_repository/backends/postgres_repository.py (within PostgresRepository class)

    def _handle_db_error(self, error: Exception, context: str = "") -> None:
        """
        Map specific database errors or internal errors to appropriate exceptions.

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

        # --- Handle asyncpg specific errors ---
        if isinstance(error, asyncpg.PostgresError):
            # Log database errors with traceback for debugging
            self._logger.error(log_message, exc_info=True)

            # UniqueViolationError -> KeyAlreadyExistsException or ValueError
            if isinstance(error, asyncpg.UniqueViolationError):
                # Check if it's the primary key constraint (common naming convention)
                if self.db_id_field in str(error.detail or "") or (
                    e.constraint_name and f"{self._table_name}_pkey" in e.constraint_name
                ):
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
                        f"Unique constraint '{error.constraint_name}' violated during "
                        f"{context}. Detail: {error}"
                    ) from error

            # NotNullViolationError -> ValueError
            elif isinstance(error, asyncpg.NotNullViolationError):
                raise ValueError(
                    f"NOT NULL constraint violated for column "
                    f"'{error.column_name}' during {context}. Detail: {error}"
                ) from error

            # ForeignKeyViolationError -> ValueError
            elif isinstance(error, asyncpg.ForeignKeyViolationError):
                raise ValueError(
                    f"Foreign key constraint '{error.constraint_name}' "
                    f"violated during {context}. Detail: {error}"
                ) from error

            # CheckViolationError -> ValueError
            elif isinstance(error, asyncpg.CheckViolationError):
                raise ValueError(
                    f"Check constraint '{error.constraint_name}' violated "
                    f"during {context}. Detail: {error}"
                ) from error

            # Authorization / Privilege Errors -> RuntimeError
            elif isinstance(
                error,
                (
                    asyncpg.InsufficientPrivilegeError,
                    asyncpg.InvalidAuthorizationSpecificationError,
                ),
            ):
                raise RuntimeError(
                    f"DB authorization/privilege error during {context}. "
                    f"Detail: {error}"
                ) from error

            # Data errors (invalid format, syntax in data) -> ValueError
            elif isinstance(error, asyncpg.DataError):
                 # Catch common data format issues
                 if ('invalid input syntax' in str(error)
                     or 'invalid jsonb input' in str(error)
                     or 'invalid uuid' in str(error) # Example if using UUIDs
                     or 'invalid datetime format' in str(error)):
                    raise ValueError(
                        f"Invalid data format encountered during {context}. "
                        f"Detail: {error}"
                    ) from error
                 else: # Other data errors
                    raise RuntimeError(
                        f"Database data error during {context}: {error}"
                    ) from error


            # Schema Mismatch Errors -> RuntimeError
            elif isinstance(
                error,
                (
                    asyncpg.UndefinedTableError,
                    asyncpg.UndefinedColumnError,
                    asyncpg.UndefinedFunctionError,
                ),
            ):
                raise RuntimeError(
                    f"DB schema mismatch or missing function during "
                    f"{context}. Detail: {error}"
                ) from error

            # Syntax errors likely indicate a bug in query generation
            elif isinstance(error, asyncpg.PostgresSyntaxError):
                 self._logger.error(
                     "PostgresSyntaxError indicates a likely bug in SQL generation.",
                     exc_info=True
                 )
                 raise RuntimeError(
                    f"Invalid SQL syntax generated during {context}. "
                    f"Detail: {error}"
                 ) from error


            # Catch-all for other specific Postgres errors -> RuntimeError
            else:
                raise RuntimeError(
                    f"A specific database error occurred during {context}: {error}"
                ) from error

        # --- Handle specific non-DB errors from repository/framework logic ---
        elif isinstance(
            error,
            (
                ValueError,  # Includes unsupported ops, bad values, translation errors
                TypeError,  # Bad types passed to operations
                NotImplementedError, # Features not implemented
                InvalidPathError, # From ModelValidator
                KeyAlreadyExistsException, # Re-raise if passed through
                ObjectNotFoundException, # Re-raise if passed through
            )
        ):
            # Log potentially expected errors without traceback
            self._logger.error(log_message)
            raise error # Re-raise the original specific exception

        # --- Handle truly unexpected errors ---
        else:
            # Log with traceback as it's not a recognized DB or framework error
            self._logger.error(log_message, exc_info=True)
            raise RuntimeError(
                f"An unexpected error occurred during {context}"
            ) from error

