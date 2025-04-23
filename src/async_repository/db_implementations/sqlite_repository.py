# src/async_repository/backends/sqlite_repository.py

import json
import logging
import re
from contextlib import asynccontextmanager
from logging import LoggerAdapter
from typing import (Any, AsyncGenerator, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar, get_type_hints, get_origin, get_args, Union, Set)

# --- aiosqlite Driver Import ---
import aiosqlite

# --- Framework Imports ---
from async_repository.base.exceptions import (KeyAlreadyExistsException,
                                              ObjectNotFoundException)
from async_repository.base.interfaces import Repository
from async_repository.base.model_validator import _is_none_type
from async_repository.base.query import (QueryExpression, QueryFilter,
                                         QueryLogical, QueryOperator,
                                         QueryOptions)
from async_repository.base.update import (MaxOperation, MinOperation,
                                          MultiplyOperation, PopOperation,
                                          PullOperation, PushOperation,
                                          SetOperation, UnsetOperation, Update,
                                          UpdateOperation, IncrementOperation)
from async_repository.base.utils import prepare_for_storage

# --- Type Variables ---
T = TypeVar("T")  # Your entity type
DB_CONNECTION_TYPE = aiosqlite.Connection
# Cursor not strictly needed as we use conn.execute directly often
DB_CURSOR_TYPE = aiosqlite.Cursor
DB_RECORD_TYPE = aiosqlite.Row  # Using default row factory

base_logger = logging.getLogger("async_repository.backends.sqlite_repository")


class SqliteRepository(Repository[T], Generic[T]):
    """
    SQLite repository implementation using aiosqlite.

    This repository expects an active `aiosqlite.Connection` to be provided
    during initialization, typically managed by a Unit of Work or Service Layer
    that handles transaction boundaries (commit/rollback).

    Assumes complex types (lists, dicts) are stored as JSON strings.
    Assumes the app_id_field is the PRIMARY KEY (TEXT).
    Push/Pop/Pull operations on JSON arrays are NOT implemented by default.
    """

    # --- Initialization ---
    def __init__(
        self,
        db_connection: aiosqlite.Connection,
        table_name: str,
        entity_type: Type[T],
        app_id_field: str = "id",
        db_id_field: Optional[str] = None,
    ):
        """
        Initialize the SQLite repository with an existing connection.

        Args:
            db_connection: An active aiosqlite.Connection object managed externally.
            table_name: The name of the database table.
            entity_type: The Python class representing the entity.
            app_id_field: The attribute/column name for the application ID (used as PK).
            db_id_field: Database-specific ID field name. If None, defaults to app_id_field.
        """
        if not isinstance(db_connection, aiosqlite.Connection):
            raise TypeError(
                "db_connection must be an instance of aiosqlite.Connection"
            )

        self._conn = db_connection
        # Ensure connection uses dict-like rows for convenience
        self._conn.row_factory = aiosqlite.Row

        self._table_name = table_name
        self._entity_type = entity_type
        self._app_id_field = app_id_field
        self._db_id_field = (
            db_id_field if db_id_field is not None else app_id_field
        )

        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}[{entity_type.__name__}]"
        )
        self._logger.info(
            f"Repository instance created for {entity_type.__name__} using table '{table_name}'."
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

    # --- Connection/Session Management (UoW Aware) ---
    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Provides the externally managed connection within a context.
        Does NOT handle commit/rollback; expects the caller (UoW) to manage it.
        Provides a try/except block for operations using the connection.
        """
        try:
            # Yield the connection passed during __init__
            yield self._conn
        except Exception as e:
            # Log error, but let the Unit of Work handle rollback by re-raising
            self._logger.error(
                f"Error during repository operation within external transaction: {e}",
                exc_info=True,
            )
            raise  # Re-raise the exception to signal failure to the UoW

    # --- Schema and Index Implementation ---

    async def check_schema(self, logger: LoggerAdapter) -> bool:
        """Check if the schema (table) exists."""
        logger.info(f"Checking schema for '{self._table_name}'...")
        try:
            # Use the connection directly, no separate transaction needed for check
            async with self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (self._table_name,),
            ) as cursor:
                exists = await cursor.fetchone() is not None
            if exists:
                logger.info(f"Schema check PASSED for '{self._table_name}'.")
                # TODO: Optionally add column checks here for more robustness
                return True
            else:
                logger.warning(
                    f"Schema check FAILED: Table '{self._table_name}' not found."
                )
                return False
        except Exception as e:
            logger.error(
                f"Error during schema check for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"checking schema for {self._table_name}")
            return False

    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """Check if essential indexes (on app_id_field) seem to exist."""
        logger.info(f"Checking essential indexes for '{self._table_name}'...")
        # Ensure the app_id_field isn't the same as the rowid alias or PK if it's INTEGER
        # For TEXT PKs (our assumption), we need an explicit index or rely on PK constraint
        pk_check_sql = f"PRAGMA index_list('{self._table_name}');"
        idx_exists = False
        try:
            # Check if app_id_field is part of the primary key
            pk_info_sql = f"PRAGMA table_info('{self._table_name}')"
            is_pk = False
            async with self._conn.execute(pk_info_sql) as cursor:
                async for row in cursor:
                    if row["name"] == self.app_id_field and row["pk"] > 0:
                        is_pk = True
                        break
            if is_pk:
                 logger.info(f"Index check: '{self.app_id_field}' is part of PRIMARY KEY for '{self._table_name}'.")
                 idx_exists = True # PK implies an index
            else:
                # If not PK, check for an explicit index involving the app_id_field
                # This check is basic and might need refinement for multi-column indexes
                 check_sql = f"SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND (sql LIKE ? OR sql LIKE ?);"
                 # Check for patterns like CREATE INDEX ... ON ... (app_id_field ...) or UNIQUE (app_id_field ...)
                 like_pattern1 = f"%({self.app_id_field}%"
                 like_pattern2 = f"%UNIQUE ({self.app_id_field}%"
                 async with self._conn.execute(check_sql, (self._table_name, like_pattern1, like_pattern2)) as cursor:
                     idx_exists = await cursor.fetchone() is not None

            if idx_exists:
                logger.info(
                    f"Index check PASSED for '{self._table_name}' (found PK or index on '{self.app_id_field}')."
                )
                return True
            else:
                logger.warning(
                    f"Index check FAILED: PK or explicit index on '{self.app_id_field}' "
                    f"for table '{self._table_name}' not found."
                )
                return False
        except Exception as e:
            logger.error(
                f"Error during index check for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"checking indexes for {self._table_name}")
            return False

    async def create_schema(self, logger: LoggerAdapter) -> None:
        """Explicitly create the table if it doesn't exist."""
        logger.info(
            f"Attempting to create schema (table '{self._table_name}')..."
        )
        # NOTE: Inferring column types accurately is complex.
        # This example assumes TEXT PRIMARY KEY for app_id and TEXT for others.
        pk_col = self.db_id_field  # Assumed PK column name
        cols_def = [f'"{pk_col}" TEXT PRIMARY KEY NOT NULL']
        try:
            hints = get_type_hints(self.entity_type)
            for name, hint in hints.items():
                if name == pk_col or name.startswith("_"):
                    continue
                # Basic type mapping (customize as needed)
                origin = get_origin(hint)
                if origin is Union and _is_none_type(get_args(hint)[-1]):
                     col_type = "TEXT" # Default nullable TEXT
                     # Get actual type from Optional[T]
                     actual_type = get_args(hint)[0]
                else:
                     col_type = "TEXT NOT NULL" # Default non-null TEXT
                     actual_type = hint

                if actual_type is int: col_type = "INTEGER" + ("" if origin is Union else " NOT NULL")
                elif actual_type is float: col_type = "REAL" + ("" if origin is Union else " NOT NULL")
                elif actual_type is bool: col_type = "INTEGER" + ("" if origin is Union else " NOT NULL") # 0/1
                elif actual_type is bytes: col_type = "BLOB" + ("" if origin is Union else " NOT NULL")
                # Complex types default to TEXT (for JSON)
                cols_def.append(f'"{name}" {col_type}')

        except Exception as e:
            logger.warning(
                f"Could not fully infer columns from type hints for {self.entity_type.__name__}: {e}"
            )
            # Proceed with basic schema anyway? Or raise error?

        create_sql = f'CREATE TABLE IF NOT EXISTS "{self._table_name}" ({", ".join(cols_def)})'

        try:
            # Use the session context manager even for schema changes
            # The UoW would commit this change if called within its scope.
            async with self._get_session() as conn:
                await conn.execute(create_sql)
            logger.info(
                f"Schema creation/verification complete for '{self._table_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to create schema for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"creating schema for {self._table_name}")

    async def create_indexes(self, logger: LoggerAdapter) -> None:
        """Explicitly create necessary database indexes if they don't exist."""
        logger.info(
            f"Attempting to create indexes for '{self._table_name}'..."
        )
        indexes_to_create = []
        # --- Define required indexes here ---
        # 1. Unique index on app_id_field (if it's not the PK, though our init assumes it is)
        #    SQLite automatically creates an index for PRIMARY KEY and UNIQUE constraints.
        #    So, if app_id_field is the TEXT PK, we might not need an explicit separate index.
        #    If app_id_field is just UNIQUE NOT NULL, SQLite creates an index.
        #    If app_id_field is just TEXT and needs to be unique, create one:
        if self.app_id_field != self.db_id_field: # Only if app_id isn't the PK
             index_name = f"idx_{self._table_name}_{self.app_id_field}_unique"
             sql = f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{self._table_name}"("{self.app_id_field}");'
             indexes_to_create.append(sql)

        # 2. Add other indexes based on common query patterns
        # Example: index_name = f"idx_{self._table_name}_some_field"
        # Example: sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self._table_name}"("some_field");'
        # indexes_to_create.append(sql)

        if not indexes_to_create:
             logger.info(f"No explicit indexes defined for creation on '{self._table_name}'.")
             return

        try:
            async with self._get_session() as conn:
                for sql in indexes_to_create:
                    logger.debug(f"Executing index SQL: {sql}")
                    await conn.execute(sql)
            logger.info(
                f"Index creation/verification complete for '{self._table_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to create indexes for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"creating indexes for {self._table_name}")

    # --- Core CRUD Method Implementations ---
    # Implementations need to use 'async with self._get_session() as conn:'
    # but should NOT call conn.commit() or conn.rollback() themselves.

    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        field_to_match = self.db_id_field
        logger.debug(
            f"Getting {self.entity_type.__name__} by {field_to_match}='{id}'"
        )
        query = f'SELECT * FROM "{self._table_name}" WHERE "{field_to_match}" = ? LIMIT 1'
        params = (id,)
        execute_kwargs = {}
        if timeout is not None: execute_kwargs["timeout"] = timeout

        try:
            async with self._get_session() as conn:
                async with conn.execute(query, params, **execute_kwargs) as cursor:
                    record_data = await cursor.fetchone()

            if record_data is None:
                id_type = "database (PK)" if use_db_id else f"application ({self.app_id_field})"
                logger.warning(
                    f"{self.entity_type.__name__} with {id_type} '{id}' not found."
                )
                raise ObjectNotFoundException(
                    f"{self.entity_type.__name__} with ID '{id}' not found."
                )

            entity = self._deserialize_record(record_data)
            logger.info(
                f"Retrieved {self.entity_type.__name__} with {field_to_match} '{id}'"
            )
            return entity
        except Exception as e:
            self._handle_db_error(e, f"getting entity ID {id}")


    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
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
        self.validate_entity(entity)
        app_id = getattr(entity, self.app_id_field, None)

        if generate_app_id and app_id is None:
            app_id = self.id_generator()
            setattr(entity, self.app_id_field, app_id)
            logger.debug(
                f"Generated application ID '{app_id}' for new {self.entity_type.__name__}"
            )
        elif app_id is None:
            raise ValueError(
                f"Entity of type {self.entity_type.__name__} must have an ID "
                f"field '{self.app_id_field}' set or generate_app_id must be True."
            )

        logger.debug(
            f"Storing new {self.entity_type.__name__} with {self.app_id_field} '{app_id}'"
        )
        try:
            db_data = self._serialize_entity(entity)
            if self.db_id_field not in db_data:
                db_data[self.db_id_field] = app_id # Ensure PK is present

            cols = ", ".join(f'"{k}"' for k in db_data.keys())
            placeholders = ", ".join(["?"] * len(db_data))
            query = f'INSERT INTO "{self._table_name}" ({cols}) VALUES ({placeholders})'
            params = tuple(db_data.values())

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                await conn.execute(query, params, **execute_kwargs)

            logger.info(
                f"Stored new {self.entity_type.__name__} with {self.app_id_field} '{app_id}'. "
                f"(Commit handled externally)"
            )

            return entity if return_value else None

        except aiosqlite.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e) and self.db_id_field in str(e):
                logger.warning(
                    f"Failed to store {self.entity_type.__name__}: ID '{app_id}' "
                    f"already exists (PK constraint). Details: {e}"
                )
                raise KeyAlreadyExistsException(
                    f"Entity with ID '{app_id}' already exists."
                ) from e
            else:
                self._handle_db_error(e, f"storing entity app_id {app_id}")
        except Exception as e:
            self._handle_db_error(e, f"storing entity app_id {app_id}")

    async def upsert(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
    ) -> None:
        self.validate_entity(entity)
        app_id = getattr(entity, self.app_id_field, None)

        if generate_app_id and app_id is None:
            app_id = self.id_generator()
            setattr(entity, self.app_id_field, app_id)
            logger.debug(
                f"Generated potential application ID '{app_id}' for upserting {self.entity_type.__name__}"
            )
        elif app_id is None:
            raise ValueError(
                f"Entity for upsert must have ID field '{self.app_id_field}' set or generate_app_id must be True."
            )

        logger.debug(
            f"Upserting {self.entity_type.__name__} with {self.app_id_field} '{app_id}'"
        )
        try:
            db_data = self._serialize_entity(entity)
            if self.db_id_field not in db_data:
                db_data[self.db_id_field] = app_id

            cols = ", ".join(f'"{k}"' for k in db_data.keys())
            placeholders = ", ".join(["?"] * len(db_data))
            update_clause = ", ".join(
                f'"{k}" = excluded."{k}"'
                for k in db_data.keys()
                if k != self.db_id_field
            )

            # Requires SQLite 3.24+
            query = (
                f'INSERT INTO "{self._table_name}" ({cols}) VALUES ({placeholders}) '
                f'ON CONFLICT("{self.db_id_field}") DO UPDATE SET {update_clause}'
            )
            params = tuple(db_data.values())

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                await conn.execute(query, params, **execute_kwargs)

            logger.info(
                f"Upserted {self.entity_type.__name__} with ID '{app_id}'. "
                f"(Commit handled externally)"
            )
        except Exception as e:
            self._handle_db_error(e, f"upserting entity app_id {app_id}")

    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        logger.debug(
            f"Updating one {self.entity_type.__name__} matching: {options!r} with update: {update!r}"
        )
        if not update:
            logger.warning("update_one called with empty update operations.")
            if return_value: return await self.find_one(logger=logger, options=options)
            return None
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update_one.")

        execute_kwargs = {}
        if timeout is not None: execute_kwargs["timeout"] = timeout

        # Non-atomic approach for SQLite without RETURNING
        target_id = None
        try:
            async with self._get_session() as conn:
                # 1. Find the ID
                find_options = options.copy() if hasattr(options, 'copy') else options # Assume copy exists
                find_options.limit = 1
                find_parts = self._translate_query_options(find_options, False)
                find_sql = f'SELECT "{self.db_id_field}" FROM "{self._table_name}" WHERE {find_parts["where"]} LIMIT 1'

                async with conn.execute(find_sql, find_parts["params"], **execute_kwargs) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        raise ObjectNotFoundException(f"No {self.entity_type.__name__} found matching criteria for update.")
                    target_id = row[self.db_id_field]

                # 2. Perform Update
                update_parts = self._translate_update(update)
                if not update_parts.get("set"): # Handle empty update after translation
                    logger.warning("Update resulted in no SET clauses, skipping DB call.")
                    if return_value: return await self.get(str(target_id), logger, use_db_id=True)
                    else: return None

                update_sql = f'UPDATE "{self._table_name}" SET {update_parts["set"]} WHERE "{self.db_id_field}" = ?'
                update_params = update_parts["params"] + [target_id]

                cursor = await conn.execute(update_sql, update_params, **execute_kwargs)

                if cursor.rowcount == 0:
                    logger.warning(f"Update target ID '{target_id}' was not found during update execution.")
                    raise ObjectNotFoundException(f"Entity with ID '{target_id}' disappeared before update.")
                elif cursor.rowcount > 1:
                    logger.warning(f"Update_one modified {cursor.rowcount} rows for ID '{target_id}'.")

            logger.info(
                f"Updated one {self.entity_type.__name__} with ID '{target_id}'. "
                f"(Commit handled externally)"
            )

            # 3. Re-fetch if needed
            if return_value:
                return await self.get(str(target_id), logger, use_db_id=True)
            else:
                return None
        except ObjectNotFoundException:
             raise # Propagate ObjectNotFoundException
        except Exception as e:
             self._handle_db_error(e, f"updating one entity (target_id: {target_id})")


    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(
            f"Updating many {self.entity_type.__name__} matching: {options!r} with update: {update!r}"
        )
        if not update:
            logger.warning("update_many called with empty update operations. Returning 0.")
            return 0
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update_many.")

        if options.limit or options.offset or options.sort_by or options.random_order:
            logger.warning("limit, offset, sort_by, random_order are ignored for SQLite update_many operation.")

        try:
            query_parts = self._translate_query_options(options, False)
            update_parts = self._translate_update(update)

            if not query_parts or not query_parts.get("where"):
                raise ValueError("Cannot update_many without a valid filter expression.")
            if not update_parts or not update_parts.get("set"):
                 logger.warning("Update resulted in no SET clauses, returning 0.")
                 return 0

            sql = f'UPDATE "{self._table_name}" SET {update_parts["set"]} WHERE {query_parts["where"]}'
            params = update_parts["params"] + query_parts["params"]

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                cursor = await conn.execute(sql, params, **execute_kwargs)
                affected_count = cursor.rowcount

            logger.info(
                f"Updated {affected_count} {self.entity_type.__name__}(s). "
                f"(Commit handled externally)"
            )
            return affected_count
        except Exception as e:
            self._handle_db_error(e, "updating many entities")


    async def delete_many(
        self,
        options: QueryOptions,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(
            f"Deleting many {self.entity_type.__name__}(s) matching: {options!r}"
        )
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete_many.")

        if options.limit or options.offset or options.sort_by or options.random_order:
            logger.warning("limit, offset, sort_by, random_order are ignored for SQLite delete_many operation.")

        try:
            query_parts = self._translate_query_options(options, False)
            if not query_parts or not query_parts.get("where"):
                raise ValueError("Cannot delete_many without a valid filter expression.")

            sql = f'DELETE FROM "{self._table_name}" WHERE {query_parts["where"]}'
            params = query_parts["params"]

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                cursor = await conn.execute(sql, params, **execute_kwargs)
                affected_count = cursor.rowcount

            logger.info(
                f"Deleted {affected_count} {self.entity_type.__name__}(s). "
                f"(Commit handled externally)"
            )
            return affected_count
        except Exception as e:
            self._handle_db_error(e, "deleting many entities")


    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        effective_options = options or QueryOptions()
        logger.debug(
            f"Listing {self.entity_type.__name__}(s) with options: {effective_options!r}"
        )

        execute_kwargs = {}
        if effective_options.timeout is not None:
            execute_kwargs["timeout"] = effective_options.timeout

        try:
            query_parts = self._translate_query_options(effective_options, True)
            sql = f'SELECT * FROM "{self._table_name}"'
            if query_parts.get("where"): sql += f" WHERE {query_parts['where']}"
            if query_parts.get("order"): sql += f" ORDER BY {query_parts['order']}"
            limit = query_parts.get("limit", -1) # Use -1 for unlimited
            if limit >= 0: sql += f" LIMIT ?"
            offset = query_parts.get("offset", 0)
            if offset > 0: sql += f" OFFSET ?"

            params = list(query_parts.get("params", []))
            if limit >= 0: params.append(limit)
            if offset > 0: params.append(offset)


            async with self._get_session() as conn:
                async with conn.execute(sql, params, **execute_kwargs) as cursor:
                    async for record_data in cursor:
                        try:
                            yield self._deserialize_record(record_data)
                        except Exception as deserialization_error:
                            logger.error(
                                f"Failed to deserialize record during list: {deserialization_error}. Record: {dict(record_data)}",
                                exc_info=True,
                            )
                            continue # Skip bad record
        except Exception as e:
            self._handle_db_error(e, "listing entities")


    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        effective_options = options or QueryOptions()
        logger.debug(
            f"Counting {self.entity_type.__name__}(s) with options: {effective_options!r}"
        )

        execute_kwargs = {}
        if effective_options.timeout is not None:
            execute_kwargs["timeout"] = effective_options.timeout

        try:
            query_parts = self._translate_query_options(effective_options, False)
            sql = f'SELECT COUNT(*) FROM "{self._table_name}"'
            params = []
            if query_parts.get("where"):
                sql += f" WHERE {query_parts['where']}"
                params = query_parts.get("params", [])

            async with self._get_session() as conn:
                async with conn.execute(sql, params, **execute_kwargs) as cursor:
                    result = await cursor.fetchone()
                    count_val = result[0] if result else 0

            logger.info(f"Counted {count_val} {self.entity_type.__name__}(s).")
            return int(count_val)
        except Exception as e:
            self._handle_db_error(e, "counting entities")


    # --- Helper Method Implementations ---

    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Converts the entity object into a dictionary suitable for SQLite."""
        # Basic serialization (adapt if not using Pydantic/dataclasses)
        if hasattr(entity, 'model_dump'):
             data = entity.model_dump(by_alias=False, mode='python')
        elif hasattr(entity, 'dict'):
             data = entity.dict(by_alias=False)
        elif hasattr(entity, '__dict__'):
             data = entity.__dict__
        else:
             raise TypeError(f"Cannot automatically serialize entity: {type(entity).__name__}")

        prepared_data = prepare_for_storage(data)
        serialized_data = {}
        for key, value in prepared_data.items():
            if key.startswith("_"): continue # Skip private attributes

            if isinstance(value, (dict, list, tuple, set)):
                try: serialized_data[key] = json.dumps(value)
                except TypeError: serialized_data[key] = None # Store NULL on JSON error
            elif isinstance(value, bool): serialized_data[key] = 1 if value else 0
            else: serialized_data[key] = value

        # Ensure PK mapping
        if self.app_id_field != self.db_id_field and self.app_id_field in serialized_data:
             serialized_data[self.db_id_field] = serialized_data[self.app_id_field]

        # Remove app_id_field if it's different from db_id_field and not a DB column
        # This depends on whether app_id_field *also* exists as a separate column.
        # Usually, if db_id is the PK, app_id might not be stored separately.
        # For this template, assume db_id_field is the only ID column stored if different.
        if self.app_id_field != self.db_id_field:
             serialized_data.pop(self.app_id_field, None)


        return serialized_data

    def _deserialize_record(self, record_data: aiosqlite.Row) -> T:
        """Converts an aiosqlite.Row into an entity object T."""
        if record_data is None: raise ValueError("Cannot deserialize None record data.")

        entity_dict = dict(record_data)
        processed_dict = {}
        hints = get_type_hints(self.entity_type)

        for key, value in entity_dict.items():
            target_type = hints.get(key)
            origin = get_origin(target_type)
            args = get_args(target_type)
            is_optional = origin is Union and _is_none_type(args[-1])
            actual_type = args[0] if is_optional else target_type

            # JSON Deserialization
            if isinstance(value, str) and get_origin(actual_type) in (list, List, dict, Dict, set, Set, tuple, Tuple):
                try: processed_dict[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError): processed_dict[key] = None if is_optional else value
            # Boolean Deserialization
            elif isinstance(value, int) and actual_type is bool:
                processed_dict[key] = bool(value)
            # Handle None for optional fields correctly
            elif value is None and is_optional:
                 processed_dict[key] = None
            else:
                 processed_dict[key] = value

        # Map PK (db_id_field) back to app_id_field if needed
        db_id_value = processed_dict.get(self.db_id_field)
        if db_id_value is not None:
            # Set app_id if it's different and not already present
            if self.app_id_field != self.db_id_field and self.app_id_field not in processed_dict:
                 processed_dict[self.app_id_field] = str(db_id_value)

            # Remove db_id_field if it's not an explicit field in the entity model
            entity_fields = getattr(self.entity_type, '__annotations__', {}).keys()
            if self.db_id_field not in entity_fields:
                 processed_dict.pop(self.db_id_field, None)

        try:
            return self.entity_type(**processed_dict)
        except Exception as e:
            self._logger.error(f"Failed to instantiate {self.entity_type.__name__}: {e}. Data: {processed_dict!r}", exc_info=True)
            raise ValueError(f"Failed to deserialize record into {self.entity_type.__name__}") from e

    def _translate_query_options(
        self, options: QueryOptions, include_sorting_pagination: bool = False
    ) -> Dict[str, Any]:
        """Translates QueryOptions into SQLite WHERE, ORDER BY, LIMIT, OFFSET parts."""
        sql_parts: Dict[str, Any] = {"where": "1=1", "params": []}

        if options.expression:
            where_clause, params = self._translate_expression_recursive(
                options.expression
            )
            # Only use WHERE clause if it's not empty or trivial
            if where_clause and where_clause != "1=1":
                sql_parts["where"] = where_clause
                sql_parts["params"] = params

        if include_sorting_pagination:
            if options.random_order:
                sql_parts["order"] = "RANDOM()"
            elif options.sort_by:
                direction = "DESC" if options.sort_desc else "ASC"
                # Quote sort column defensively
                sql_parts["order"] = f'"{options.sort_by}" {direction}'
            else:
                sql_parts["order"] = None

            sql_parts["limit"] = options.limit if options.limit > 0 else -1 # -1 for unlimited
            sql_parts["offset"] = options.offset

        return sql_parts

    def _translate_expression_recursive(
        self, expression: QueryExpression
    ) -> Tuple[str, List[Any]]:
        """Recursively translates QueryExpression nodes into SQLite WHERE clause + params."""
        if isinstance(expression, QueryFilter):
            field = expression.field_path
            op = expression.operator
            val = expression.value  # Value prepared by builder

            # Quote field name
            quoted_field = f'"{field}"'

            sql_op_map = {
                QueryOperator.EQ: "=", QueryOperator.NE: "!=",
                QueryOperator.GT: ">", QueryOperator.GTE: ">=",
                QueryOperator.LT: "<", QueryOperator.LTE: "<=",
                QueryOperator.IN: "IN", QueryOperator.NIN: "NOT IN",
                QueryOperator.LIKE: "LIKE", QueryOperator.STARTSWITH: "LIKE",
                QueryOperator.ENDSWITH: "LIKE", QueryOperator.CONTAINS: "LIKE",
                QueryOperator.EXISTS: "IS NOT NULL", # Base case
            }
            sql_op = sql_op_map.get(op)
            params: List[Any] = []

            if sql_op:
                if op == QueryOperator.EXISTS:
                    # Value determines IS NULL or IS NOT NULL
                    sql_op = "IS NOT NULL" if val else "IS NULL"
                    return f"{quoted_field} {sql_op}", [] # No parameter
                elif op == QueryOperator.IN or op == QueryOperator.NIN:
                    if not isinstance(val, (list, tuple)) or not val:
                        return ("0=1", []) if op == QueryOperator.IN else ("1=1", [])
                    placeholders = ", ".join(["?"] * len(val))
                    params.extend(val)
                    return f"{quoted_field} {sql_op} ({placeholders})", params
                elif op == QueryOperator.STARTSWITH:
                    params.append(f"{val}%")
                    return f"{quoted_field} {sql_op} ?", params
                elif op == QueryOperator.ENDSWITH:
                    params.append(f"%{val}")
                    return f"{quoted_field} {sql_op} ?", params
                elif op == QueryOperator.CONTAINS:
                    # Check if target field is TEXT (implying JSON string)
                    # Requires knowledge of schema or hints - complex for generic translator
                    # Assuming LIKE for non-JSON, requires JSON functions for JSON search
                    self._logger.warning(f"CONTAINS operator for SQLite field '{field}' translated to LIKE. For JSON columns, use specific JSON functions.")
                    params.append(f"%{val}%")
                    return f"{quoted_field} {sql_op} ?", params
                else: # EQ, NE, GT, LT, GTE, LTE, LIKE
                    params.append(val)
                    return f"{quoted_field} {sql_op} ?", params
            else:
                raise ValueError(f"Unsupported query operator for SQLite: {op}")

        elif isinstance(expression, QueryLogical):
            if not expression.conditions: return ("1=1", [])
            sql_fragments = []
            all_params: List[Any] = []
            for cond in expression.conditions:
                fragment, params = self._translate_expression_recursive(cond)
                if fragment and fragment != "1=1": # Avoid including trivial conditions
                    sql_fragments.append(f"({fragment})")
                    all_params.extend(params)
            if not sql_fragments: return ("1=1", []) # All conditions were trivial
            logical_op_sql = " AND " if expression.operator == "and" else " OR "
            return logical_op_sql.join(sql_fragments), all_params
        else:
            raise TypeError(f"Unknown QueryExpression type: {type(expression)}")

    def _translate_update(self, update: Update) -> Dict[str, Any]:
        """Translates Update object into SQLite SET clause and params."""
        set_clauses: List[str] = []
        params: List[Any] = []

        for op in update.build():
            field = op.field_path
            quoted_field = f'"{field}"'

            if isinstance(op, SetOperation):
                set_clauses.append(f"{quoted_field} = ?")
                value = op.value # Value prepared by builder
                # Serialize complex types to JSON
                if isinstance(value, (dict, list, tuple, set)):
                    params.append(json.dumps(value))
                elif isinstance(value, bool): params.append(1 if value else 0)
                else: params.append(value)
            elif isinstance(op, UnsetOperation):
                set_clauses.append(f"{quoted_field} = NULL")
            elif isinstance(op, IncrementOperation):
                set_clauses.append(f"{quoted_field} = COALESCE({quoted_field}, 0) + ?") # Use COALESCE for potentially NULL fields
                params.append(op.amount)
            elif isinstance(op, MultiplyOperation):
                set_clauses.append(f"{quoted_field} = COALESCE({quoted_field}, 0) * ?")
                params.append(op.factor)
            elif isinstance(op, MinOperation):
                set_clauses.append(f"{quoted_field} = MIN(COALESCE({quoted_field}, ?), ?)") # Use MIN for SQLite
                params.extend([op.value, op.value]) # Need value twice for COALESCE and MIN
            elif isinstance(op, MaxOperation):
                set_clauses.append(f"{quoted_field} = MAX(COALESCE({quoted_field}, ?), ?)") # Use MAX for SQLite
                params.extend([op.value, op.value])
            elif isinstance(op, (PushOperation, PopOperation, PullOperation)):
                self._logger.error(f"SQLite backend does not support {type(op).__name__} directly via this template.")
                raise NotImplementedError(f"SQLite array operation {type(op).__name__} not implemented.")
            else:
                raise TypeError(f"Unsupported UpdateOperation type: {type(op)}")

        return {"set": ", ".join(set_clauses), "params": params}


    def _handle_db_error(self, error: Exception, context: str = "") -> None:
        """Maps specific database errors to repository exceptions or logs."""
        self._logger.error(
            f"SQLite error during {context}: {error}", exc_info=True
        )
        if isinstance(error, aiosqlite.IntegrityError):
            # Can check error message for more specifics if needed
            raise KeyAlreadyExistsException(
                f"Database integrity constraint violated during {context}."
            ) from error
        elif isinstance(error, aiosqlite.OperationalError):
            raise RuntimeError(
                f"Database operational error during {context}: {error}"
            ) from error
        else:
            raise RuntimeError(
                f"An unexpected database error occurred during {context}"
            ) from error