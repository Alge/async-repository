# src/async_repository/backends/sqlite_repository.py

import json
import logging
import re
from contextlib import asynccontextmanager
from logging import LoggerAdapter
from typing import (Any, AsyncGenerator, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar, get_type_hints, get_origin, get_args, Union, Set, Mapping) # Added Mapping
from dataclasses import is_dataclass # Import is_dataclass
from datetime import datetime, date # Import datetime types

# --- aiosqlite Driver Import ---
import aiosqlite

# --- Framework Imports ---
from async_repository.base.exceptions import (KeyAlreadyExistsException,
                                              ObjectNotFoundException)
from async_repository.base.interfaces import Repository
from async_repository.base.model_validator import _is_none_type, ModelValidator, InvalidPathError # Added InvalidPathError
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

    Assumes complex types (lists, dicts, sets, tuples, custom classes) are stored as JSON strings.
    Assumes datetime/date objects are stored as ISO 8601 strings (TEXT).
    Assumes the app_id_field maps to a column used as the PRIMARY KEY (typically TEXT).

    Features/Limitations:
        - Update operations for nested fields (e.g., 'dict.key') are supported for
          SetOperation and UnsetOperation using SQLite's json_set/json_remove functions.
        - Update: PushOperation is implemented using json_insert.
        - Update: PopOperation is implemented using json_remove.
        - Update: PullOperation is implemented for simple value removal using json_group_array/json_each. Conditional pull is NOT implemented.
        - Numerical/Min/Max operations on nested fields are NOT implemented.
        - Query operations involving 'contains' on list/array fields stored as JSON
          are implemented using SQLite's json_each function (requires SQLite >= 3.9.0
          with JSON1 extension enabled). Nested path support for CONTAINS is included.
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
            app_id_field: The attribute/column name for the application ID.
                          This field's corresponding column is assumed to be the PRIMARY KEY.
            db_id_field: Database-specific ID field name. If None, defaults to app_id_field.
                         This is the actual column name used as the PRIMARY KEY in the DB.
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
        # If db_id_field is not provided, it means the primary key column name
        # is the same as the application ID field name.
        self._db_id_field = (
            db_id_field if db_id_field is not None else app_id_field
        )

        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}[{entity_type.__name__}]"
        )
        # Initialize validator for type checking during query/update construction
        # It's okay if this fails, graceful fallback happens in QueryBuilder/Update
        try:
            self._validator = ModelValidator(entity_type)
        except Exception:
            self._validator = None # Proceed without validation if model is problematic
            self._logger.warning(f"Could not initialize ModelValidator for {entity_type.__name__}. Proceeding without validation.")

        self._logger.info(
            f"Repository instance created for {entity_type.__name__} using table '{table_name}' "
            f"(App ID Field: '{self._app_id_field}', DB PK Field: '{self._db_id_field}')."
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
            # Use the internal handler, which might raise a specific repo exception
            self._handle_db_error(e, f"checking schema for {self._table_name}")
            return False # Should not be reached if _handle_db_error raises


    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """Check if essential indexes (on the primary key field) seem to exist."""
        # SQLite automatically creates an index for PRIMARY KEY constraints.
        # This check verifies that the designated db_id_field is indeed part of the PK.
        logger.info(f"Checking essential indexes (PK on '{self.db_id_field}') for '{self._table_name}'...")
        pk_check_sql = f"PRAGMA table_info('{self._table_name}')"
        pk_found = False
        try:
            async with self._conn.execute(pk_check_sql) as cursor:
                async for row in cursor:
                    # Check if the column name matches db_id_field and if it's part of the PK (pk > 0)
                    if row["name"] == self.db_id_field and row["pk"] > 0:
                        pk_found = True
                        break

            if pk_found:
                logger.info(
                    f"Index check PASSED for '{self._table_name}': Field '{self.db_id_field}' is part of the PRIMARY KEY."
                )
                return True
            else:
                logger.warning(
                    f"Index check FAILED: Field '{self.db_id_field}' is NOT part of the PRIMARY KEY "
                    f"for table '{self._table_name}'. An explicit index might exist but primary key is recommended."
                )
                # Optionally, could add a check for explicit indexes like before, but PK is the main assumption.
                return False # Returning False as the primary assumption (PK index) failed.
        except Exception as e:
            logger.error(
                f"Error during index check for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"checking indexes for {self._table_name}")
            return False


    async def create_schema(self, logger: LoggerAdapter) -> None:
        """Explicitly create the table if it doesn't exist, inferring columns from type hints."""
        logger.info(
            f"Attempting to create schema (table '{self._table_name}')..."
        )
        # Use db_id_field as the primary key column name
        pk_col = self.db_id_field
        # Basic assumption: TEXT PK. Adjust if different numeric PK needed.
        cols_def = [f'"{pk_col}" TEXT PRIMARY KEY NOT NULL']

        processed_fields = {pk_col} # Keep track of fields added

        try:
            hints = get_type_hints(self.entity_type)
            for name, hint in hints.items():
                # Skip if already handled as PK or if it's the app_id field that's different from PK
                if name in processed_fields or (name == self.app_id_field and name != pk_col):
                    continue
                if name.startswith("_"): continue # Skip private attributes

                # Default to TEXT, allow NULL if Optional/Union with None
                col_type = "TEXT"
                is_optional = False
                origin = get_origin(hint)
                actual_type = hint

                if origin is Union and _is_none_type(get_args(hint)[-1]):
                    is_optional = True
                    actual_type = get_args(hint)[0] # Get the non-None type
                    # Check if the actual type is still a Union after removing None
                    origin = get_origin(actual_type) # Re-evaluate origin for nested checks
                    if origin is Union:
                         # Complex Optional[Union[...]] - treat as TEXT for simplicity
                         col_type = "TEXT" # Can store JSON representation
                    else:
                        # Simple Optional[T], determine T's SQL type
                        pass # Proceed to type mapping below
                elif origin is Union:
                    # Union without None, treat as TEXT for simplicity
                    col_type = "TEXT" # Can store JSON representation
                    actual_type = hint # Keep original hint

                # Map basic Python types to SQLite types
                if actual_type is int: col_type = "INTEGER"
                elif actual_type is float: col_type = "REAL"
                elif actual_type is bool: col_type = "INTEGER" # 0/1
                elif actual_type is bytes: col_type = "BLOB"
                elif actual_type is datetime: col_type = "TEXT" # Store datetime as ISO string
                elif actual_type is date: col_type = "TEXT" # Store date as ISO string
                elif actual_type in (str, Any): col_type = "TEXT"
                elif origin in (list, List, dict, Dict, set, Set, tuple, Tuple, Mapping) or (isinstance(actual_type, type) and (issubclass(actual_type, (list, dict, set, tuple)) or is_dataclass(actual_type) or hasattr(actual_type, 'model_dump'))):
                     # Check if it's a complex type origin OR if the concrete type is list/dict/set/tuple/dataclass/pydantic model
                     col_type = "TEXT" # Store complex types as JSON TEXT

                # Add NOT NULL constraint if not optional
                if not is_optional:
                    col_type += " NOT NULL"

                cols_def.append(f'"{name}" {col_type}')
                processed_fields.add(name)

        except Exception as e:
            logger.warning(
                f"Could not fully infer columns from type hints for {self.entity_type.__name__}: {e}. "
                f"Proceeding with inferred columns.", exc_info=True # Log traceback for debugging
            )
            # Proceed with the columns inferred so far

        create_sql = f'CREATE TABLE IF NOT EXISTS "{self._table_name}" ({", ".join(cols_def)})'
        logger.debug(f"Schema creation SQL: {create_sql}")

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
        """
        Explicitly create necessary database indexes if they don't exist.
        Assumes the PRIMARY KEY index on `db_id_field` is sufficient and automatically created.
        Creates a UNIQUE index on `app_id_field` only if it differs from `db_id_field`.
        """
        logger.info(
            f"Attempting to create indexes for '{self._table_name}'..."
        )
        indexes_to_create = []

        # SQLite automatically creates an index for PRIMARY KEY.
        # We only need an explicit index on app_id_field if it's different from db_id_field
        # AND needs to be unique/queried frequently.
        if self.app_id_field != self.db_id_field:
            # Also check if app_id_field actually exists as a column in the model definition
            field_exists = False
            try:
                hints = get_type_hints(self.entity_type)
                if self.app_id_field in hints:
                    field_exists = True
            except Exception:
                logger.warning(f"Could not get type hints to check if app_id_field '{self.app_id_field}' exists as a separate column.")

            if field_exists:
                index_name = f"idx_{self._table_name}_{self.app_id_field}_unique"
                # Assuming app_id_field should be unique if it's different from PK
                sql = f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{self._table_name}"("{self.app_id_field}");'
                indexes_to_create.append(sql)
                logger.info(f"Defining potential unique index on non-PK app_id_field '{self.app_id_field}'.")
            else:
                 logger.info(f"Skipping index creation for app_id_field '{self.app_id_field}' as it differs from PK ('{self.db_id_field}') but is not found as a distinct field in the model.")


        # --- Add other desirable indexes based on common query patterns ---
        # Example: Index on 'owner' field if frequently used in WHERE clauses
        # try:
        #     hints = get_type_hints(self.entity_type)
        #     if 'owner' in hints:
        #         index_name = f"idx_{self._table_name}_owner"
        #         sql = f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{self._table_name}"("owner");'
        #         indexes_to_create.append(sql)
        #         logger.info("Defining potential index on 'owner' field.")
        # except Exception:
        #     logger.warning("Could not check for 'owner' field to create index.")
        # ----------------------------------------------------------------

        if not indexes_to_create:
             logger.info(f"No explicit indexes defined for creation on '{self._table_name}' beyond the automatic PK index.")
             return

        try:
            async with self._get_session() as conn:
                for sql in indexes_to_create:
                    logger.debug(f"Executing index SQL: {sql}")
                    await conn.execute(sql)
            logger.info(
                f"Explicit index creation/verification complete for '{self._table_name}'."
            )
        except Exception as e:
            # Catch specific error if index already exists but is not UNIQUE, etc.
            logger.error(
                f"Failed to create indexes for '{self._table_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"creating indexes for {self._table_name}")


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
        id_type_str = "database (PK)" if use_db_id else f"application ('{self.app_id_field}')"
        logger.debug(
            f"Getting {self.entity_type.__name__} by {id_type_str}='{id}' using column '{field_to_match}'"
        )

        # Default to querying the PK column
        query_column = self.db_id_field
        query_value = id
        if not use_db_id and self.app_id_field != self.db_id_field:
            # Need to query based on the separate app_id_field column
            query_column = self.app_id_field

            # Check if app_id_field actually exists as a column in the model definition
            # If it doesn't exist as a column, querying it will fail.
            field_exists = False
            try:
                hints = get_type_hints(self.entity_type)
                if self.app_id_field in hints:
                    field_exists = True
            except Exception:
                logger.warning(f"Could not get type hints to check if app_id_field '{self.app_id_field}' exists as a separate column.")

            if not field_exists:
                 log_msg = (f"Querying by app_id_field ('{self.app_id_field}') which is not the DB PK ('{self.db_id_field}') "
                            f"and does not appear to be a distinct column in the model {self.entity_type.__name__}. "
                            f"The query will likely fail unless the column exists manually.")
                 logger.error(log_msg)
                 # Raise an error immediately as the query is guaranteed to fail if column doesn't exist
                 raise ValueError(log_msg)

        query = f'SELECT * FROM "{self._table_name}" WHERE "{query_column}" = ? LIMIT 1'
        params = (query_value,)
        execute_kwargs = {}
        if timeout is not None: execute_kwargs["timeout"] = timeout

        try:
            async with self._get_session() as conn:
                async with conn.execute(query, params, **execute_kwargs) as cursor:
                    record_data = await cursor.fetchone()

            if record_data is None:
                logger.warning(
                    f"{self.entity_type.__name__} with {id_type_str} '{id}' not found (Query: {query} | Params: {params})."
                )
                raise ObjectNotFoundException(
                    f"{self.entity_type.__name__} with ID '{id}' not found."
                )

            entity = self._deserialize_record(record_data)
            logger.info(
                f"Retrieved {self.entity_type.__name__} with {id_type_str} '{id}'"
            )
            return entity
        except ObjectNotFoundException:
            raise # Re-raise specific exception
        except Exception as e:
            logger.error(f"Error during get operation (id={id}, use_db_id={use_db_id}): {e}", exc_info=True)
            # Check if it's a DB error before handling
            if isinstance(e, aiosqlite.Error):
                self._handle_db_error(e, f"getting entity ID {id}")
            else:
                raise # Re-raise other exceptions (like ValueError from deserialization)


    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        """Retrieve an entity specifically by its database primary key ID."""
        logger.debug(
            f"Getting {self.entity_type.__name__} by DB ID '{db_id}' "
            f"(using PK column '{self.db_id_field}')"
        )
        # Ensure db_id is converted to string if the PK column is TEXT (common case)
        # If PK is INTEGER, this might need adjustment based on actual PK type.
        # For this implementation assuming TEXT PK based on create_schema.
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
            logger.debug(
                f"Generated application ID '{app_id}' for new {self.entity_type.__name__}"
            )
        elif app_id is None:
            raise ValueError(
                f"Entity of type {self.entity_type.__name__} must have an ID "
                f"field '{self.app_id_field}' set or generate_app_id must be True."
            )

        # The value to insert into the primary key column (db_id_field)
        # is derived from the entity's app_id_field.
        db_id_value = getattr(entity, self.app_id_field)

        logger.debug(
            f"Storing new {self.entity_type.__name__} with {self.app_id_field}='{app_id}' "
            f"(using PK '{self.db_id_field}'='{db_id_value}')"
        )
        try:
            db_data = self._serialize_entity(entity)

            # Ensure the PK column (db_id_field) is in the data to be inserted
            # and holds the correct value derived from app_id.
            db_data[self.db_id_field] = db_id_value

            # If app_id_field is different from db_id_field, decide whether to store
            # app_id_field as a separate column. Our create_schema infers columns,
            # so if app_id_field exists in the model, it should have a column.
            # _serialize_entity should handle preparing app_id_field if it's distinct.

            cols = ", ".join(f'"{k}"' for k in db_data.keys())
            placeholders = ", ".join(["?"] * len(db_data))
            query = f'INSERT INTO "{self._table_name}" ({cols}) VALUES ({placeholders})'
            params = tuple(db_data.values())

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                await conn.execute(query, params, **execute_kwargs)

            logger.info(
                f"Stored new {self.entity_type.__name__} with {self.app_id_field}='{app_id}'. "
                f"(Commit handled externally)"
            )

            # Return the original entity (possibly updated with generated ID)
            return entity if return_value else None

        except aiosqlite.IntegrityError as e:
            # Check if the error message specifically mentions the PK column
            if "UNIQUE constraint failed" in str(e) and f"{self._table_name}.{self.db_id_field}" in str(e):
                logger.warning(
                    f"Failed to store {self.entity_type.__name__}: PK '{db_id_value}' "
                    f"derived from App ID '{app_id}' already exists. Details: {e}"
                )
                raise KeyAlreadyExistsException(
                    f"Entity with ID '{app_id}' already exists (PK constraint on '{self.db_id_field}')."
                ) from e
            else:
                # Handle other integrity errors (e.g., NOT NULL, other UNIQUE constraints)
                logger.error(f"Integrity error storing entity (app_id {app_id}): {e}", exc_info=True)
                self._handle_db_error(e, f"storing entity app_id {app_id}") # Let handle_db_error wrap it
        except Exception as e:
            logger.error(f"Error storing entity (app_id {app_id}): {e}", exc_info=True)
            if isinstance(e, aiosqlite.Error):
                self._handle_db_error(e, f"storing entity app_id {app_id}")
            else:
                raise # Re-raise other exceptions (like ValueError from ID check)


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
            logger.debug(
                f"Generated potential application ID '{app_id}' for upserting {self.entity_type.__name__}"
            )
        elif app_id is None:
            raise ValueError(
                f"Entity for upsert must have ID field '{self.app_id_field}' set or generate_app_id must be True."
            )

        # The value for the primary key column (db_id_field)
        db_id_value = getattr(entity, self.app_id_field)

        logger.debug(
            f"Upserting {self.entity_type.__name__} with {self.app_id_field}='{app_id}' "
            f"(using PK '{self.db_id_field}'='{db_id_value}')"
        )
        try:
            db_data = self._serialize_entity(entity)

            # Ensure the PK column is set correctly
            db_data[self.db_id_field] = db_id_value

            # Prepare for ON CONFLICT: list columns excluding the PK
            update_cols = [
                k for k in db_data.keys() if k != self.db_id_field
            ]

            cols_clause = ", ".join(f'"{k}"' for k in db_data.keys())
            placeholders = ", ".join(["?"] * len(db_data))

            # Handle case where there are no columns to update (only PK)
            if not update_cols:
                # ON CONFLICT DO NOTHING instead of UPDATE SET
                 update_clause_str = "DO NOTHING"
                 query = (
                    f'INSERT INTO "{self._table_name}" ({cols_clause}) VALUES ({placeholders}) '
                    f'ON CONFLICT("{self.db_id_field}") {update_clause_str}'
                 )
            else:
                 update_clause_str = ", ".join(
                    # Use excluded pseudo-table to get values from the proposed insertion
                    f'"{k}" = excluded."{k}"' for k in update_cols
                 )
                 query = (
                     f'INSERT INTO "{self._table_name}" ({cols_clause}) VALUES ({placeholders}) '
                     f'ON CONFLICT("{self.db_id_field}") DO UPDATE SET {update_clause_str}'
                 )


            params = tuple(db_data.values())

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                await conn.execute(query, params, **execute_kwargs)

            logger.info(
                f"Upserted {self.entity_type.__name__} with {self.app_id_field}='{app_id}'. "
                f"(Commit handled externally)"
            )
        except Exception as e:
            logger.error(f"Error upserting entity (app_id {app_id}): {e}", exc_info=True)
            if isinstance(e, aiosqlite.Error):
                 self._handle_db_error(e, f"upserting entity app_id {app_id}")
            else:
                 raise


    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        """
        Update a single entity matching criteria.
        Uses a non-atomic find-then-update approach for SQLite compatibility
        and to support return_value=True.
        """
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

        target_id = None # This will store the PK (db_id_field) of the target row
        try:
            async with self._get_session() as conn:
                # --- Step 1: Find the Primary Key (db_id_field) of the entity to update ---
                find_options = options.copy()
                find_options.limit = 1
                find_options.offset = 0

                find_parts = self._translate_query_options(find_options, include_sorting_pagination=True)
                find_sql = f'SELECT "{self.db_id_field}" FROM "{self._table_name}"'
                if find_parts.get("where"): find_sql += f" WHERE {find_parts['where']}"
                if find_parts.get("order"): find_sql += f" ORDER BY {find_parts['order']}"
                if find_parts.get("limit", -1) > 0 : find_sql += f" LIMIT ?"
                if find_parts.get("offset", 0) > 0: find_sql += f" OFFSET ?"

                find_params = list(find_parts.get("params", []))
                if find_parts.get("limit", -1) > 0 : find_params.append(find_parts["limit"])
                if find_parts.get("offset", 0) > 0 : find_params.append(find_parts["offset"])

                logger.debug(f"Finding target PK for update: SQL='{find_sql}', Params={find_params}")
                async with conn.execute(find_sql, tuple(find_params), **execute_kwargs) as cursor:
                    row = await cursor.fetchone()
                    if not row:
                        raise ObjectNotFoundException(f"No {self.entity_type.__name__} found matching criteria for update: {options.expression!r}")
                    target_id = row[self.db_id_field]

                # --- Step 2: Perform the Update using the retrieved Primary Key ---
                update_parts = self._translate_update(update)
                if not update_parts.get("set"):
                    logger.warning("Update resulted in no SET clauses, skipping DB call.")
                    if return_value: return await self.get(str(target_id), logger, use_db_id=True)
                    else: return None

                update_sql = f'UPDATE "{self._table_name}" SET {update_parts["set"]} WHERE "{self.db_id_field}" = ?'
                update_params = update_parts["params"] + [target_id]

                logger.debug(f"Executing update: SQL='{update_sql}', Params={update_params}")
                cursor = await conn.execute(update_sql, tuple(update_params), **execute_kwargs)

                if cursor.rowcount == 0:
                    logger.warning(f"Update target PK '{target_id}' was not found during update execution (possible race condition).")
                    raise ObjectNotFoundException(f"Entity with PK '{target_id}' disappeared before update.")
                elif cursor.rowcount > 1:
                    logger.error(f"Update_one modified {cursor.rowcount} rows for PK '{target_id}'. This indicates a serious issue.")

            app_id_of_updated = target_id # Assuming PK is app_id for logging
            logger.info(
                f"Updated one {self.entity_type.__name__} with PK '{target_id}'. "
                f"(Commit handled externally)"
            )

            # --- Step 3: Re-fetch if needed ---
            if return_value:
                return await self.get(str(target_id), logger, use_db_id=True)
            else:
                return None
        except ObjectNotFoundException:
             logger.warning(f"ObjectNotFoundException during update_one for filter: {options.expression!r}")
             raise # Propagate ObjectNotFoundException
        except Exception as e:
             logger.error(f"Error updating one entity (target_id: {target_id}, filter: {options.expression!r}): {e}", exc_info=True)
             if isinstance(e, (aiosqlite.Error, NotImplementedError, ValueError, TypeError)): # Handle DB and translation/value errors
                 self._handle_db_error(e, f"updating one entity (target_id: {target_id})")
             else:
                 raise # Re-raise other unexpected errors


    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        """Update multiple entities matching the criteria."""
        logger.debug(
            f"Updating many {self.entity_type.__name__} matching: {options!r} with update: {update!r}"
        )
        if not update:
            logger.warning("update_many called with empty update operations. Returning 0.")
            return 0
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update_many.")

        if options.limit is not None and options.limit > 0:
            logger.warning("QueryOptions 'limit' is ignored for SQLite update_many operation.")
        if options.offset is not None and options.offset > 0:
             logger.warning("QueryOptions 'offset' is ignored for SQLite update_many operation.")
        if options.sort_by:
             logger.warning("QueryOptions 'sort_by' is ignored for SQLite update_many operation.")
        if options.random_order:
             logger.warning("QueryOptions 'random_order' is ignored for SQLite update_many operation.")

        try:
            query_parts = self._translate_query_options(options, include_sorting_pagination=False)
            update_parts = self._translate_update(update)

            if not query_parts or not query_parts.get("where") or query_parts["where"] == "1=1":
                raise ValueError("Cannot update_many without a specific filter expression (safety check).")
            if not update_parts or not update_parts.get("set"):
                 logger.warning("Update resulted in no SET clauses, returning 0.")
                 return 0

            sql = f'UPDATE "{self._table_name}" SET {update_parts["set"]} WHERE {query_parts["where"]}'
            params = update_parts["params"] + query_parts["params"]

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                logger.debug(f"Executing update_many: SQL='{sql}', Params={params}")
                cursor = await conn.execute(sql, tuple(params), **execute_kwargs)
                affected_count = cursor.rowcount

            logger.info(
                f"Updated {affected_count} {self.entity_type.__name__}(s) matching filter: {options.expression!r}. "
                f"(Commit handled externally)"
            )
            return affected_count
        except Exception as e:
            logger.error(f"Error updating many entities (filter: {options.expression!r}): {e}", exc_info=True)
            if isinstance(e, (aiosqlite.Error, NotImplementedError, ValueError, TypeError)):
                self._handle_db_error(e, "updating many entities")
            else:
                raise


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
            raise ValueError("QueryOptions must include an 'expression' for delete_many.")

        if options.limit is not None and options.limit > 0:
            logger.warning("QueryOptions 'limit' is ignored for SQLite delete_many operation.")
        if options.offset is not None and options.offset > 0:
             logger.warning("QueryOptions 'offset' is ignored for SQLite delete_many operation.")
        if options.sort_by:
             logger.warning("QueryOptions 'sort_by' is ignored for SQLite delete_many operation.")
        if options.random_order:
             logger.warning("QueryOptions 'random_order' is ignored for SQLite delete_many operation.")

        try:
            query_parts = self._translate_query_options(options, include_sorting_pagination=False)
            if not query_parts or not query_parts.get("where") or query_parts["where"] == "1=1":
                raise ValueError("Cannot delete_many without a specific filter expression (safety check).")

            sql = f'DELETE FROM "{self._table_name}" WHERE {query_parts["where"]}'
            params = query_parts["params"]

            execute_kwargs = {}
            if timeout is not None: execute_kwargs["timeout"] = timeout

            async with self._get_session() as conn:
                logger.debug(f"Executing delete_many: SQL='{sql}', Params={params}")
                cursor = await conn.execute(sql, tuple(params), **execute_kwargs)
                affected_count = cursor.rowcount

            logger.info(
                f"Deleted {affected_count} {self.entity_type.__name__}(s) matching filter: {options.expression!r}. "
                f"(Commit handled externally)"
            )
            return affected_count
        except Exception as e:
            logger.error(f"Error deleting many entities (filter: {options.expression!r}): {e}", exc_info=True)
            if isinstance(e, (aiosqlite.Error, NotImplementedError, ValueError, TypeError)):
                self._handle_db_error(e, "deleting many entities")
            else:
                raise


    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        """List entities matching the criteria, with sorting and pagination."""
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
            params = list(query_parts.get("params", []))

            if query_parts.get("where") and query_parts["where"] != "1=1":
                sql += f" WHERE {query_parts['where']}"
            if query_parts.get("order"):
                sql += f" ORDER BY {query_parts['order']}"

            limit = query_parts.get("limit", -1)
            if limit >= 0:
                sql += f" LIMIT ?"
                params.append(limit)

            offset = query_parts.get("offset", 0)
            if offset > 0 and limit >= 0:
                sql += f" OFFSET ?"
                params.append(offset)
            elif offset > 0 and limit < 0:
                 logger.warning("Offset specified without limit, ignored by SQLite.")

            logger.debug(f"Executing list query: SQL='{sql}', Params={params}")
            async with self._get_session() as conn:
                async with conn.execute(sql, tuple(params), **execute_kwargs) as cursor:
                    async for record_data in cursor:
                        try:
                            yield self._deserialize_record(record_data)
                        except Exception as deserialization_error:
                            logger.error(
                                f"Failed to deserialize record during list: {deserialization_error}. Record: {dict(record_data)}",
                                exc_info=True,
                            )
                            continue
        except Exception as e:
            logger.error(f"Error listing entities (options: {effective_options!r}): {e}", exc_info=True)
            # Handle DB errors vs Translation/Validation errors
            if isinstance(e, (aiosqlite.Error, NotImplementedError, ValueError, TypeError)):
                self._handle_db_error(e, "listing entities")
            else:
                raise # Re-raise other unexpected errors


    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        """Count entities matching the criteria."""
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
            if query_parts.get("where") and query_parts["where"] != "1=1":
                sql += f" WHERE {query_parts['where']}"
                params = query_parts.get("params", [])

            logger.debug(f"Executing count query: SQL='{sql}', Params={params}")
            async with self._get_session() as conn:
                async with conn.execute(sql, tuple(params), **execute_kwargs) as cursor:
                    result = await cursor.fetchone()
                    count_val = result[0] if result else 0

            logger.info(f"Counted {count_val} {self.entity_type.__name__}(s) matching filter: {effective_options.expression!r}.")
            return int(count_val)
        except Exception as e:
            logger.error(f"Error counting entities (options: {effective_options!r}): {e}", exc_info=True)
            # Handle DB errors vs Translation/Validation errors
            if isinstance(e, (aiosqlite.Error, NotImplementedError, ValueError, TypeError)):
                 self._handle_db_error(e, "counting entities")
            else:
                 raise # Re-raise ValueErrors, TypeErrors etc.


    # --- Helper Method Implementations ---

    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        """Converts the entity object into a dictionary suitable for SQLite storage."""
        data = prepare_for_storage(entity)

        if not isinstance(data, dict):
             raise TypeError(f"prepare_for_storage did not return a dict for entity: {type(entity).__name__}")

        serialized_data = {}
        hints = {}
        try: # Getting hints can fail for complex types
             hints = get_type_hints(self.entity_type)
        except Exception as e:
             self._logger.warning(f"Could not get type hints for {self.entity_type.__name__} during serialization: {e}")


        for key, value in data.items():
            if key.startswith("_"): continue

            target_type = hints.get(key)
            actual_type = target_type
            origin = get_origin(target_type)

            if origin is Union and target_type and _is_none_type(get_args(target_type)[-1]):
                actual_type = get_args(target_type)[0]
                origin = get_origin(actual_type)

            # Determine if type is complex based on origin or concrete type
            is_complex = origin in (list, List, dict, Dict, set, Set, tuple, Tuple, Mapping) or isinstance(value, (dict, list, set, tuple)) or (isinstance(actual_type, type) and (is_dataclass(actual_type) or hasattr(actual_type, 'model_dump')))

            if is_complex:
                if value is not None:
                    try:
                        serialized_data[key] = json.dumps(value)
                    except TypeError as json_error:
                        self._logger.warning(f"Could not JSON serialize field '{key}' (type: {type(value)}). Storing as NULL. Error: {json_error}")
                        serialized_data[key] = None
                else:
                    serialized_data[key] = None
            elif isinstance(value, bool):
                 serialized_data[key] = 1 if value else 0
            elif isinstance(value, (datetime, date)): # Serialize datetime/date to ISO string
                 serialized_data[key] = value.isoformat()
            else:
                 serialized_data[key] = value

        # Ensure the database PK field (db_id_field) is present.
        app_id_value = data.get(self.app_id_field)
        if app_id_value is not None:
            serialized_data[self.db_id_field] = app_id_value
        else:
             self._logger.warning(f"Application ID field '{self.app_id_field}' not found in prepared data for serialization or has value None.")
             db_id_value_fallback = getattr(entity, self.app_id_field, None)
             if db_id_value_fallback is not None:
                 serialized_data[self.db_id_field] = db_id_value_fallback
             else:
                  raise ValueError(f"Cannot determine value for PK column '{self.db_id_field}' from entity.")

        entity_field_names = set(hints.keys())
        if self.app_id_field != self.db_id_field and self.app_id_field not in entity_field_names:
             serialized_data.pop(self.app_id_field, None)

        return serialized_data

    def _deserialize_record(self, record_data: aiosqlite.Row) -> T:
        """Converts an aiosqlite.Row (dict-like) into an entity object T."""
        if record_data is None: raise ValueError("Cannot deserialize None record data.")

        entity_dict = dict(record_data)
        processed_dict = {}
        hints = {}
        try: # Getting hints can fail
            hints = get_type_hints(self.entity_type)
        except Exception as e:
             self._logger.warning(f"Could not get type hints for {self.entity_type.__name__} during deserialization: {e}")

        for key, value in entity_dict.items():
            target_type = hints.get(key)
            if target_type is None:
                processed_dict[key] = value # Include unknown fields
                continue

            origin = get_origin(target_type)
            args = get_args(target_type)
            is_optional = origin is Union and _is_none_type(args[-1])
            actual_type = args[0] if is_optional else target_type
            actual_origin = get_origin(actual_type)

            # Handle NULL from DB first
            if value is None:
                processed_dict[key] = None
                if not is_optional:
                    self._logger.warning(f"Field '{key}' is not Optional in model but is NULL in DB record.")
                continue

            # Check if target type is complex
            is_complex_target = actual_origin in (list, List, dict, Dict, set, Set, tuple, Tuple, Mapping) or (isinstance(actual_type, type) and (issubclass(actual_type, (list, dict, set, tuple)) or is_dataclass(actual_type) or hasattr(actual_type, 'model_dump')))

            # If DB value is string and target type is complex, attempt JSON decode
            if isinstance(value, str) and is_complex_target:
                try:
                    processed_dict[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    self._logger.warning(f"Failed to JSON decode TEXT field '{key}' for expected type {target_type}. Value: '{value[:100]}...'. Falling back to raw string value.")
                    processed_dict[key] = value
            # Handle boolean conversion
            elif isinstance(value, int) and actual_type is bool:
                processed_dict[key] = bool(value)
            # Handle datetime/date conversion from ISO string
            elif isinstance(value, str) and actual_type in (datetime, date):
                 try:
                      if actual_type is datetime:
                           # Handle potential 'Z' for UTC which fromisoformat doesn't like directly before 3.11
                           if value.endswith('Z'):
                                value = value[:-1] + '+00:00'
                           processed_dict[key] = datetime.fromisoformat(value)
                      elif actual_type is date:
                           processed_dict[key] = date.fromisoformat(value)
                 except ValueError as dt_error:
                      self._logger.warning(f"Failed to parse TEXT field '{key}' as {actual_type.__name__}. Value: '{value}'. Error: {dt_error}. Assigning raw string value.")
                      processed_dict[key] = value # Fallback to raw string
            else:
                processed_dict[key] = value

        # --- ID Field Mapping ---
        db_id_value = entity_dict.get(self.db_id_field)
        if db_id_value is not None:
            processed_dict[self.app_id_field] = str(db_id_value)
            entity_field_names = set(hints.keys())
            if self.db_id_field != self.app_id_field and self.db_id_field not in entity_field_names:
                 processed_dict.pop(self.db_id_field, None)
        else:
             self._logger.error(f"Primary key column '{self.db_id_field}' has NULL value in fetched record: {entity_dict}")
             if self.app_id_field in hints:
                  processed_dict[self.app_id_field] = None

        try:
            return self.entity_type(**processed_dict)
        except Exception as e:
            self._logger.error(f"Failed to instantiate {self.entity_type.__name__} from DB data: {e}. Data: {processed_dict!r}", exc_info=True)
            raise ValueError(f"Failed to create {self.entity_type.__name__} instance from record") from e


    def _translate_query_options(
        self, options: QueryOptions, include_sorting_pagination: bool = False
    ) -> Dict[str, Any]:
        """Translates QueryOptions into SQLite WHERE, ORDER BY, LIMIT, OFFSET parts."""
        sql_parts: Dict[str, Any] = {"where": "1=1", "params": []} # Default to match all if no expression

        if options.expression:
            try:
                where_clause, params = self._translate_expression_recursive(
                    options.expression
                )
                if where_clause and where_clause != "1=1":
                    sql_parts["where"] = where_clause
                    sql_parts["params"] = params
            except Exception as e:
                self._logger.error(f"Failed to translate query expression: {options.expression!r}", exc_info=True)
                raise # Re-raise translation errors

        if include_sorting_pagination:
            if options.random_order:
                sql_parts["order"] = "RANDOM()"
            elif options.sort_by:
                direction = "DESC" if options.sort_desc else "ASC"
                # Quote sort column, handle dot notation (sorting by nested JSON might not work well)
                if '.' in options.sort_by:
                     self._logger.warning(f"Sorting by nested path '{options.sort_by}' might not work as expected in SQLite without specific JSON functions.")
                sql_parts["order"] = f'"{options.sort_by}" {direction}' # Basic quoting
            else:
                sql_parts["order"] = None

            sql_parts["limit"] = options.limit if (options.limit is not None and options.limit >= 0) else -1
            sql_parts["offset"] = options.offset if (options.offset is not None and options.offset > 0) else 0

        return sql_parts


    def _translate_expression_recursive(
        self, expression: QueryExpression
    ) -> Tuple[str, List[Any]]:
        """Recursively translates QueryExpression nodes into SQLite WHERE clause + params."""
        if isinstance(expression, QueryFilter):
            field = expression.field_path
            op = expression.operator # Can be QueryOperator enum or string
            val = expression.value

            # Defensive check for operator type
            if not isinstance(op, QueryOperator):
                valid_str_op = False
                for enum_member in QueryOperator:
                    if op == enum_member.value:
                        op = enum_member
                        valid_str_op = True
                        break
                if not valid_str_op:
                    raise ValueError(f"Unsupported query operator for SQLite translation: {op}")

            sql_op_map = {
                QueryOperator.EQ: "=", QueryOperator.NE: "!=",
                QueryOperator.GT: ">", QueryOperator.GTE: ">=",
                QueryOperator.LT: "<", QueryOperator.LTE: "<=",
                QueryOperator.IN: "IN", QueryOperator.NIN: "NOT IN",
                QueryOperator.LIKE: "LIKE", QueryOperator.STARTSWITH: "LIKE",
                QueryOperator.ENDSWITH: "LIKE", QueryOperator.CONTAINS: "LIKE",
                QueryOperator.EXISTS: "IS NOT NULL",
            }
            sql_op = sql_op_map.get(op)
            params: List[Any] = []

            if not sql_op:
                 raise ValueError(f"Unsupported query operator for SQLite translation: {op}")

            base_column_name = field.split('.', 1)[0]
            quoted_base_column = f'"{base_column_name}"'
            json_path_suffix = None
            is_json_column = False

            if '.' in field:
                json_path_suffix = f"$.{'.'.join(field.split('.')[1:])}"
                try:
                    hints = get_type_hints(self.entity_type)
                    base_type_hint = hints.get(base_column_name)
                    if get_origin(base_type_hint) is Union and _is_none_type(get_args(base_type_hint)[-1]):
                         base_type_hint = get_args(base_type_hint)[0]

                    base_origin = get_origin(base_type_hint)
                    is_base_complex = base_origin in (dict, list, Mapping, List) or \
                                      (isinstance(base_type_hint, type) and \
                                       (issubclass(base_type_hint, (list, dict, set, tuple)) or \
                                        is_dataclass(base_type_hint) or hasattr(base_type_hint, 'model_dump')))

                    if is_base_complex:
                         is_json_column = True
                except Exception: pass

                if not is_json_column:
                     self._logger.error(f"Query path '{field}' uses dot notation, but the base column '{base_column_name}' does not appear to be a JSON-storing type (Dict/List/Model/Dataclass) in the model definition.")
                     raise TypeError(f"Cannot query nested path '{field}' on non-JSON-like base column '{base_column_name}'.")

                quoted_field = f"json_extract({quoted_base_column}, '{json_path_suffix}')"
            else:
                quoted_field = f'"{field}"'

            # --- Operator specific logic ---
            if op == QueryOperator.EXISTS:
                sql_op_str = "IS NOT NULL" if val else "IS NULL"
                if is_json_column:
                     check_sql = f"json_type({quoted_base_column}, '{json_path_suffix}')"
                     sql_op_str = "IS NOT 'null'" if val else "IS 'null'"
                     return f"(json_valid({quoted_base_column}) AND {check_sql} {sql_op_str})", []
                else:
                     return f"{quoted_base_column} {sql_op_str}", []


            elif op == QueryOperator.IN or op == QueryOperator.NIN:
                if is_json_column:
                    raise NotImplementedError(f"IN/NIN operators on nested JSON paths ('{field}') are not supported by the SQLite backend.")
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"Value for {op} operator must be a list or tuple, got {type(val)}")
                if not val:
                    return ("0=1", []) if op == QueryOperator.IN else ("1=1", [])
                placeholders = ", ".join(["?"] * len(val))
                prepared_val = [1 if isinstance(item, bool) and item else 0 if isinstance(item, bool) else item for item in val]
                params.extend(prepared_val)
                return f"{quoted_field} {sql_op} ({placeholders})", params

            elif op == QueryOperator.STARTSWITH:
                if is_json_column:
                    sql_fragment = f"{quoted_field} LIKE ?"
                    if not isinstance(val, str): raise ValueError("Value for STARTSWITH must be a string.")
                    params.append(f"{val}%")
                    return f"(json_valid({quoted_base_column}) AND {sql_fragment})", params
                else:
                    if not isinstance(val, str): raise ValueError("Value for STARTSWITH must be a string.")
                    params.append(f"{val}%")
                    return f"{quoted_field} {sql_op} ?", params

            elif op == QueryOperator.ENDSWITH:
                 if is_json_column:
                    sql_fragment = f"{quoted_field} LIKE ?"
                    if not isinstance(val, str): raise ValueError("Value for ENDSWITH must be a string.")
                    params.append(f"%{val}")
                    return f"(json_valid({quoted_base_column}) AND {sql_fragment})", params
                 else:
                    if not isinstance(val, str): raise ValueError("Value for ENDSWITH must be a string.")
                    params.append(f"%{val}")
                    return f"{quoted_field} {sql_op} ?", params

            elif op == QueryOperator.CONTAINS:
                is_list_like = False
                if self._validator:
                    try:
                        field_type = self._validator.get_field_type(field)
                        is_list_like, _ = self._validator.get_list_item_type(field)
                    except InvalidPathError: pass

                if is_list_like:
                    effective_json_path = json_path_suffix if json_path_suffix else '$'
                    prepared_val = prepare_for_storage(val)

                    if isinstance(prepared_val, (int, float, str)): param_val = prepared_val
                    elif isinstance(prepared_val, bool): param_val = 1 if prepared_val else 0
                    else:
                         try: param_val = json.dumps(prepared_val)
                         except TypeError:
                              self._logger.error(f"Cannot JSON encode value for CONTAINS check on field '{field}': {prepared_val}")
                              return "0=1", []

                    sql_fragment = f"EXISTS (SELECT 1 FROM json_each({quoted_base_column}, '{effective_json_path}') WHERE value = ?)"
                    params.append(param_val)
                    return f"(json_valid({quoted_base_column}) AND {sql_fragment})", params
                else:
                    if is_json_column:
                        sql_fragment = f"{quoted_field} LIKE ?"
                        if not isinstance(val, str): self._logger.warning(f"CONTAINS value for JSON path '{field}' is not a string ({type(val)}), LIKE comparison might behave unexpectedly.")
                        params.append(f"%{val}%")
                        return f"(json_valid({quoted_base_column}) AND {sql_fragment})", params
                    else:
                        self._logger.warning(f"CONTAINS operator for non-list SQLite field '{field}' translated to LIKE '%value%'.")
                        if not isinstance(val, str): self._logger.warning(f"CONTAINS value for field '{field}' is not a string ({type(val)}), LIKE comparison might behave unexpectedly.")
                        params.append(f"%{val}%")
                        return f"{quoted_field} {sql_op} ?", params

            else: # EQ, NE, GT, LT, GTE, LTE, LIKE
                param_val: Any
                if isinstance(val, bool): param_val = 1 if val else 0
                elif isinstance(val, (datetime, date)): param_val = val.isoformat()
                else: param_val = val
                params.append(param_val)

                if is_json_column:
                     self._logger.debug(f"Applying operator {op} to JSON path '{field}' using json_extract.")
                     return f"(json_valid({quoted_base_column}) AND {quoted_field} {sql_op} ?)", params
                else:
                     return f"{quoted_field} {sql_op} ?", params

        elif isinstance(expression, QueryLogical):
            if not expression.conditions:
                return ("1=1", [])

            sql_fragments = []
            all_params: List[Any] = []
            for cond in expression.conditions:
                fragment, params = self._translate_expression_recursive(cond)
                if fragment and fragment != "1=1":
                    sql_fragments.append(f"({fragment})")
                    all_params.extend(params)

            if not sql_fragments:
                 return ("1=1", [])

            logical_op_sql = f" {expression.operator.upper()} "
            return logical_op_sql.join(sql_fragments), all_params
        else:
            raise TypeError(f"Unknown QueryExpression type encountered during translation: {type(expression)}")


    def _translate_update(self, update: Update) -> Dict[str, Any]:
        """Translates Update object into SQLite SET clause and parameters."""
        set_clauses: List[str] = []
        params: List[Any] = []

        operations = update.build()
        if not operations:
            return {"set": "", "params": []}

        for op in operations:
            field = op.field_path
            is_nested = '.' in field
            base_column_name = field.split('.', 1)[0] if is_nested else field
            quoted_base_column = f'"{base_column_name}"'

            if isinstance(op, SetOperation):
                value = op.value # Python value
                # Prepare the value for binding: JSON strings for complex types, direct for primitives
                param_value: Any
                if value is None: param_value = None
                elif isinstance(value, bool): param_value = 1 if value else 0
                elif isinstance(value, (datetime, date)): param_value = value.isoformat()
                elif isinstance(value, (int, float, str)): param_value = value
                else: # dict, list, set, tuple etc. -> needs JSON string
                     try:
                          prepared_nested_value = prepare_for_storage(value)
                          param_value = json.dumps(prepared_nested_value)
                     except TypeError as e:
                          raise ValueError(f"Cannot JSON serialize complex value for update on '{field}': {value}") from e

                if is_nested:
                    nested_path = field.split('.', 1)[1]
                    json_path = f"$.{nested_path}"
                    value_placeholder = "json(?)"
                    if isinstance(value, (int, float, bool, str, bytes, type(None), datetime, date)):
                         value_placeholder = "?"
                    set_clauses.append(f"{quoted_base_column} = json_set(CASE WHEN json_valid(COALESCE({quoted_base_column}, '{{}}')) THEN COALESCE({quoted_base_column}, '{{}}') ELSE '{{}}' END, ?, {value_placeholder})")
                    params.append(json_path)
                    params.append(param_value)
                else: # Top-level Set
                    quoted_field = f'"{field}"'
                    set_clauses.append(f"{quoted_field} = ?")
                    params.append(param_value)

            elif isinstance(op, UnsetOperation):
                 if is_nested:
                      nested_path = field.split('.', 1)[1]
                      json_path = f"$.{nested_path}"
                      set_clauses.append(f"{quoted_base_column} = CASE WHEN json_valid({quoted_base_column}) THEN json_remove({quoted_base_column}, ?) ELSE {quoted_base_column} END")
                      params.append(json_path)
                 else: # Top-level unset
                    quoted_field = f'"{field}"'
                    set_clauses.append(f"{quoted_field} = NULL")

            elif isinstance(op, (IncrementOperation, MultiplyOperation, MinOperation, MaxOperation)):
                 if is_nested:
                     raise NotImplementedError(f"{type(op).__name__} on nested JSON paths ('{field}') is not supported by the SQLite backend.")
                 quoted_field = f'"{field}"'
                 if isinstance(op, IncrementOperation):
                    set_clauses.append(f"{quoted_field} = COALESCE({quoted_field}, 0) + ?")
                    params.append(op.amount)
                 elif isinstance(op, MultiplyOperation):
                    set_clauses.append(f"{quoted_field} = COALESCE({quoted_field}, 0) * ?")
                    params.append(op.factor)
                 elif isinstance(op, MinOperation):
                    set_clauses.append(f"{quoted_field} = CASE WHEN COALESCE({quoted_field}, ?) > ? THEN ? ELSE COALESCE({quoted_field}, ?) END")
                    null_placeholder_val = op.value
                    params.extend([null_placeholder_val, op.value, op.value, null_placeholder_val])
                 elif isinstance(op, MaxOperation):
                    set_clauses.append(f"{quoted_field} = CASE WHEN COALESCE({quoted_field}, ?) < ? THEN ? ELSE COALESCE({quoted_field}, ?) END")
                    null_placeholder_val = op.value
                    params.extend([null_placeholder_val, op.value, op.value, null_placeholder_val])

            elif isinstance(op, PushOperation):
                # --- Simplified Push Logic ---
                # NOTE: Assumes target is NULL or valid JSON array to avoid 'incomplete input' errors.
                if is_nested:
                    nested_path = field.split('.', 1)[1]
                    json_path_base = f"$.{nested_path}"
                else:
                    json_path_base = '$'
                quoted_field = quoted_base_column
                if not op.items: continue
                sql_item_placeholders = []
                current_item_params = []
                for item in op.items:
                    param_value: Any; is_complex_item = False
                    if item is None: param_value = None
                    elif isinstance(item, bool): param_value = 1 if item else 0
                    elif isinstance(item, (datetime, date)): param_value = item.isoformat()
                    elif isinstance(item, (int, float, str)): param_value = item
                    else:
                         try:
                              prepared_item = prepare_for_storage(item); param_value = json.dumps(prepared_item); is_complex_item = True
                         except TypeError as e: raise ValueError(f"Cannot JSON serialize item for push to '{field}': {item}") from e
                    append_path_str = f"{json_path_base}[#]"
                    value_placeholder = "json(?)" if is_complex_item else "?"
                    sql_item_placeholders.append("?"); sql_item_placeholders.append(value_placeholder)
                    current_item_params.append(append_path_str); current_item_params.append(param_value)
                first_arg_expr = f"COALESCE({quoted_field}, json('[]'))"
                insert_args = ", ".join(sql_item_placeholders)
                set_clause = f"{quoted_field} = json_insert({first_arg_expr}, {insert_args})"
                operation_params = current_item_params
                set_clauses.append(set_clause)
                params.extend(operation_params)

            elif isinstance(op, PopOperation):
                # --- Simplified Pop Logic ---
                # NOTE: Bypasses json_valid/json_type/json_array_length checks entirely
                # to avoid 'incomplete input' errors potentially caused by complex CASE
                # with multiple JSON path parameters.
                # Assumes the target column/path contains a valid JSON array or is NULL.
                # May fail at runtime if the target is not an array or is empty.

                if is_nested:
                    nested_path = field.split('.', 1)[1]
                    json_path_base = f"$.{nested_path}"
                else:
                    json_path_base = '$'

                quoted_field = quoted_base_column

                if op.position == 1: # Pop last
                    json_path_index_str = f"{json_path_base}[#-1]"
                elif op.position == -1: # Pop first
                    json_path_index_str = f"{json_path_base}[0]"
                else:
                    raise ValueError(f"Invalid position for PopOperation: {op.position}. Must be 1 (last) or -1 (first).")

                # Directly use json_remove, relying on COALESCE for NULL handling.
                # This avoids the complex CASE statement.
                set_clause = f"{quoted_field} = json_remove(COALESCE({quoted_field}, json('[]')), ?)"

                # Parameters needed: only the path for json_remove
                current_params = [json_path_index_str]

                set_clauses.append(set_clause)
                params.extend(current_params)


            elif isinstance(op, PullOperation):
                # Handles both nested and top-level pulls.
                # Uses simplified logic for top-level to avoid 'incomplete input'.

                if isinstance(op.value_or_condition, dict):
                     raise NotImplementedError("Conditional PullOperation with dict criteria not supported by SQLite backend.")
                value_to_pull = op.value_or_condition
                param_value: Any
                if value_to_pull is None: param_value = None
                elif isinstance(value_to_pull, bool): param_value = 1 if value_to_pull else 0
                elif isinstance(value_to_pull, (datetime, date)): param_value = value_to_pull.isoformat()
                elif isinstance(value_to_pull, (int, float, str)): param_value = value_to_pull
                else:
                     try:
                          prepared_val = prepare_for_storage(value_to_pull); param_value = json.dumps(prepared_val)
                     except TypeError as e: raise ValueError(f"Cannot JSON serialize value for pull from '{field}': {value_to_pull}") from e
                where_clause_subquery = "value IS NOT NULL" if param_value is None else "value != ?"

                if is_nested:
                    # --- Nested Pull Logic (using json_set + CASE) ---
                    nested_path_parts = field.split('.')[1:]
                    json_path_to_array = f"$.{'.'.join(nested_path_parts)}"
                    filtered_array_subquery = f"""( SELECT json_group_array(value) FROM json_each({quoted_base_column}, ?) WHERE {where_clause_subquery} )"""
                    set_clause = f"""{quoted_base_column} = CASE WHEN json_valid({quoted_base_column}) AND json_type({quoted_base_column}, ?) = 'array' THEN json_set({quoted_base_column}, ?, {filtered_array_subquery}) ELSE {quoted_base_column} END"""
                    current_params = []; current_params.append(json_path_to_array); current_params.append(json_path_to_array); current_params.append(json_path_to_array)
                    if param_value is not None: current_params.append(param_value)
                    set_clauses.append(set_clause.strip().replace('\n', ' ')); params.extend(current_params)
                else:
                    # --- Simplified Top-Level Pull Logic (No CASE) ---
                    # NOTE: Bypasses json_valid/json_type checks to avoid 'incomplete input' errors. Assumes valid array or NULL.
                    json_path_to_array = '$'
                    quoted_field = quoted_base_column
                    filtered_array_subquery = f"""( SELECT json_group_array(value) FROM json_each(COALESCE({quoted_field}, json('[]')), ?) WHERE {where_clause_subquery} )"""
                    set_clause = f"{quoted_field} = {filtered_array_subquery}"
                    current_params = []; current_params.append(json_path_to_array)
                    if param_value is not None: current_params.append(param_value)
                    set_clauses.append(set_clause.strip().replace('\n', ' ')); params.extend(current_params)

            else:
                raise TypeError(f"Unsupported UpdateOperation type encountered during translation: {type(op)}")

        set_clause_str = ", ".join(set_clauses)
        return {"set": set_clause_str, "params": params}
    
     
    def _handle_db_error(self, error: Exception, context: str = "") -> None:
        """Maps specific database errors to repository exceptions or raises/re-raises."""
        log_message = f"Error during {context}: {error}"
        # Log DB errors with traceback, others maybe less verbosely
        if isinstance(error, aiosqlite.Error):
            self._logger.error(log_message, exc_info=True)
        else:
            self._logger.error(log_message) # Log application errors without traceback by default

        # Specific DB error mapping
        if isinstance(error, aiosqlite.IntegrityError):
            # Let specific handlers in store/upsert deal with KeyAlreadyExistsException based on message
            # If it reaches here, raise a general ValueError for integrity issues
            raise ValueError(
                f"Database integrity constraint violated during {context}. Detail: {error}"
            ) from error
        elif isinstance(error, aiosqlite.OperationalError):
            # Check for specific operational errors like "no such function: json_each"
            no_such_func_match = re.search(r"no such function: (json_\w+)", str(error))
            if no_such_func_match:
                 func_name = no_such_func_match.group(1)
                 msg = f"SQLite JSON1 extension function '{func_name}' not found. Ensure SQLite version >= 3.9 with JSON1 enabled. Context: {context}. Error: {error}"
                 self._logger.error(msg)
                 raise NotImplementedError(msg) from error
            # Check for malformed JSON error during json_set/json_remove etc.
            if "malformed JSON" in str(error):
                msg = f"Malformed JSON encountered during {context}. Check if the column contains valid JSON or if the update path/value is correct. Error: {error}"
                self._logger.error(msg)
                # Raise ValueError as it might indicate bad input data or invalid existing data
                raise ValueError(msg) from error
            # Check for incomplete input error (often syntax errors in generated SQL)
            if "incomplete input" in str(error):
                 msg = f"Incomplete SQL input during {context}. This likely indicates a syntax error in the generated SQL query. Error: {error}"
                 self._logger.error(f"{msg} (Check generated SQL in logs if DEBUG level enabled)")
                 raise ValueError(msg) from error

            raise RuntimeError(
                f"Database operational error during {context}: {error}"
            ) from error
        elif isinstance(error, aiosqlite.ProgrammingError):
             # Check for the specific parameter binding error
             if "Error binding parameter" in str(error) and "type 'dict' is not supported" in str(error):
                  msg = f"Failed to bind Python dictionary as parameter during {context}, likely requires JSON serialization using json() SQL function. Error: {error}"
                  self._logger.error(msg)
                  # Raise a TypeError or ValueError as it's an application/translation issue
                  raise TypeError(msg) from error
             raise ValueError(
                 f"Database programming error during {context} (likely bad SQL or params): {error}"
             ) from error
        # If a NotImplementedError or ValueError/TypeError from our own logic reaches here, re-raise it
        elif isinstance(error, (NotImplementedError, ValueError, TypeError)):
             raise error
        else:
            # Wrap other unexpected errors as RuntimeError
            raise RuntimeError(
                f"An unexpected error occurred during {context}"
            ) from error