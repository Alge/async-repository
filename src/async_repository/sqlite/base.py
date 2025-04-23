# src/async_repository/sqlite/base.py
import json
import re
import logging
from logging import LoggerAdapter
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
    get_args,
    get_origin,
    Union,
)
import sqlite3
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone

import aiosqlite
from pydantic import AnyUrl  # Used for type hint checking

# Adjust relative imports based on your project structure
from ..base.interfaces import Repository
from ..base.query import QueryOptions
from ..base.exceptions import ObjectNotFoundException, KeyAlreadyExistsException
from ..base.utils import prepare_for_storage
from ..base.update import Update

T = TypeVar("T")
logger = logging.getLogger(__name__)  # Module-level logger


# --- Helper functions for SQLite REGEXP support ---
def _sqlite_regexp(pattern: str, text: str) -> bool:
    """Python function to implement SQLite REGEXP operator using re.search."""
    if text is None:
        return False
    try:
        # Make sure pattern is treated as a raw string if it contains backslashes
        return re.search(pattern, str(text)) is not None
    except re.error:
        logger.warning(f"Invalid REGEXP pattern evaluated: '{pattern}'")
        return False


async def _register_sqlite_extensions(connection: aiosqlite.Connection):
    """Function to register custom functions and pragmas for a new connection."""
    # Register REGEXP function
    await connection.create_function("REGEXP", 2, _sqlite_regexp, deterministic=True)
    # Enable Write-Ahead Logging for better concurrency with aiosqlite
    try:
        await connection.execute("PRAGMA journal_mode=WAL;")
    except Exception as e:
        logger.debug(f"Could not set PRAGMA journal_mode=WAL: {e}")


# --- End Helper functions ---


def quote_identifier(identifier: str) -> str:
    """Quote an identifier for SQLite (SQLite uses double quotes for identifiers)."""
    safe_identifier = identifier.replace('"', '""')
    return f'"{safe_identifier}"'


class SQLiteRepository(Repository[T], Generic[T]):
    """
    Base SQLite repository implementation using aiosqlite.
    Includes REGEXP support via a user-defined function registered post-connect.
    Handles basic type conversions (JSON, bool, datetime).
    """

    def __init__(
        self,
        db_path: str,
        table_name: str,
        entity_cls: Type[T],
        app_id_field: str = "id",
        db_id_field: str = "db_id",  # SQLite often uses INTEGER PK
    ):
        self._db_path = db_path
        self._table_name = table_name
        self._entity_cls = entity_cls
        self._app_id_field = app_id_field
        self._db_id_field = db_id_field

    # --- Properties ---
    @property
    def entity_type(self) -> Type[T]:
        return self._entity_cls

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    # --- Connection Helper ---
    async def _get_connection(self) -> aiosqlite.Connection:
        """Connects to the SQLite DB and registers necessary functions/pragmas."""
        try:
            conn = await aiosqlite.connect(self._db_path)
            conn.row_factory = aiosqlite.Row  # Set row factory for dict-like access
            await _register_sqlite_extensions(conn)  # Apply REGEXP func and pragmas
            return conn
        except Exception as e:
            logger.error(
                f"Failed to connect or initialize SQLite connection to {self._db_path}: {e}",
                exc_info=True,
            )
            raise

    # --- Core Methods ---
    async def get(
        self,
        id_value: Any,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        field = self._db_id_field if use_db_id else self._app_id_field
        id_type = "database (PK)" if use_db_id else "application"
        logger_adapter.debug(
            f"Getting {self.entity_type.__name__} with {id_type} ID: {id_value}"
        )
        query = (
            f"SELECT * FROM {quote_identifier(self._table_name)} "
            f"WHERE {quote_identifier(field)} = ?"
        )
        params = [id_value]
        row_dict = None
        conn = await self._get_connection()
        try:
            async with conn.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row:
                    row_dict = dict(row)
        except Exception as e:
            logger_adapter.error(f"Error during get operation: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

        if not row_dict:
            raise ObjectNotFoundException(
                f"{self.entity_type.__name__} with {id_type} ID {id_value} not found"
            )

        return self._row_to_entity(row_dict)

    async def get_by_db_id(
        self, db_id: Any, logger_adapter: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        return await self.get(db_id, logger_adapter, timeout, use_db_id=True)

    async def store(
        self,
        entity: T,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
        return_value: bool = False,
    ) -> Optional[T]:
        self.validate_entity(entity)
        entity_dict_raw = self._entity_to_initial_dict(entity)

        app_id_val = entity_dict_raw.get(self._app_id_field)
        if generate_app_id and app_id_val is None:
            generated_id = self.id_generator()
            entity_dict_raw[self._app_id_field] = generated_id
            app_id_val = generated_id

        entity_dict = self._prepare_dict_for_storage(entity_dict_raw)

        if self._db_id_field in entity_dict and entity_dict[self._db_id_field] is None:
            del entity_dict[self._db_id_field]

        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = ["?" for _ in fields]

        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} "
            f"({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)})"
        )

        row_to_return = None
        conn = await self._get_connection()
        try:
            last_id = None
            try:
                async with conn.execute(query, values) as cursor:
                    last_id = cursor.lastrowid
                await conn.commit()

                if return_value:
                    fetch_by_pk = last_id is not None
                    fetch_value = last_id if fetch_by_pk else app_id_val

                    if fetch_value is None:
                        logger_adapter.error(
                            "Cannot determine ID to fetch stored entity after insert."
                        )
                    else:
                        row_to_return = await self._fetch_row_by_id(
                            conn, fetch_value, use_db_id=fetch_by_pk
                        )

            except sqlite3.IntegrityError as e:
                await conn.rollback()
                if "UNIQUE constraint failed" in str(e):
                    failed_field = "unique constraint"
                    if f".{self._app_id_field}" in str(e):
                        failed_field = self._app_id_field
                    elif f".{self._db_id_field}" in str(e):
                        failed_field = self._db_id_field
                    failed_value = entity_dict_raw.get(
                        self._app_id_field
                    ) or entity_dict_raw.get(self._db_id_field)
                    raise KeyAlreadyExistsException(
                        f"{self.entity_type.__name__} with {failed_field} '{failed_value}' already exists."
                    ) from e
                else:
                    raise
            except Exception as e:
                await conn.rollback()
                logger_adapter.error(
                    f"Error during store operation: {e}", exc_info=True
                )
                raise

        except Exception as e:  # Catch errors during connect
            logger_adapter.error(
                f"Error connecting to SQLite DB for store: {e}", exc_info=True
            )
            raise
        finally:
            if conn:
                await conn.close()

        return (
            self._row_to_entity(row_to_return)
            if return_value and row_to_return
            else None
        )

    async def upsert(
        self,
        entity: T,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
    ) -> None:
        self.validate_entity(entity)
        entity_dict_raw = self._entity_to_initial_dict(entity)

        app_id_val = entity_dict_raw.get(self._app_id_field)
        if generate_app_id and app_id_val is None:
            entity_dict_raw[self._app_id_field] = self.id_generator()

        entity_dict = self._prepare_dict_for_storage(entity_dict_raw)

        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = ["?" for _ in fields]

        logger_adapter.debug(
            f"Upserting {self.entity_type.__name__}: {entity_dict_raw.get(self._app_id_field)}"
        )

        set_clauses = [
            f"{quote_identifier(f)} = excluded.{quote_identifier(f)}"
            for f in fields
            if f != self._db_id_field
        ]

        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} ({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT({quote_identifier(self._db_id_field)}) DO UPDATE SET {', '.join(set_clauses)}"
        )

        conn = await self._get_connection()
        try:
            await conn.execute(query, values)
            await conn.commit()
        except Exception as e:
            await conn.rollback()
            logger_adapter.error(f"Error during upsert: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        logger_adapter.debug(f"Updating one {self.entity_type.__name__}...")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        where_clause, params, _ = self._transform_expression(options.expression, 1)
        where_clause = re.sub(r"\$\d+", "?", where_clause)

        updated_row_dict = None
        entity_pk_value = None

        conn = await self._get_connection()
        try:
            id_field_to_select = self._db_id_field
            find_query = (
                f"SELECT {quote_identifier(id_field_to_select)} FROM {quote_identifier(self._table_name)} "
                f"WHERE {where_clause} LIMIT 1"
            )
            async with conn.execute(find_query, params) as cursor:
                row_to_update = await cursor.fetchone()

            if not row_to_update:
                raise ObjectNotFoundException(
                    f"{self.entity_type.__name__} not found with criteria {options.expression}"
                )
            entity_pk_value = row_to_update[id_field_to_select]

            update_payload = prepare_for_storage(update.build())
            set_clause, set_params = self._build_set_clause(update_payload, 1)
            set_clause = re.sub(r"\$\d+", "?", set_clause)

            if not set_clause:
                logger_adapter.warning("Update resulted in empty SET clause.")
                if return_value:
                    updated_row_dict = await self._fetch_row_by_id(
                        conn, entity_pk_value, use_db_id=True
                    )
            else:
                update_query = (
                    f"UPDATE {quote_identifier(self._table_name)} SET {set_clause} "
                    f"WHERE {quote_identifier(id_field_to_select)} = ?"
                )
                final_params = set_params + [entity_pk_value]

                async with conn.execute(update_query, final_params) as cursor:
                    if cursor.rowcount == 0:
                        logger_adapter.warning(
                            f"Update affected 0 rows for PK {entity_pk_value}."
                        )
                await conn.commit()

                if return_value:
                    updated_row_dict = await self._fetch_row_by_id(
                        conn, entity_pk_value, use_db_id=True
                    )
                    if not updated_row_dict:
                        raise ObjectNotFoundException(
                            f"Entity PK {entity_pk_value} missing after update."
                        )

        except ObjectNotFoundException:
            raise
        except Exception as e:
            await conn.rollback()
            logger_adapter.error(f"Error during update_one: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

        return (
            self._row_to_entity(updated_row_dict)
            if return_value and updated_row_dict
            else None
        )

    async def _fetch_row_by_id(
        self, conn: aiosqlite.Connection, id_value: Any, use_db_id: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Internal helper to fetch a row by PK or App ID."""
        id_field = self._db_id_field if use_db_id else self._app_id_field
        get_query = f"SELECT * FROM {quote_identifier(self._table_name)} WHERE {quote_identifier(id_field)} = ?"
        async with conn.execute(get_query, [id_value]) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger_adapter.debug(f"Updating many {self.entity_type.__name__} ...")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        id_field_to_select = self._db_id_field
        final_where_clause = ""
        final_where_params = []
        ids_to_update = []

        conn = await self._get_connection()
        modified_count = 0
        try:
            if (
                options.limit > 0
                or options.offset > 0
                or options.sort_by
                or options.random_order
            ):
                where_clause, params, _ = self._transform_expression(
                    options.expression, 1
                )
                where_clause = re.sub(r"\$\d+", "?", where_clause)
                order_clause = ""
                limit_clause = ""
                offset_clause = ""
                if options.random_order:
                    order_clause = " ORDER BY RANDOM()"
                elif options.sort_by:
                    sort_field = options.sort_by
                    if sort_field == "id" and self.app_id_field != "id":
                        sort_field = self.app_id_field
                    elif sort_field == "db_id" and self.db_id_field != "db_id":
                        sort_field = self.db_id_field
                    direction = "DESC" if options.sort_desc else "ASC"
                    order_clause = (
                        f" ORDER BY {quote_identifier(sort_field)} {direction}"
                    )
                if options.limit > 0:
                    limit_clause = f" LIMIT {options.limit}"
                if options.offset > 0:
                    offset_clause = f" OFFSET {options.offset}"

                id_query = f"SELECT {quote_identifier(id_field_to_select)} FROM {quote_identifier(self._table_name)} WHERE {where_clause}{order_clause}{limit_clause}{offset_clause}"
                async with conn.execute(id_query, params) as cursor:
                    rows = await cursor.fetchall()
                if not rows:
                    return 0
                ids_to_update = [row[id_field_to_select] for row in rows]
                placeholders = ["?" for _ in ids_to_update]
                final_where_clause = f"{quote_identifier(id_field_to_select)} IN ({','.join(placeholders)})"
                final_where_params = ids_to_update
            else:
                final_where_clause, final_where_params, _ = self._transform_expression(
                    options.expression, 1
                )
                final_where_clause = re.sub(r"\$\d+", "?", final_where_clause)

            update_payload = prepare_for_storage(update.build())
            set_clause, set_params = self._build_set_clause(update_payload, 1)
            set_clause = re.sub(r"\$\d+", "?", set_clause)

            if not set_clause:
                logger_adapter.warning("Update many resulted in empty SET clause.")
                return 0

            query = f"UPDATE {quote_identifier(self._table_name)} SET {set_clause} WHERE {final_where_clause}"
            final_params = set_params + final_where_params

            async with conn.execute(query, final_params) as cursor:
                modified_count = cursor.rowcount
            await conn.commit()

        except Exception as e:
            await conn.rollback()
            logger_adapter.error(f"Error during update_many: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

        return modified_count

    async def delete_many(
        self,
        options: QueryOptions,
        logger_adapter: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger_adapter.debug(f"Deleting many {self.entity_type.__name__} ...")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete.")

        id_field_to_select = self._db_id_field
        final_where_clause = ""
        final_where_params = []
        ids_to_delete = []

        conn = await self._get_connection()
        deleted_count = 0
        try:
            if (
                options.limit > 0
                or options.offset > 0
                or options.sort_by
                or options.random_order
            ):
                where_clause, params, _ = self._transform_expression(
                    options.expression, 1
                )
                where_clause = re.sub(r"\$\d+", "?", where_clause)
                order_clause = ""
                limit_clause = ""
                offset_clause = ""
                if options.random_order:
                    order_clause = " ORDER BY RANDOM()"
                elif options.sort_by:
                    sort_field = options.sort_by
                    if sort_field == "id" and self.app_id_field != "id":
                        sort_field = self.app_id_field
                    elif sort_field == "db_id" and self.db_id_field != "db_id":
                        sort_field = self.db_id_field
                    direction = "DESC" if options.sort_desc else "ASC"
                    order_clause = (
                        f" ORDER BY {quote_identifier(sort_field)} {direction}"
                    )
                if options.limit > 0:
                    limit_clause = f" LIMIT {options.limit}"
                if options.offset > 0:
                    offset_clause = f" OFFSET {options.offset}"
                id_query = f"SELECT {quote_identifier(id_field_to_select)} FROM {quote_identifier(self._table_name)} WHERE {where_clause}{order_clause}{limit_clause}{offset_clause}"
                async with conn.execute(id_query, params) as cursor:
                    rows = await cursor.fetchall()
                if not rows:
                    return 0
                ids_to_delete = [row[id_field_to_select] for row in rows]
                placeholders = ["?" for _ in ids_to_delete]
                final_where_clause = f"{quote_identifier(id_field_to_select)} IN ({','.join(placeholders)})"
                final_where_params = ids_to_delete
            else:
                final_where_clause, final_where_params, _ = self._transform_expression(
                    options.expression, 1
                )
                final_where_clause = re.sub(r"\$\d+", "?", final_where_clause)

            query = f"DELETE FROM {quote_identifier(self._table_name)} WHERE {final_where_clause}"

            async with conn.execute(query, final_where_params) as cursor:
                deleted_count = cursor.rowcount
            await conn.commit()

        except Exception as e:
            await conn.rollback()
            logger_adapter.error(f"Error during delete_many: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

        return deleted_count

    async def list(
        self, logger_adapter: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        options = options or QueryOptions()
        logger_adapter.debug(f"Listing {self.entity_type.__name__} ...")

        query, params = self._build_query(options)
        query = re.sub(r"\$\d+", "?", query)  # Convert placeholders $n -> ?

        logger_adapter.debug(f"SQLite Query: {query}")
        logger_adapter.debug(f"SQLite Params: {params}")

        conn = await self._get_connection()
        try:
            async with conn.execute(query, params) as cursor:
                async for row in cursor:
                    yield self._row_to_entity(dict(row))
        except Exception as e:
            logger_adapter.error(f"Error during list: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

    async def count(
        self, logger_adapter: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        options = options or QueryOptions()
        logger_adapter.debug(f"Counting {self.entity_type.__name__} ...")

        base_query, params = "", []
        if options.expression:
            base_query, params, _ = self._transform_expression(options.expression, 1)
            base_query = re.sub(r"\$\d+", "?", base_query)

        count_query = f"SELECT COUNT(*) FROM {quote_identifier(self._table_name)}"
        if base_query:
            count_query += f" WHERE {base_query}"

        result = 0
        conn = await self._get_connection()
        try:
            async with conn.execute(count_query, params) as cursor:
                row = await cursor.fetchone()
                result = row[0] if row else 0
        except Exception as e:
            logger_adapter.error(f"Error during count: {e}", exc_info=True)
            raise
        finally:
            await conn.close()

        return cast(int, result)

    # --- Internal Helper Methods ---

    def _build_query(self, options: QueryOptions) -> Tuple[str, List[Any]]:
        """Builds the SELECT query with WHERE, ORDER BY, LIMIT, OFFSET using $n placeholders."""
        where_clause, params, next_param_idx = (
            self._transform_expression(options.expression, 1)
            if options.expression
            else ("", [], 1)
        )
        query = f"SELECT * FROM {quote_identifier(self._table_name)}"
        if where_clause:
            query += f" WHERE {where_clause}"
        if options.random_order:
            query += " ORDER BY RANDOM()"
        elif options.sort_by:
            sort_field = options.sort_by
            if sort_field == "id" and self._app_id_field != "id":
                sort_field = self._app_id_field
            elif sort_field == "db_id" and self._db_id_field != "db_id":
                sort_field = self._db_id_field
            direction = "DESC" if options.sort_desc else "ASC"
            query += f" ORDER BY {quote_identifier(sort_field)} {direction}"
        else:
            query += f" ORDER BY {quote_identifier(self._db_id_field)} ASC"
        limit_val = options.limit if options.limit > 0 else -1
        query += f" LIMIT ${next_param_idx}"
        params.append(limit_val)
        if options.offset > 0:
            query += f" OFFSET ${next_param_idx + 1}"
            params.append(options.offset)
        return query, params

    def _transform_expression(
        self, expr: Dict[str, Any], start_index: int = 1
    ) -> Tuple[str, List[Any], int]:
        """Transforms DSL expression to SQLite WHERE clause using $n placeholders."""
        clauses = []
        params = []
        current_index = start_index
        if not expr:
            return "", [], current_index

        if "and" in expr:
            sub_clauses = []
            for sub_expr in expr["and"]:
                clause, sub_params, current_index = self._transform_expression(
                    sub_expr, current_index
                )
                if clause:
                    sub_clauses.append(f"({clause})")
                params.extend(sub_params)
            return (
                " AND ".join(sub_clauses) if sub_clauses else "",
                params,
                current_index,
            )
        if "or" in expr:
            sub_clauses = []
            for sub_expr in expr["or"]:
                clause, sub_params, current_index = self._transform_expression(
                    sub_expr, current_index
                )
                if clause:
                    sub_clauses.append(f"({clause})")
                params.extend(sub_params)
            return (
                " OR ".join(sub_clauses) if sub_clauses else "",
                params,
                current_index,
            )

        for field, condition in expr.items():
            is_json_path = "." in field
            sql_field_expr = ""
            if is_json_path:
                parts = field.split(".")
                col_name = quote_identifier(parts[0])
                json_path_sqlite = f"$.{'.'.join(parts[1:])}"
                sql_field_expr = f"JSON_EXTRACT({col_name}, '{json_path_sqlite}')"
            else:
                sql_field_expr = quote_identifier(field)

            operator = "eq"
            value = condition
            if isinstance(condition, dict):
                operator = condition.get("operator", "eq").lower()
                value = condition.get("value")

            prepared_value = self._prepare_value_for_sqlite(value)

            sql_op_map = {
                "eq": "=",
                "ne": "!=",
                "gt": ">",
                "ge": ">=",
                "lt": "<",
                "le": "<=",
            }
            clause = ""

            if operator in sql_op_map:
                comparison_expr = sql_field_expr
                if is_json_path and isinstance(prepared_value, (int, float)):
                    comparison_expr = f"CAST({sql_field_expr} AS REAL)"
                elif not is_json_path and isinstance(
                    prepared_value, bool
                ):  # Compare bools as integers
                    prepared_value = 1 if prepared_value else 0

                clause = f"{comparison_expr} {sql_op_map[operator]} ${current_index}"
                params.append(prepared_value)
                current_index += 1
            elif operator == "in" or operator == "nin":
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"Value for {operator} must be list/tuple")
                if not value:
                    clause = "0" if operator == "in" else "1"
                else:
                    prepared_list = [self._prepare_value_for_sqlite(v) for v in value]
                    placeholders = [
                        f"${current_index + i}" for i in range(len(prepared_list))
                    ]
                    sql_operator = "IN" if operator == "in" else "NOT IN"
                    clause = (
                        f"{sql_field_expr} {sql_operator} ({', '.join(placeholders)})"
                    )
                    params.extend(prepared_list)
                    current_index += len(prepared_list)
            elif operator == "contains":
                value_to_find_json = json.dumps(value)
                # Use json_each for arrays; use LIKE for simple string contains in non-JSON text
                # Assuming list/dict fields are stored as JSON TEXT
                if is_json_path or field in [
                    "tags",
                    "metadata",
                    "profile",
                ]:  # Check if field likely contains JSON
                    clause = f"EXISTS (SELECT 1 FROM json_each({sql_field_expr}) WHERE json_each.value = json(${current_index}))"
                else:  # Fallback to simple LIKE for non-JSON text fields
                    clause = f"{sql_field_expr} LIKE '%' || ${current_index} || '%'"
                params.append(
                    value_to_find_json
                    if is_json_path or field in ["tags", "metadata", "profile"]
                    else value
                )  # Pass raw value for LIKE
                current_index += 1
            elif operator == "startswith":
                clause = f"{sql_field_expr} LIKE ${current_index} || '%'"
                params.append(prepared_value)
                current_index += 1
            elif operator == "endswith":
                clause = f"{sql_field_expr} LIKE '%' || ${current_index}"
                params.append(prepared_value)
                current_index += 1
            elif operator == "like":
                # Add '%' wildcards for substring search
                clause = f"{sql_field_expr} LIKE '%' || ${current_index} || '%'"
                params.append(prepared_value)
                current_index += 1
            elif operator == "exists":
                clause = f"{sql_field_expr} IS {'NOT ' if value else ''}NULL"  # Value is True/False
            elif operator == "regex":
                clause = f"{sql_field_expr} REGEXP ${current_index}"
                params.append(prepared_value)
                current_index += 1
            else:
                raise ValueError(f"Unsupported operator: {operator}")
            if clause:
                clauses.append(clause)

        return " AND ".join(clauses) if clauses else "", params, current_index

    def _build_set_clause(
        self, update_payload: Dict[str, Any], start_index: int
    ) -> Tuple[str, List[Any]]:
        """Builds the SET clause for SQLite UPDATE using $n placeholders."""
        set_parts = []
        params = []
        current_index = start_index
        if "$set" in update_payload:
            for field, value in update_payload["$set"].items():
                prepared_value = self._prepare_value_for_sqlite(value)
                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path_sqlite = f"$.{'.'.join(parts[1:])}"
                    # Need valid JSON for JSON() function, prepared_value might already be stringified
                    json_arg = (
                        prepared_value
                        if isinstance(prepared_value, str)
                        and prepared_value.startswith(("[", "{"))
                        else json.dumps(prepared_value)
                    )
                    set_parts.append(
                        f"{col_name} = JSON_SET(COALESCE({col_name}, '{{}}'), '{json_path_sqlite}', JSON(${current_index}))"
                    )
                    params.append(json_arg)
                    current_index += 1
                else:
                    sql_field = quote_identifier(field)
                    set_parts.append(f"{sql_field} = ${current_index}")
                    params.append(prepared_value)
                    current_index += 1
        if "$unset" in update_payload:
            for field in update_payload["$unset"]:
                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path_sqlite = f"$.{'.'.join(parts[1:])}"
                    set_parts.append(
                        f"{col_name} = JSON_REMOVE(COALESCE({col_name}, '{{}}'), '{json_path_sqlite}')"
                    )
                else:
                    sql_field = quote_identifier(field)
                    set_parts.append(f"{sql_field} = NULL")
        if "$push" in update_payload:
            for field, value in update_payload["$push"].items():
                prepared_value = self._prepare_value_for_sqlite(value)
                json_arg = (
                    prepared_value
                    if isinstance(prepared_value, str)
                    and prepared_value.startswith(("[", "{"))
                    else json.dumps(prepared_value)
                )
                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path_sqlite = f"$.{'.'.join(parts[1:])}"
                    set_parts.append(
                        f"{col_name} = JSON_INSERT(COALESCE({col_name}, '{{}}'), '{json_path_sqlite}[#]', JSON(${current_index}))"
                    )
                    params.append(json_arg)
                    current_index += 1
                else:
                    sql_field = quote_identifier(field)
                    set_parts.append(
                        f"{sql_field} = JSON_INSERT(COALESCE({sql_field}, '[]'), '$[#]', JSON(${current_index}))"
                    )
                    params.append(json_arg)
                    current_index += 1
        if "$pull" in update_payload:
            logger.warning("$pull op SQLite simplified")
            pass
        if "$pop" in update_payload:
            logger.warning("$pop op SQLite simplified")
            pass
        if not set_parts and (
            "$pull" not in update_payload and "$pop" not in update_payload
        ):
            raise ValueError(
                "Update object did not result in any valid SET/UNSET/PUSH clauses."
            )
        return ", ".join(set_parts), params

    def _entity_to_initial_dict(self, entity: T) -> Dict[str, Any]:
        """Helper to get the raw dict from the entity."""
        if hasattr(entity, "model_dump"):
            try:
                entity_dict = entity.model_dump(mode="json", by_alias=True)
            except Exception:
                entity_dict = entity.model_dump(by_alias=True)
        elif hasattr(entity, "dict"):
            entity_dict = entity.dict(by_alias=True)
        elif is_dataclass(entity):
            entity_dict = asdict(entity)
        elif hasattr(entity, "__dict__"):
            entity_dict = dict(entity.__dict__)
        else:
            raise TypeError(f"Cannot convert entity {type(entity)} to dict")
        return entity_dict

    def _prepare_value_for_sqlite(self, value: Any) -> Any:
        """Prepares a single basic Python value for SQLite storage."""
        if value is None:
            return None
        elif isinstance(value, bool):
            return 1 if value else 0
        elif isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, datetime):
            # Ensure UTC for storage consistency
            if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                # Assume naive is UTC - CHANGE THIS IF LOCAL TIME IS USED
                # value = value.replace(tzinfo=timezone.utc) # Option 1: Assume UTC
                # Option 2: Assume local and convert to UTC (requires system tz info)
                # value = value.astimezone(timezone.utc) # Use this if naive times are local
                # Sticking with Option 1 for simplicity as source data likely UTC
                value = value.replace(tzinfo=timezone.utc)
            else:  # Convert aware datetime to UTC
                value = value.astimezone(timezone.utc)
            return value.isoformat().replace("+00:00", "Z")  # Use Z for UTC marker
        elif isinstance(value, (int, float, str, bytes)):
            return value
        else:  # Fallback for other types (like Pydantic custom types)
            logger.debug(f"SQLite Prep: Auto-converting type {type(value)} to string.")
            return str(value)

    def _prepare_dict_for_storage(self, entity_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares the dictionary for SQLite storage."""
        prepared_basic_types_dict = prepare_for_storage(entity_dict)
        final_dict = {
            k: self._prepare_value_for_sqlite(v)
            for k, v in prepared_basic_types_dict.items()
        }
        return final_dict

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Converts entity to dict ready for SQLite, removing db_id if None."""
        initial_dict = self._entity_to_initial_dict(entity)
        prepared_dict = self._prepare_dict_for_storage(initial_dict)
        if (
            self._db_id_field in prepared_dict
            and prepared_dict[self._db_id_field] is None
        ):
            del prepared_dict[self._db_id_field]
        return prepared_dict

    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Converts SQLite row (dict) to entity, handling JSON/bool/datetime/URL."""
        entity_data = {}
        anno = getattr(self._entity_cls, "__annotations__", {})
        for key, value in row.items():
            target_type_hint = anno.get(key)
            origin_type = get_origin(target_type_hint)
            actual_target_type = origin_type or target_type_hint
            if value is None:
                entity_data[key] = None
                continue
            converted = False
            if target_type_hint is datetime and isinstance(value, str):
                try:
                    dt_str = value.replace("Z", "+00:00")
                    entity_data[key] = datetime.fromisoformat(dt_str)
                    converted = True
                except ValueError:
                    entity_data[key] = None
                    converted = True
            elif target_type_hint is bool and isinstance(value, int):
                entity_data[key] = bool(value)
                converted = True
            elif actual_target_type in (list, dict) and isinstance(value, str):
                try:
                    entity_data[key] = json.loads(value)
                    converted = True
                except json.JSONDecodeError:
                    logger.warning(
                        f"Field '{key}' expected {actual_target_type}, got non-JSON string: '{value[:50]}...'"
                    )
                    entity_data[key] = value
                    converted = True

            if not converted and target_type_hint and isinstance(value, str):
                possible_types = (
                    get_args(target_type_hint)
                    if origin_type is Union
                    else (
                        [target_type_hint] if isinstance(target_type_hint, type) else []
                    )
                )
                is_url_type = False
                url_target_class = None
                for p_type in possible_types:
                    if isinstance(p_type, type) and issubclass(p_type, AnyUrl):
                        is_url_type = True
                        url_target_class = p_type
                        break
                if is_url_type and url_target_class:
                    try:
                        entity_data[key] = url_target_class(value)
                        converted = True
                    except Exception:
                        logger.warning(
                            f"Failed hydration URL type {url_target_class} for field '{key}' value '{value}'"
                        )
                        entity_data[key] = value
                        converted = True

            if not converted:
                entity_data[key] = value

        db_id_val = entity_data.get(self._db_id_field)
        if (
            self._app_id_field not in entity_data
            and self._app_id_field in anno
            and db_id_val is not None
        ):
            entity_data[self._app_id_field] = str(db_id_val)
        if self._db_id_field != self._app_id_field and self._db_id_field in entity_data:
            if self._db_id_field not in anno:
                del entity_data[self._db_id_field]
        try:
            return self._entity_cls(**entity_data)
        except TypeError as e:
            logger.error(
                f"Error instantiating {self._entity_cls.__name__} with data: {entity_data}. Error: {e}",
                exc_info=True,
            )
            import inspect

            sig = inspect.signature(self._entity_cls.__init__)
            expected = set(sig.parameters.keys()) - {"self"}
            provided = set(entity_data.keys())
            logger.error(
                f"Missing params: {expected - provided}, Extra params: {provided - expected}"
            )
            raise ValueError(
                f"Mismatch row/entity schema {self._entity_cls.__name__}: {e}"
            ) from e
