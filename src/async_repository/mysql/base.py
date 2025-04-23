# ./mysql/base.py

import json
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
)

import aiomysql  # Requires: pip install aiomysql
from aiomysql import Pool
from aiomysql.cursors import DictCursor  # To get rows as dictionaries

from async_repository.base.interfaces import Repository
from async_repository.base.query import QueryOptions
from async_repository.base.update import Update
from async_repository.base.exceptions import (
    ObjectNotFoundException,
    KeyAlreadyExistsException,
)
from async_repository.base.utils import prepare_for_storage

T = TypeVar("T")
base_logger = logging.getLogger(__name__)


# Helper to quote identifiers for MySQL (using backticks)
def quote_identifier(identifier: str) -> str:
    """Quotes an identifier for safe use in MySQL queries."""
    if not isinstance(identifier, str):
        raise TypeError("Identifier must be a string")
    if "`" in identifier:
        # Basic protection against injection in identifiers
        raise ValueError("Identifier cannot contain backticks")
    return f"`{identifier}`"


class MySQLRepository(Repository[T], Generic[T]):
    """
    MySQL repository implementation using aiomysql.

    Handles basic CRUD operations, DSL queries, and MongoDB-style updates
    translated to MySQL JSON functions where applicable.
    """

    def __init__(
        self,
        pool: Pool,
        table_name: str,
        entity_cls: Type[T],
        app_id_field: str = "id",
        db_id_field: str = "db_id",  # Often the auto-incrementing PK in MySQL
    ):
        self._pool = pool
        self._table_name = table_name
        self._entity_cls = entity_cls
        self._app_id_field = app_id_field
        self._db_id_field = db_id_field  # Corresponds to the primary key column

        base_logger.debug(
            "Initialized MySQL repository for class: %s, pool: %s, table: %s",
            entity_cls.__name__,
            pool,
            table_name,
        )

    @property
    def entity_type(self) -> Type[T]:
        return self._entity_cls

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        """The primary key column name in the MySQL table."""
        return self._db_id_field

    async def _get_connection(self):
        """Acquires a connection from the pool."""
        return await self._pool.acquire()

    def _release_connection(self, conn):
        """Releases a connection back to the pool."""
        self._pool.release(conn)

    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[
            float
        ] = None,  # Timeout not directly supported by aiomysql execute
        use_db_id: bool = False,
    ) -> T:
        field = self._db_id_field if use_db_id else self._app_id_field
        id_type = "database (PK)" if use_db_id else "application"
        logger.debug(f"Getting {self._entity_cls.__name__} with {id_type} ID: {id}")

        query = f"SELECT * FROM {quote_identifier(self._table_name)} WHERE {quote_identifier(field)} = %s"
        params = [id]

        conn = None
        try:
            conn = await self._get_connection()
            async with conn.cursor(DictCursor) as cursor:
                await cursor.execute(query, params)
                row = await cursor.fetchone()
        finally:
            if conn:
                self._release_connection(conn)

        if not row:
            raise ObjectNotFoundException(
                f"{self._entity_cls.__name__} with {id_type} ID {id} not found"
            )
        return self._row_to_entity(row)

    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        return await self.get(str(db_id), logger, timeout, use_db_id=True)

    async def store(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,  # Less relevant if app_id isn't auto-generated
        return_value: bool = False,
    ) -> Optional[T]:
        self.validate_entity(entity)
        entity_dict = self._entity_to_dict(entity)

        # Generate app_id if needed and not present
        app_id = entity_dict.get(self._app_id_field)
        if generate_app_id and app_id is None:
            generated_app_id = self.id_generator()
            entity_dict[self._app_id_field] = generated_app_id
            setattr(
                entity, self._app_id_field, generated_app_id
            )  # Update original entity if needed
            app_id = generated_app_id

        # Remove db_id_field if it's None, assuming it's auto-increment
        # If db_id_field needs generation (e.g., UUID PK), handle it here or ensure it's set
        if self._db_id_field in entity_dict and entity_dict[self._db_id_field] is None:
            del entity_dict[self._db_id_field]  # Let MySQL handle auto-increment PK

        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = ["%s"] * len(fields)

        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} "
            f"({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)})"
        )

        conn = None
        new_db_id = None
        try:
            conn = await self._get_connection()
            async with conn.cursor(DictCursor) as cursor:
                await cursor.execute(query, values)
                await conn.commit()
                new_db_id = (
                    cursor.lastrowid
                )  # Get the auto-incremented ID if applicable

                if return_value:
                    # Fetch the newly inserted row to return the complete entity
                    inserted_id_to_fetch = (
                        new_db_id
                        if new_db_id is not None
                        else entity_dict.get(self._db_id_field)
                    )
                    if inserted_id_to_fetch is None and app_id is not None:
                        # If PK is not auto-increment, try fetching by app_id
                        fetch_field = self._app_id_field
                        fetch_value = app_id
                    elif inserted_id_to_fetch is not None:
                        fetch_field = self._db_id_field
                        fetch_value = inserted_id_to_fetch
                    else:
                        # Cannot determine how to fetch the stored entity
                        logger.warning(
                            "Cannot reliably fetch stored entity after insert without a known ID."
                        )
                        return None  # Or raise an error

                    fetch_query = f"SELECT * FROM {quote_identifier(self._table_name)} WHERE {quote_identifier(fetch_field)} = %s"
                    await cursor.execute(fetch_query, [fetch_value])
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_entity(row)
                    else:
                        # Should not happen if insert succeeded, but handle defensively
                        logger.error(
                            f"Failed to fetch entity after insert with {fetch_field}={fetch_value}"
                        )
                        return None

        except aiomysql.IntegrityError as e:
            await conn.rollback()  # Rollback on error
            # Check common MySQL error codes for duplicate entry
            if e.args[0] == 1062:  # Error code for duplicate entry
                # Try to determine which key caused the violation (might be fragile)
                constraint_msg = str(e).lower()
                field_name = (
                    self._app_id_field
                    if f"key '{self._app_id_field}" in constraint_msg
                    else (
                        self._db_id_field
                        if f"key '{self._db_id_field}" in constraint_msg
                        else "unique constraint"
                    )
                )
                id_value = entity_dict.get(
                    (
                        self._app_id_field
                        if field_name == self._app_id_field
                        else self._db_id_field
                    ),
                    "unknown",
                )
                raise KeyAlreadyExistsException(
                    f"{self._entity_cls.__name__} with {field_name} '{id_value}' already exists."
                ) from e
            else:  # Re-raise other integrity errors
                raise e
        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                self._release_connection(conn)

        if return_value:
            # This part is reached only if fetching failed or wasn't possible
            return None
        else:
            return None

    async def upsert(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
    ) -> None:
        self.validate_entity(entity)
        entity_dict = self._entity_to_dict(entity)

        # Generate app_id if needed
        if generate_app_id and entity_dict.get(self._app_id_field) is None:
            entity_dict[self._app_id_field] = self.id_generator()

        # Remove db_id if None (for auto-increment)
        if self._db_id_field in entity_dict and entity_dict[self._db_id_field] is None:
            del entity_dict[self._db_id_field]

        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = ["%s"] * len(fields)

        # Build the ON DUPLICATE KEY UPDATE part
        update_clauses = [
            f"{quote_identifier(field)} = VALUES({quote_identifier(field)})"
            for field in fields
            # Don't update the primary key itself in the UPDATE clause
            if field != self._db_id_field
        ]

        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} ({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON DUPLICATE KEY UPDATE {', '.join(update_clauses)}"
        )

        conn = None
        try:
            conn = await self._get_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(query, values)
                await conn.commit()
        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                self._release_connection(conn)

    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        logger.debug(
            f"Updating one {self._entity_cls.__name__} with options: {options.expression}, update: {update.build()}"
        )
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        where_clause, where_params, _ = self._transform_expression(
            options.expression, 1
        )
        update_payload = prepare_for_storage(update.build())  # Prepare values early
        set_clause, set_params = self._build_set_clause(
            update_payload, len(where_params) + 1
        )

        if not set_clause:
            logger.warning(
                "Update command resulted in empty SET clause, skipping update."
            )
            # Optionally fetch and return if return_value is True, otherwise return None
            if return_value:
                try:
                    return await self.find_one(logger=logger, options=options)
                except ObjectNotFoundException:
                    raise ObjectNotFoundException(
                        f"{self._entity_cls.__name__} not found with criteria {options.expression}"
                    )
            return None

        query = (
            f"UPDATE {quote_identifier(self._table_name)} SET {set_clause} "
            f"WHERE {where_clause} LIMIT 1"
        )
        params = where_params + set_params

        conn = None
        updated_count = 0
        try:
            conn = await self._get_connection()
            async with conn.cursor(DictCursor) as cursor:
                updated_count = await cursor.execute(query, params)
                await conn.commit()

            if updated_count == 0:
                # Verify if the object exists at all before raising not found
                # (it might exist but not match the update criteria *now*)
                # Re-querying adds overhead, consider if this check is essential.
                check_exists_query = f"SELECT 1 FROM {quote_identifier(self._table_name)} WHERE {where_clause} LIMIT 1"
                async with conn.cursor() as check_cursor:
                    await check_cursor.execute(check_exists_query, where_params)
                    exists = await check_cursor.fetchone()
                    if not exists:
                        raise ObjectNotFoundException(
                            f"{self._entity_cls.__name__} not found with criteria {options.expression}"
                        )
                    else:
                        # Object exists, but wasn't updated (maybe already had the target values?)
                        logger.debug(
                            "update_one matched criteria but resulted in 0 updated rows."
                        )

            if return_value:
                # Re-fetch the potentially updated entity using the original criteria
                # This ensures we get the state *after* the update attempt
                fetch_query = f"SELECT * FROM {quote_identifier(self._table_name)} WHERE {where_clause} LIMIT 1"
                async with conn.cursor(DictCursor) as cursor:
                    await cursor.execute(fetch_query, where_params)
                    row = await cursor.fetchone()
                    if not row:
                        # This case should ideally be caught by the updated_count check or exists check
                        raise ObjectNotFoundException(
                            f"{self._entity_cls.__name__} not found after update attempt with criteria {options.expression}"
                        )
                    return self._row_to_entity(row)

        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                self._release_connection(conn)

        return None  # Only reached if return_value is False

    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(
            f"Updating many {self._entity_cls.__name__} with options: {options.expression}, update: {update.build()}"
        )
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        update_payload = prepare_for_storage(update.build())  # Prepare values
        set_clause, set_params = self._build_set_clause(
            update_payload, 1
        )  # Placeholder indices start from 1

        if not set_clause:
            logger.warning(
                "Update command resulted in empty SET clause, skipping update."
            )
            return 0

        # If limit/offset/sort/random are involved, we need to fetch IDs first
        if (
            options.limit > 0
            or options.offset > 0
            or options.sort_by
            or options.random_order
        ):
            where_clause_ids, where_params_ids, _ = self._transform_expression(
                options.expression, 1
            )

            order_clause = ""
            if options.random_order:
                order_clause = " ORDER BY RAND()"
            elif options.sort_by:
                sort_field = options.sort_by
                # Handle 'id' alias for app_id_field
                if sort_field == "id" and self._app_id_field != "id":
                    sort_field = self._app_id_field
                elif sort_field == "db_id" and self._db_id_field != "db_id":
                    sort_field = self._db_id_field
                direction = "DESC" if options.sort_desc else "ASC"
                order_clause = f" ORDER BY {quote_identifier(sort_field)} {direction}"

            # MySQL LIMIT syntax: LIMIT offset, row_count
            limit_clause = (
                f" LIMIT {options.offset}, {options.limit}" if options.limit > 0 else ""
            )
            if not limit_clause and options.offset > 0:
                # LIMIT is required for OFFSET in MySQL
                limit_clause = f" LIMIT {options.offset}, 18446744073709551615"  # Max value for BIGINT UNSIGNED

            id_query = (
                f"SELECT {quote_identifier(self._db_id_field)} FROM {quote_identifier(self._table_name)} "
                f"WHERE {where_clause_ids}{order_clause}{limit_clause}"
            )

            conn_ids = None
            ids_to_update = []
            try:
                conn_ids = await self._get_connection()
                async with conn_ids.cursor() as cursor:
                    await cursor.execute(id_query, where_params_ids)
                    rows = await cursor.fetchall()
                    ids_to_update = [row[0] for row in rows]
            finally:
                if conn_ids:
                    self._release_connection(conn_ids)

            if not ids_to_update:
                return 0

            # Build final WHERE clause using IN (id1, id2, ...)
            id_placeholders = ["%s"] * len(ids_to_update)
            final_where_clause = f"{quote_identifier(self._db_id_field)} IN ({', '.join(id_placeholders)})"
            # Parameters for the final update: SET params first, then WHERE (ID) params
            final_params = set_params + ids_to_update

        else:
            # No limit/offset/sort, update directly based on expression
            final_where_clause, where_params, _ = self._transform_expression(
                options.expression, len(set_params) + 1
            )
            # Parameters for the final update: SET params first, then WHERE params
            final_params = set_params + where_params

        query = (
            f"UPDATE {quote_identifier(self._table_name)} SET {set_clause} "
            f"WHERE {final_where_clause}"
        )

        conn = None
        updated_count = 0
        try:
            conn = await self._get_connection()
            async with conn.cursor() as cursor:
                updated_count = await cursor.execute(query, final_params)
                await conn.commit()
        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                self._release_connection(conn)

        return updated_count

    async def delete_many(
        self,
        options: QueryOptions,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(
            f"Deleting many {self._entity_cls.__name__} with options: {options.expression}"
        )
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete.")

        # If limit/offset/sort/random are involved, fetch IDs first
        if (
            options.limit > 0
            or options.offset > 0
            or options.sort_by
            or options.random_order
        ):
            where_clause_ids, where_params_ids, _ = self._transform_expression(
                options.expression, 1
            )

            order_clause = ""
            if options.random_order:
                order_clause = " ORDER BY RAND()"
            elif options.sort_by:
                sort_field = options.sort_by
                if sort_field == "id" and self._app_id_field != "id":
                    sort_field = self._app_id_field
                elif sort_field == "db_id" and self._db_id_field != "db_id":
                    sort_field = self._db_id_field
                direction = "DESC" if options.sort_desc else "ASC"
                order_clause = f" ORDER BY {quote_identifier(sort_field)} {direction}"

            limit_clause = (
                f" LIMIT {options.offset}, {options.limit}" if options.limit > 0 else ""
            )
            if not limit_clause and options.offset > 0:
                limit_clause = f" LIMIT {options.offset}, 18446744073709551615"

            id_query = (
                f"SELECT {quote_identifier(self._db_id_field)} FROM {quote_identifier(self._table_name)} "
                f"WHERE {where_clause_ids}{order_clause}{limit_clause}"
            )

            conn_ids = None
            ids_to_delete = []
            try:
                conn_ids = await self._get_connection()
                async with conn_ids.cursor() as cursor:
                    await cursor.execute(id_query, where_params_ids)
                    rows = await cursor.fetchall()
                    ids_to_delete = [row[0] for row in rows]
            finally:
                if conn_ids:
                    self._release_connection(conn_ids)

            if not ids_to_delete:
                return 0

            id_placeholders = ["%s"] * len(ids_to_delete)
            final_where_clause = f"{quote_identifier(self._db_id_field)} IN ({', '.join(id_placeholders)})"
            final_params = ids_to_delete

        else:
            # No limit/offset/sort, delete directly based on expression
            final_where_clause, final_params, _ = self._transform_expression(
                options.expression, 1
            )

        query = f"DELETE FROM {quote_identifier(self._table_name)} WHERE {final_where_clause}"

        conn = None
        deleted_count = 0
        try:
            conn = await self._get_connection()
            async with conn.cursor() as cursor:
                deleted_count = await cursor.execute(query, final_params)
                await conn.commit()
        except Exception as e:
            if conn:
                await conn.rollback()
            raise e
        finally:
            if conn:
                self._release_connection(conn)

        return deleted_count

    # delete_one can reuse the base implementation which calls delete_many

    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        options = options or QueryOptions()
        logger.debug(
            f"Listing {self._entity_cls.__name__} with options: {options.__dict__}"
        )

        where_clause, params, next_param_idx = (
            self._transform_expression(options.expression, 1)
            if options.expression
            else ("", [], 1)
        )

        query = f"SELECT * FROM {quote_identifier(self._table_name)}"
        if where_clause:
            query += f" WHERE {where_clause}"

        # Order
        if options.random_order:
            query += " ORDER BY RAND()"
        elif options.sort_by:
            sort_field = options.sort_by
            if sort_field == "id" and self._app_id_field != "id":
                sort_field = self._app_id_field
            elif sort_field == "db_id" and self._db_id_field != "db_id":
                sort_field = self._db_id_field
            direction = "DESC" if options.sort_desc else "ASC"
            query += f" ORDER BY {quote_identifier(sort_field)} {direction}"
        else:
            # Default sort by primary key (db_id_field)
            query += f" ORDER BY {quote_identifier(self._db_id_field)} ASC"

        # Limit & Offset
        # Using %s for limit/offset requires MySQL 8+ or specific config, pass directly for wider compatibility
        # Ensure they are integers!
        limit = (
            int(options.limit) if options.limit > 0 else 18446744073709551615
        )  # Max BIGINT UNSIGNED
        offset = int(options.offset) if options.offset >= 0 else 0
        query += f" LIMIT {offset}, {limit}"  # MySQL LIMIT syntax

        conn = None
        try:
            conn = await self._get_connection()
            async with conn.cursor(DictCursor) as cursor:
                await cursor.execute(query, params)
                # Fetch rows one by one or in chunks if memory is a concern
                while True:
                    row = await cursor.fetchone()
                    if row is None:
                        break
                    yield self._row_to_entity(row)
        finally:
            if conn:
                self._release_connection(conn)

    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        options = options or QueryOptions()
        logger.debug(
            f"Counting {self._entity_cls.__name__} with options: {options.__dict__}"
        )

        where_clause, params, _ = (
            self._transform_expression(options.expression, 1)
            if options.expression
            else ("", [], 1)
        )

        query = f"SELECT COUNT(*) as count FROM {quote_identifier(self._table_name)}"
        if where_clause:
            query += f" WHERE {where_clause}"

        conn = None
        count = 0
        try:
            conn = await self._get_connection()
            async with conn.cursor(DictCursor) as cursor:
                await cursor.execute(query, params)
                result = await cursor.fetchone()
                count = result["count"] if result else 0
        finally:
            if conn:
                self._release_connection(conn)

        return cast(int, count)

    # --- Helper Methods ---

    def _transform_expression(
        self, expr: Dict[str, Any], start_index: int = 1
    ) -> Tuple[str, List[Any], int]:
        """Transforms DSL expression into SQL WHERE clause and parameters."""
        clauses = []
        params = []
        current_index = start_index

        if not expr:
            return "1=1", [], current_index  # No conditions means match all

        if "and" in expr:
            sub_clauses = []
            for sub_expr in expr["and"]:
                clause, sub_params, current_index = self._transform_expression(
                    sub_expr, current_index
                )
                if clause != "1=1":  # Avoid adding trivial clauses
                    sub_clauses.append(f"({clause})")
                    params.extend(sub_params)
            # Handle case where 'and' list might be empty or only contain empty expressions
            return (
                " AND ".join(sub_clauses) if sub_clauses else "1=1",
                params,
                current_index,
            )
        if "or" in expr:
            sub_clauses = []
            for sub_expr in expr["or"]:
                clause, sub_params, current_index = self._transform_expression(
                    sub_expr, current_index
                )
                if clause != "1=1":
                    sub_clauses.append(f"({clause})")
                    params.extend(sub_params)
            # Handle case where 'or' list might be empty or only contain empty expressions
            return (
                " OR ".join(sub_clauses) if sub_clauses else "1=0",
                params,
                current_index,
            )  # OR of nothing is false

        for field, condition in expr.items():
            sql_operator = "="
            value = condition
            is_json_path = "." in field
            sql_field = ""

            # Prepare field name (handling JSON paths)
            if is_json_path:
                # Assumes column stores JSON. Format: col->"$.path.to.value"
                # Use JSON_EXTRACT for broader compatibility, JSON_UNQUOTE needed for string comparison
                parts = field.split(".")
                col_name = quote_identifier(parts[0])
                json_path = f"$.{'.'.join(parts[1:])}"
                # Default to text extraction, casting might be needed for numeric ops
                sql_field = f"JSON_UNQUOTE(JSON_EXTRACT({col_name}, %s))"
                params.append(json_path)
            else:
                sql_field = quote_identifier(field)

            # Prepare operator and value
            if isinstance(condition, dict):
                operator = condition.get("operator", "eq").lower()
                value = condition.get("value")

                # Apply prepare_for_storage specifically to the value part of the condition
                value = prepare_for_storage(value)

                if operator == "eq":
                    sql_operator = "="
                elif operator == "ne":
                    sql_operator = "!="
                elif operator == "gt":
                    sql_operator = ">"
                elif operator in ("ge", "gte"):
                    sql_operator = ">="
                elif operator == "lt":
                    sql_operator = "<"
                elif operator in ("le", "lte"):
                    sql_operator = "<="
                elif operator == "in":
                    sql_operator = "IN"
                elif operator == "nin":
                    sql_operator = "NOT IN"
                elif operator == "contains":
                    # For simple arrays (non-JSON): FIND_IN_SET(value, field) > 0 (if stored as comma-separated string)
                    # For JSON arrays: JSON_CONTAINS(field, %s)
                    if is_json_path:
                        # Adjust field for JSON_CONTAINS - needs the column itself
                        col_name = quote_identifier(field.split(".")[0])
                        # Need to check if the value exists within the array at the specific path
                        # JSON_CONTAINS(json_doc, candidate[, path])
                        sql_field = f"JSON_CONTAINS({col_name}, %s, %s)"
                        params[-1] = json.dumps(
                            value
                        )  # Candidate value needs to be JSON encoded scalar/object
                        params.append(json_path)  # Path where the array is expected
                        clause = f"{sql_field}"  # No operator needed, JSON_CONTAINS returns 0 or 1
                        clauses.append(clause)
                        continue  # Skip default clause generation
                    else:
                        # Assume native array or JSON array for non-nested field
                        sql_field = f"JSON_CONTAINS({quote_identifier(field)}, %s)"
                        params.append(
                            json.dumps(value)
                        )  # Value to search for, JSON encoded
                        clause = f"{sql_field}"
                        clauses.append(clause)
                        continue
                elif operator == "startswith":
                    sql_operator = "LIKE"
                    value = f"{value}%"
                elif operator == "endswith":
                    sql_operator = "LIKE"
                    value = f"%{value}"
                elif operator == "like":
                    sql_operator = "LIKE"
                    # Keep value as is (assuming user includes %)
                elif operator == "exists":
                    # Check for non-null for standard columns
                    # For JSON paths, check if the path exists
                    if is_json_path:
                        col_name = quote_identifier(field.split(".")[0])
                        sql_field = f"JSON_EXTRACT({col_name}, %s)"
                        # Path param already added
                    sql_operator = "IS NOT NULL" if value else "IS NULL"
                    clause = f"{sql_field} {sql_operator}"
                    clauses.append(clause)
                    continue  # Skip default clause generation
                elif operator == "regex":
                    sql_operator = "REGEXP"
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            else:
                # Simple equality check if condition is not a dict
                sql_operator = "="
                value = prepare_for_storage(value)  # Prepare value

            # Build clause
            if sql_operator in ("IN", "NOT IN"):
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"Value for {sql_operator} must be a list or tuple"
                    )
                if not value:
                    # Handle empty list for IN/NOT IN
                    clause = "1=0" if sql_operator == "IN" else "1=1"
                else:
                    in_placeholders = ["%s"] * len(value)
                    clause = (
                        f"{sql_field} {sql_operator} ({', '.join(in_placeholders)})"
                    )
                    params.extend(value)
            else:
                clause = f"{sql_field} {sql_operator} %s"
                params.append(value)

            clauses.append(clause)

        # Combine clauses with AND
        final_clause = " AND ".join(clauses) if clauses else "1=1"
        return final_clause, params, current_index

    def _build_set_clause(
        self, update_payload: Dict[str, Any], start_index: int
    ) -> Tuple[str, List[Any]]:
        """Builds the SET clause for SQL UPDATE from MongoDB-style payload."""
        set_parts = []
        params = []
        current_index = start_index

        # $set: Direct field updates
        if "$set" in update_payload:
            for field, value in update_payload["$set"].items():
                if "." in field:
                    # Nested field update using JSON_SET
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path = f"$.{'.'.join(parts[1:])}"
                    # Ensure value is JSON-serializable
                    json_value = json.dumps(prepare_for_storage(value))
                    set_parts.append(
                        f"{col_name} = JSON_SET(COALESCE({col_name}, '{{}}'), %s, CAST(%s AS JSON))"
                    )
                    params.extend([json_path, json_value])
                else:
                    # Standard field update
                    sql_field = quote_identifier(field)
                    prepared_value = prepare_for_storage(value)
                    # If value is dict/list, assume JSON column and serialize
                    if isinstance(prepared_value, (dict, list)):
                        set_parts.append(f"{sql_field} = CAST(%s AS JSON)")
                        params.append(json.dumps(prepared_value))
                    else:
                        set_parts.append(f"{sql_field} = %s")
                        params.append(prepared_value)

        # $unset: Set field to NULL or appropriate default
        if "$unset" in update_payload:
            for field in update_payload["$unset"]:
                if "." in field:
                    # Nested field removal using JSON_REMOVE
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path = f"$.{'.'.join(parts[1:])}"
                    set_parts.append(
                        f"{col_name} = JSON_REMOVE(COALESCE({col_name}, '{{}}'), %s)"
                    )
                    params.append(json_path)
                else:
                    # Standard field unset (set to NULL)
                    sql_field = quote_identifier(field)
                    set_parts.append(f"{sql_field} = NULL")

        # $push: Append to JSON array
        if "$push" in update_payload:
            for field, value in update_payload["$push"].items():
                json_value = json.dumps(prepare_for_storage(value))
                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    json_path = f"$.{'.'.join(parts[1:])}"
                    set_parts.append(
                        f"{col_name} = JSON_ARRAY_APPEND(COALESCE({col_name}, '{{}}'), %s, CAST(%s AS JSON))"
                    )
                    params.extend([json_path, json_value])
                else:
                    sql_field = quote_identifier(field)
                    set_parts.append(
                        f"{sql_field} = JSON_ARRAY_APPEND(COALESCE({sql_field}, '[]'), '$', CAST(%s AS JSON))"
                    )
                    params.append(json_value)

        # $pull: Remove item from JSON array (removes all occurrences)
        if "$pull" in update_payload:
            for field, value in update_payload["$pull"].items():
                # Find the path(s) to the value and remove them. This is complex.
                # Simple approach: Remove based on value match if MySQL supports it well.
                # JSON_REMOVE needs a path. We need JSON_SEARCH to find the path.
                json_value_search = prepare_for_storage(value)  # Value to search for

                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    base_path = f"$.{'.'.join(parts[1:])}"  # Path to the array itself
                    # This gets complicated quickly. A simpler (but maybe less efficient) approach might be
                    # to reconstruct the array excluding the value.
                    # Using JSON_TABLE might work on newer MySQL.
                    # Let's stick to a known function, even if limited: Find *first* path and remove.
                    set_parts.append(
                        f"""
                        {col_name} = JSON_REMOVE(
                            COALESCE({col_name}, '{{}}'),
                            JSON_UNQUOTE(JSON_SEARCH(COALESCE({col_name}, '{{}}'), 'one', %s, NULL, %s))
                        )
                    """
                    )
                    params.extend(
                        [json_value_search, base_path + "[*]"]
                    )  # Search within the array path
                else:
                    sql_field = quote_identifier(field)
                    # Find path within the top-level array
                    set_parts.append(
                        f"""
                        {sql_field} = JSON_REMOVE(
                            COALESCE({sql_field}, '[]'),
                            JSON_UNQUOTE(JSON_SEARCH(COALESCE({sql_field}, '[]'), 'one', %s, NULL, '$[*]'))
                        )
                    """
                    )
                    params.append(json_value_search)  # Search in the root array

        # $pop: Remove first (-1) or last (1) element from JSON array
        if "$pop" in update_payload:
            for field, direction in update_payload["$pop"].items():
                path_to_remove = (
                    "'$[0]'" if direction == -1 else "'$[last]'"
                )  # Requires MySQL 8.0.2+ for last
                if direction not in (-1, 1):
                    raise ValueError("$pop direction must be 1 (last) or -1 (first)")

                if "." in field:
                    parts = field.split(".")
                    col_name = quote_identifier(parts[0])
                    base_path = f"$.{'.'.join(parts[1:])}"  # Path to the array
                    # Construct the full path for JSON_REMOVE
                    # Note: Concatenating paths for JSON_REMOVE might be tricky. Re-evaluate if needed.
                    # Simpler: Use path directly if '[last]' syntax works as expected.
                    pop_path = f"{base_path}[{ '0' if direction == -1 else 'last' }]"
                    set_parts.append(
                        f"{col_name} = JSON_REMOVE(COALESCE({col_name}, '{{}}'), %s)"
                    )
                    params.append(pop_path)
                else:
                    sql_field = quote_identifier(field)
                    pop_path = f"$[{ '0' if direction == -1 else 'last' }]"
                    set_parts.append(
                        f"{sql_field} = JSON_REMOVE(COALESCE({sql_field}, '[]'), %s)"
                    )
                    params.append(pop_path)

        return ", ".join(set_parts), params

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Converts entity to dict, preparing complex types for MySQL."""
        # Get the raw dictionary representation
        if hasattr(entity, "model_dump"):
            # Pydantic v2+
            try:
                # Prefer JSON mode for compatibility if available
                entity_dict = entity.model_dump(mode="json", by_alias=True)
            except Exception:
                entity_dict = entity.model_dump(by_alias=True)
        elif hasattr(entity, "dict"):
            # Pydantic v1
            entity_dict = entity.dict(by_alias=True)
        elif hasattr(entity, "__dict__"):
            # Standard class instance
            entity_dict = dict(entity.__dict__)
        else:
            raise TypeError(f"Cannot convert entity of type {type(entity)} to dict")

        # Prepare for storage (handles nested models, dates, etc.)
        prepared_dict = prepare_for_storage(entity_dict)

        # Serialize complex types (dict, list) likely stored as JSON in MySQL
        final_dict = {}
        for key, value in prepared_dict.items():
            if isinstance(value, (dict, list)):
                # Check if the target column is likely JSON or TEXT
                # (We guess based on type - ideally schema info would be better)
                final_dict[key] = json.dumps(value)
            else:
                final_dict[key] = value

        # Handle ID mapping (ensure db_id exists if needed)
        if self._db_id_field not in final_dict and self._app_id_field in final_dict:
            final_dict[self._db_id_field] = final_dict[self._app_id_field]
        # elif self._app_id_field not in final_dict and self._db_id_field in final_dict:
        #      final_dict[self._app_id_field] = final_dict[self._db_id_field]

        return final_dict

    def _row_to_entity(self, row: Dict[str, Any]) -> T:
        """Converts a MySQL row (dict) to an entity instance."""
        entity_data = row.copy()

        # Deserialize JSON strings back into Python objects
        for key, value in entity_data.items():
            if isinstance(value, str):
                try:
                    # Attempt to parse strings that look like JSON objects or arrays
                    if (value.startswith("{") and value.endswith("}")) or (
                        value.startswith("[") and value.endswith("]")
                    ):
                        entity_data[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass

        # Handle ID mapping (ensure app_id exists)
        if self._app_id_field not in entity_data and self._db_id_field in entity_data:
            entity_data[self._app_id_field] = entity_data[self._db_id_field]

        # Remove db_id_field if it's different from app_id_field, as it's usually
        # not part of the core entity model exposed by the repository.
        if self._db_id_field != self._app_id_field and self._db_id_field in entity_data:
            del entity_data[self._db_id_field]

        # Instantiate the entity class
        try:
            return self._entity_cls(**entity_data)
        except TypeError as e:
            # Provide more context on validation errors
            base_logger.error(
                f"Error instantiating {self._entity_cls.__name__} with data: {entity_data}. Error: {e}"
            )
            # You might want to filter entity_data based on model fields here
            # Example (if using Pydantic):
            # model_fields = self._entity_cls.model_fields.keys()
            # filtered_data = {k: v for k, v in entity_data.items() if k in model_fields}
            # return self._entity_cls(**filtered_data)
            raise ValueError(
                f"Mismatch between row data and entity schema {self._entity_cls.__name__}: {e}"
            ) from e
