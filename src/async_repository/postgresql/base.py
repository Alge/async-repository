import json
from logging import LoggerAdapter
from typing import Any, AsyncGenerator, Dict, Generic, List, Optional, Tuple, Type, TypeVar, cast

from asyncpg import Pool, Record
from asyncpg.exceptions import UniqueViolationError

from repositories.base.interfaces import Repository
from repositories.base.query import QueryOptions
from repositories.base.exceptions import ObjectNotFoundException, KeyAlreadyExistsException
from repositories.base.utils import prepare_for_storage  # <-- Import the helper

T = TypeVar('T')


def quote_identifier(identifier: str) -> str:
    safe_identifier = identifier.replace('"', '""')
    return f'"{safe_identifier}"'


class PostgreSQLRepository(Repository[T], Generic[T]):
    """
    Base PostgreSQL repository implementation.
    This version supports DSL criteria for updates/deletes and does not handle any timestamp fields.
    """

    def __init__(
            self,
            pool: Pool,
            table_name: str,
            entity_cls: Type[T],
            app_id_field: str = "id",
            db_id_field: str = "db_id",
    ):
        self._pool = pool
        self._table_name = table_name
        self._entity_cls = entity_cls
        self._app_id_field = app_id_field
        self._db_id_field = db_id_field

    @property
    def entity_type(self) -> Type[T]:
        return self._entity_cls

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    async def initialize(self) -> None:
        pass

    async def get(
            self,
            id: str,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            use_db_id: bool = False
    ) -> T:
        field = self._db_id_field if use_db_id else self._app_id_field
        logger.debug(f"Getting {self._entity_cls.__name__} with {field}: {id}")
        query = (
            f"SELECT * FROM {quote_identifier(self._table_name)} "
            f"WHERE {quote_identifier(field)} = $1"
        )
        params = [id]
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
        if not row:
            id_type = "database" if use_db_id else "application"
            raise ObjectNotFoundException(
                f"{self._entity_cls.__name__} with {id_type} ID {id} not found"
            )
        return self._row_to_entity(row)

    async def get_by_db_id(
            self,
            db_id: Any,
            logger: LoggerAdapter,
            timeout: Optional[float] = None
    ) -> T:
        return await self.get(db_id, logger, timeout, use_db_id=True)

    async def store(
            self,
            entity: T,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            generate_app_id: bool = True,  # still accepted for backwards compatibility
            return_value: bool = False
    ) -> Optional[T]:
        self.validate_entity(entity)
        entity_dict = self._entity_to_dict(entity)
        # Convert special types to storage-compatible formats
        entity_dict = prepare_for_storage(entity_dict)
        # Remove id fields if they are None so the database default can be applied.
        if self._app_id_field in entity_dict and entity_dict[self._app_id_field] is None:
            del entity_dict[self._app_id_field]
        if self._db_id_field in entity_dict and entity_dict[self._db_id_field] is None:
            del entity_dict[self._db_id_field]
        # Timestamps removed.
        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = [f"${i + 1}" for i in range(len(fields))]
        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} "
            f"({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)})"
        )
        try:
            async with self._pool.acquire() as conn:
                if return_value:
                    query_returning = query + " RETURNING *"
                    row = await conn.fetchrow(query_returning, *values)
                else:
                    await conn.execute(query, *values)
        except UniqueViolationError as e:
            if self._app_id_field in str(e):
                field_name = self._app_id_field
                id_value = entity_dict.get(self._app_id_field)
            elif self._db_id_field in str(e):
                field_name = self._db_id_field
                id_value = entity_dict.get(self._db_id_field)
            else:
                field_name = "unknown"
                id_value = "unknown"
            raise KeyAlreadyExistsException(
                f"{self._entity_cls.__name__} with {field_name} {id_value} already exists"
            )
        if return_value:
            return self._row_to_entity(row) if row else None
        else:
            return None

    async def upsert(
            self,
            entity: T,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            generate_app_id: bool = True  # still accepted for backwards compatibility
    ) -> None:
        self.validate_entity(entity)
        entity_dict = self._entity_to_dict(entity)
        # Convert special types to storage-compatible formats
        entity_dict = prepare_for_storage(entity_dict)
        # Remove id fields if they are None so that the database default can be applied.
        if self._app_id_field in entity_dict and entity_dict[self._app_id_field] is None:
            del entity_dict[self._app_id_field]
        if self._db_id_field in entity_dict and entity_dict[self._db_id_field] is None:
            del entity_dict[self._db_id_field]
        # Timestamps removed.
        fields = list(entity_dict.keys())
        values = list(entity_dict.values())
        placeholders = [f"${i + 1}" for i in range(len(fields))]
        logger.debug(f"Upserting {self._entity_cls.__name__}: {entity_dict.get(self._app_id_field)}")
        query = (
            f"INSERT INTO {quote_identifier(self._table_name)} "
            f"({', '.join(quote_identifier(f) for f in fields)}) "
            f"VALUES ({', '.join(placeholders)}) "
            f"ON CONFLICT ({quote_identifier(self._app_id_field)}) DO UPDATE "
            f"SET ({', '.join(quote_identifier(f) for f in fields)}) = ({', '.join(placeholders)})"
        )
        async with self._pool.acquire() as conn:
            await conn.execute(query, *values)


    async def update_one(
            self,
            options: QueryOptions,
            update: "Update",
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            return_value: bool = False
    ) -> Optional[T]:
        """
        Update specific fields of a single entity matching the provided QueryOptions.
        Uses a CTE to limit the update to one record.
        """
        logger.debug(
            f"Updating one {self._entity_cls.__name__} with options: {options.expression}, update: {update.build()}")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")
        where_clause, params, _ = self._transform_expression(options.expression, 1)
        # Build SET clause from update payload.
        update_payload = update.build()
        # Apply the helper to update payload to convert special types (like URL types) to strings.
        update_payload = prepare_for_storage(update_payload)
        set_clause, set_params = self._build_set_clause(update_payload, len(params) + 1)
        cte = f"WITH cte AS (SELECT {quote_identifier(self._app_id_field)} FROM {quote_identifier(self._table_name)} WHERE {where_clause} LIMIT 1)"
        update_query = (
            f"{cte} UPDATE {quote_identifier(self._table_name)} SET {set_clause} "
            f"FROM cte WHERE {quote_identifier(self._table_name)}.{quote_identifier(self._app_id_field)} = cte.{quote_identifier(self._app_id_field)}"
        )
        if return_value:
            update_query += " RETURNING *"
        async with self._pool.acquire() as conn:
            if return_value:
                row = await conn.fetchrow(update_query, *(params + set_params))
                if not row:
                    raise ObjectNotFoundException(
                        f"{self._entity_cls.__name__} not found with criteria {options.expression}")
                return self._row_to_entity(row)
            else:
                result = await conn.execute(update_query, *(params + set_params))
                if result == "UPDATE 0":
                    raise ObjectNotFoundException(
                        f"{self._entity_cls.__name__} not found with criteria {options.expression}")
                return None

    async def update_many(
            self,
            options: QueryOptions,
            update: "Update",
            logger: LoggerAdapter,
            timeout: Optional[float] = None
    ) -> int:
        """
        Update specific fields of all entities matching the provided QueryOptions.
        If limit/offset/sort are provided, updates only the subset of IDs.
        Returns the number of entities updated.
        """
        logger.debug(
            f"Updating many {self._entity_cls.__name__} with options: {options.expression}, update: {update.build()}")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        # If limit/offset/sort/random_order are provided, retrieve specific IDs first.
        if options.limit > 0 or options.offset > 0 or options.sort_by or options.random_order:
            where_clause, params, _ = self._transform_expression(options.expression, 1)
            order_clause = ""
            if options.random_order:
                order_clause = " ORDER BY RANDOM()"
            elif options.sort_by:
                sort_field = options.sort_by
                if sort_field == "id" and self.app_id_field != "id":
                    sort_field = self.app_id_field
                elif sort_field == "db_id" and self.db_id_field != "db_id":
                    sort_field = self.db_id_field
                direction = "DESC" if options.sort_desc else "ASC"
                order_clause = f" ORDER BY {quote_identifier(sort_field)} {direction}"
            limit_clause = f" LIMIT {options.limit}" if options.limit > 0 else ""
            offset_clause = f" OFFSET {options.offset}" if options.offset > 0 else ""
            id_query = (
                f"SELECT {quote_identifier(self.db_id_field)} FROM {quote_identifier(self._table_name)} "
                f"WHERE {where_clause}{order_clause}{limit_clause}{offset_clause}"
            )
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(id_query, *params)
            if not rows:
                return 0
            ids = [row[self.db_id_field] for row in rows]
            placeholders = [f"${i}" for i in range(1, len(ids) + 1)]
            where_clause = f"{quote_identifier(self.db_id_field)} IN ({', '.join(placeholders)})"
            params = ids
        else:
            where_clause, params, _ = self._transform_expression(options.expression, 1)

        update_payload = update.build()
        # Apply the helper method to update payload.
        update_payload = prepare_for_storage(update_payload)
        set_clause, set_params = self._build_set_clause(update_payload, len(params) + 1)
        query = (
            f"UPDATE {quote_identifier(self._table_name)} SET {set_clause} "
            f"WHERE {where_clause}"
        )
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *(params + set_params))
            parts = result.split()
            if len(parts) == 2 and parts[0] == "UPDATE":
                return int(parts[1])
            return 0

    async def delete_many(
            self,
            options: QueryOptions,
            logger: LoggerAdapter,
            timeout: Optional[float] = None
    ) -> int:
        """
        Delete entities matching the provided QueryOptions.
        If a limit/offset/sort is provided, deletes only that subset.
        Returns the count of deleted entities.
        """
        logger.debug(f"Deleting many {self._entity_cls.__name__} with options: {options.expression}")
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete.")

        # If limit, offset, sort, or random_order is provided, select specific IDs.
        if options.limit > 0 or options.offset > 0 or options.sort_by or options.random_order:
            where_clause, params, _ = self._transform_expression(options.expression, 1)
            order_clause = ""
            if options.random_order:
                order_clause = " ORDER BY RANDOM()"
            elif options.sort_by:
                sort_field = options.sort_by
                if sort_field == "id" and self.app_id_field != "id":
                    sort_field = self.app_id_field
                elif sort_field == "db_id" and self.db_id_field != "db_id":
                    sort_field = self.db_id_field
                direction = "DESC" if options.sort_desc else "ASC"
                order_clause = f" ORDER BY {quote_identifier(sort_field)} {direction}"
            limit_clause = f" LIMIT {options.limit}" if options.limit > 0 else ""
            offset_clause = f" OFFSET {options.offset}" if options.offset > 0 else ""
            id_query = (
                f"SELECT {quote_identifier(self.db_id_field)} FROM {quote_identifier(self._table_name)} "
                f"WHERE {where_clause}{order_clause}{limit_clause}{offset_clause}"
            )
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(id_query, *params)
            if not rows:
                return 0
            ids = [row[self.db_id_field] for row in rows]
            placeholders = [f"${i}" for i in range(1, len(ids) + 1)]
            where_clause = f"{quote_identifier(self.db_id_field)} IN ({', '.join(placeholders)})"
            params = ids
        else:
            where_clause, params, _ = self._transform_expression(options.expression, 1)

        query = f"DELETE FROM {quote_identifier(self._table_name)} WHERE {where_clause}"
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *params)
            parts = result.split()
            if len(parts) == 2 and parts[0] == "DELETE":
                return int(parts[1])
            return 0

    async def delete_one(
            self,
            id: str,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            use_db_id: bool = False
    ) -> None:
        """
        Delete a single entity from the repository by reusing delete_many with a limit of 1.
        """
        field = self._db_id_field if use_db_id else self._app_id_field
        filter_opts = QueryOptions(
            expression={field: {"operator": "eq", "value": id}},
            limit=1,
            offset=0
        )
        count_deleted = await self.delete_many(filter_opts, logger, timeout)
        if count_deleted == 0:
            raise ObjectNotFoundException(f"{self._entity_cls.__name__} with ID {id} not found")
        elif count_deleted > 1:
            logger.warning(f"delete_one: More than one document deleted for ID {id}")

    async def list(
            self,
            logger: LoggerAdapter,
            options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        options = options or QueryOptions()
        logger.debug(f"Listing {self._entity_cls.__name__} with options: {options.__dict__}")
        query, params = self._build_query(options)
        logger.debug(f"SQL query: {query}")
        logger.debug(f"SQL params: {params}")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            for row in rows:
                yield self._row_to_entity(row)

    async def count(
            self,
            logger: LoggerAdapter,
            options: Optional[QueryOptions] = None
    ) -> int:
        options = options or QueryOptions()
        logger.debug(f"Counting {self._entity_cls.__name__} with options: {options.__dict__}")
        if options.expression:
            base_query, params, _ = self._transform_expression(options.expression, 1)
        else:
            base_query, params = self._build_base_query(options)
        count_query = f"SELECT COUNT(*) FROM {quote_identifier(self._table_name)}"
        if base_query:
            count_query += f" WHERE {base_query}"
        async with self._pool.acquire() as conn:
            result = await conn.fetchval(count_query, *params)
        return cast(int, result)

    def _build_query(self, options: QueryOptions) -> Tuple[str, List[Any]]:
        if options.expression:
            where_clause, params, _ = self._transform_expression(options.expression, 1)
        else:
            where_clause, params = self._build_base_query(options)
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
            query += f" ORDER BY {quote_identifier(self._app_id_field)} ASC"
        query += f" LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}"
        params.append(options.limit)
        params.append(options.offset)
        return query, params

    def _build_base_query(self, options: QueryOptions) -> Tuple[str, List[Any]]:
        return "", []

    def _transform_expression(self, expr: Dict[str, Any], start_index: int = 1) -> Tuple[str, List[Any], int]:
        clauses = []
        params = []
        current_index = start_index
        if "and" in expr:
            sub_clauses = []
            for sub_expr in expr["and"]:
                clause, sub_params, current_index = self._transform_expression(sub_expr, current_index)
                sub_clauses.append(f"({clause})")
                params.extend(sub_params)
            return " AND ".join(sub_clauses), params, current_index
        if "or" in expr:
            sub_clauses = []
            for sub_expr in expr["or"]:
                clause, sub_params, current_index = self._transform_expression(sub_expr, current_index)
                sub_clauses.append(f"({clause})")
                params.extend(sub_params)
            return " OR ".join(sub_clauses), params, current_index
        for field, condition in expr.items():
            # Check if it's a nested field (contains dots)
            if "." in field:
                main_field, sub_field = field.split(".", 1)
                safe_main_field = quote_identifier(main_field)

                # Transform dot notation to PostgreSQL JSONB path format (with commas)
                jsonb_path = sub_field.replace(".", ",")

                # Get the value to compare against
                if isinstance(condition, dict):
                    operator = condition.get("operator", "eq")
                    value = condition.get("value")
                else:
                    operator = "eq"
                    value = condition

                # Convert special types (like Pydantic URL)
                value = prepare_for_storage(value)

                # Handle different operators for JSONB nested fields
                if operator == "eq":
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' = ${current_index}"
                    params.append(str(value))
                    current_index += 1
                elif operator == "ne":
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' != ${current_index}"
                    params.append(str(value))
                    current_index += 1
                elif operator == "gt":
                    clause = f"({safe_main_field}::jsonb #>> '{{{jsonb_path}}}')::numeric > ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator in ("ge", "gte"):
                    clause = f"({safe_main_field}::jsonb #>> '{{{jsonb_path}}}')::numeric >= ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "lt":
                    clause = f"({safe_main_field}::jsonb #>> '{{{jsonb_path}}}')::numeric < ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator in ("le", "lte"):
                    clause = f"({safe_main_field}::jsonb #>> '{{{jsonb_path}}}')::numeric <= ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "in":
                    if not value:
                        clause = "FALSE"
                    else:
                        placeholders = [f"${current_index + i}" for i in range(len(value))]
                        clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' IN ({', '.join(placeholders)})"
                        params.extend([str(v) for v in value])
                        current_index += len(value)
                elif operator == "nin":
                    if not value:
                        clause = "TRUE"
                    else:
                        placeholders = [f"${current_index + i}" for i in range(len(value))]
                        clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' NOT IN ({', '.join(placeholders)})"
                        params.extend([str(v) for v in value])
                        current_index += len(value)
                elif operator == "contains":
                    # For contains on nested JSONB arrays
                    jsonb_parent_path = jsonb_path.rsplit(",", 1)[0] if "," in jsonb_path else ""
                    if jsonb_parent_path:
                        clause = f"{safe_main_field}::jsonb #> '{{{jsonb_parent_path}}}' @> ${current_index}::jsonb"
                    else:
                        clause = f"{safe_main_field}::jsonb @> ${current_index}::jsonb"
                    params.append(json.dumps({jsonb_path.split(",")[-1]: value}))
                    current_index += 1
                elif operator == "startswith":
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' ILIKE ${current_index}"
                    params.append(f"{value}%")
                    current_index += 1
                elif operator == "endswith":
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' ILIKE ${current_index}"
                    params.append(f"%{value}")
                    current_index += 1
                elif operator == "like":
                    if "%" not in value:
                        pattern = f"%{value}%"
                    else:
                        pattern = value
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' ILIKE ${current_index}"
                    params.append(pattern)
                    current_index += 1
                elif operator == "exists":
                    if value:
                        # Check if the JSON path exists
                        clause = f"jsonb_path_exists({safe_main_field}::jsonb, '$.\"{jsonb_path.replace(',', '\".\\"')}\"')"
                    else:
                        # Check if the JSON path does not exist
                        clause = f"NOT jsonb_path_exists({safe_main_field}::jsonb, '$.\"{jsonb_path.replace(',', '\".\\"')}\"')"
                elif operator == "regex":
                    clause = f"{safe_main_field}::jsonb #>> '{{{jsonb_path}}}' ~ ${current_index}"
                    params.append(value)
                    current_index += 1
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            else:
                # Non-nested field handling (existing code)
                entity_value = condition
                if isinstance(condition, dict):
                    operator = condition.get("operator", "eq")
                    value = condition.get("value")
                else:
                    operator = "eq"
                    value = condition
                safe_field = quote_identifier(field)
                if operator == "eq":
                    clause = f"{safe_field} = ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "ne":
                    clause = f"{safe_field} != ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "gt":
                    clause = f"{safe_field} > ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator in ("ge", "gte"):
                    clause = f"{safe_field} >= ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "lt":
                    clause = f"{safe_field} < ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator in ("le", "lte"):
                    clause = f"{safe_field} <= ${current_index}"
                    params.append(value)
                    current_index += 1
                elif operator == "in":
                    if not value:
                        clause = "FALSE"
                    else:
                        placeholders = [f"${current_index + i}" for i in range(len(value))]
                        clause = f"{safe_field} IN ({', '.join(placeholders)})"
                        params.extend(value)
                        current_index += len(value)
                elif operator == "nin":
                    placeholders = [f"${current_index + i}" for i in range(len(value))]
                    clause = f"{safe_field} NOT IN ({', '.join(placeholders)})"
                    params.extend(value)
                    current_index += len(value)
                elif operator == "contains":
                    clause = f"${current_index} = ANY({safe_field})"
                    params.append(value)
                    current_index += 1
                elif operator == "startswith":
                    clause = f"{safe_field} ILIKE ${current_index}"
                    params.append(f"{value}%")
                    current_index += 1
                elif operator == "endswith":
                    clause = f"{safe_field} ILIKE ${current_index}"
                    params.append(f"%{value}")
                    current_index += 1
                elif operator == "like":
                    if "%" not in value:
                        pattern = f"%{value}%"
                    else:
                        pattern = value
                    clause = f"{safe_field} ILIKE ${current_index}"
                    params.append(pattern)
                    current_index += 1
                elif operator == "exists":
                    clause = f"{safe_field} IS {'NOT ' if value else ''}NULL"
                elif operator == "regex":
                    clause = f"{safe_field} ~ ${current_index}"
                    params.append(value)
                    current_index += 1
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            clauses.append(clause)
        return " AND ".join(clauses), params, current_index

    def _build_set_clause(self, update_payload: Dict[str, Any], start_index: int) -> Tuple[str, List[Any]]:
        """
        Build the SET clause for SQL UPDATE from the MongoDB-style update payload.
        """
        set_parts = []
        params = []
        current_index = start_index

        # Handle $set operation (direct field updates)
        if "$set" in update_payload:
            for field, value in update_payload["$set"].items():
                if "." in field:
                    # Extract main field and JSON path
                    main_field, sub_field = field.split(".", 1)
                    safe_main_field = quote_identifier(main_field)

                    # Apply prepare_for_storage to convert special types
                    value = prepare_for_storage(value)

                    # Serialize the value to JSON
                    json_value = json.dumps(value)

                    # Handle multi-level paths by splitting properly
                    path_parts = sub_field.split(".")
                    array_path = ", ".join([f"'{part}'" for part in path_parts])

                    set_parts.append(f"""
                        {safe_main_field} = jsonb_set(
                            COALESCE({safe_main_field}::jsonb, '{{}}'::jsonb),
                            ARRAY[{array_path}],
                            ${current_index}::jsonb,
                            true
                        )
                    """)
                    params.append(json_value)
                    current_index += 1
                else:
                    safe_field = quote_identifier(field)
                    # Convert the value if it's a dictionary
                    if isinstance(value, dict):
                        value = prepare_for_storage(value)
                        value = json.dumps(value)
                    else:
                        value = prepare_for_storage(value)
                    set_parts.append(f"{safe_field} = ${current_index}")
                    params.append(value)
                    current_index += 1

        # Handle $unset operation (set fields to appropriate default values)
        if "$unset" in update_payload:
            for field in update_payload["$unset"]:
                if "." in field:
                    # Extract main field and JSON path for nested fields
                    main_field, sub_field = field.split(".", 1)
                    safe_main_field = quote_identifier(main_field)

                    # Handle multi-level paths by converting dots to PostgreSQL JSON path format
                    path_parts = sub_field.split(".")
                    array_path = ", ".join([f"'{part}'" for part in path_parts])

                    # Use jsonb_set with the #- operator to remove the specified path
                    set_parts.append(f"""
                        {safe_main_field} = ({safe_main_field}::jsonb #- ARRAY[{array_path}])
                    """)
                else:
                    safe_field = quote_identifier(field)
                    # Generic field type handling based on name patterns
                    if field.endswith('_at') or field in ('created_at', 'updated_at'):
                        set_parts.append(f"{safe_field} = CURRENT_TIMESTAMP")
                    elif field in ('tags',) or field.endswith('_tags') or field.endswith('_list') or field.endswith(
                            '_array'):
                        set_parts.append(f"{safe_field} = '{{}}'::jsonb")
                    elif field in ('metadata', 'profile') or field.endswith('_data') or field.endswith('_info'):
                        set_parts.append(f"{safe_field} = '{{}}'::jsonb")
                    elif field in ('active',) or field.startswith('is_') or field.endswith('_active'):
                        set_parts.append(f"{safe_field} = false")
                    elif field.endswith('_count') or field == 'value' or field.endswith('_value'):
                        set_parts.append(f"{safe_field} = 0")
                    else:
                        set_parts.append(f"{safe_field} = ''")

        # Handle array operations: $push, $pull, $pop
        if "$push" in update_payload:
            for field, value in update_payload["$push"].items():
                # Apply prepare_for_storage to handle special types
                value = prepare_for_storage(value)

                # Serialize the value if it's a dictionary or list
                if isinstance(value, (dict, list)):
                    json_value = json.dumps(value)
                else:
                    json_value = json.dumps(value)

                if "." in field:
                    # Handle nested array pushes
                    main_field, sub_field = field.split(".", 1)
                    safe_main_field = quote_identifier(main_field)

                    # Handle multi-level paths
                    path_parts = sub_field.split(".")
                    array_path = ", ".join([f"'{part}'" for part in path_parts])

                    set_parts.append(f"""
                        {safe_main_field} = jsonb_set(
                            COALESCE({safe_main_field}::jsonb, '{{}}'::jsonb),
                            ARRAY[{array_path}],
                            CASE
                                WHEN jsonb_typeof(
                                    jsonb_extract_path_text(
                                        {safe_main_field}::jsonb, 
                                        {", ".join([f"'{p}'" for p in path_parts])}
                                    )::jsonb
                                ) = 'array'
                                THEN (
                                    jsonb_extract_path_text(
                                        {safe_main_field}::jsonb, 
                                        {", ".join([f"'{p}'" for p in path_parts])}
                                    )::jsonb || ${current_index}::jsonb
                                )
                                ELSE jsonb_build_array(${current_index}::jsonb)
                            END,
                            true
                        )
                    """)
                    params.append(json_value)
                    current_index += 1
                else:
                    safe_field = quote_identifier(field)
                    set_parts.append(
                        f"{safe_field} = CASE WHEN {safe_field} IS NULL THEN ARRAY[${current_index}] ELSE array_append({safe_field}, ${current_index}) END")
                    params.append(value)
                    current_index += 1

        if "$pull" in update_payload:
            for field, value in update_payload["$pull"].items():
                # Apply prepare_for_storage to handle special types
                value = prepare_for_storage(value)

                # Serialize the value if it's a dictionary or list
                if isinstance(value, (dict, list)):
                    json_value = json.dumps(value)
                else:
                    json_value = json.dumps(value)

                if "." in field:
                    # Handle nested array pulls
                    main_field, sub_field = field.split(".", 1)
                    safe_main_field = quote_identifier(main_field)

                    # Handle multi-level paths
                    path_parts = sub_field.split(".")
                    array_path = ", ".join([f"'{part}'" for part in path_parts])

                    set_parts.append(f"""
                        {safe_main_field} = jsonb_set(
                            COALESCE({safe_main_field}::jsonb, '{{}}'::jsonb),
                            ARRAY[{array_path}],
                            (
                                SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
                                FROM jsonb_array_elements(
                                    COALESCE(
                                        jsonb_extract_path(
                                            {safe_main_field}::jsonb, 
                                            {", ".join([f"'{p}'" for p in path_parts])}
                                        ), 
                                        '[]'::jsonb
                                    )
                                ) AS elem
                                WHERE elem::text != ${current_index}::text
                            ),
                            true
                        )
                    """)
                    params.append(json_value)
                    current_index += 1
                else:
                    safe_field = quote_identifier(field)
                    set_parts.append(f"{safe_field} = array_remove({safe_field}, ${current_index})")
                    params.append(value)
                    current_index += 1

        if "$pop" in update_payload:
            for field, direction in update_payload["$pop"].items():
                if "." in field:
                    # Handle nested array pops
                    main_field, sub_field = field.split(".", 1)
                    safe_main_field = quote_identifier(main_field)

                    # Handle multi-level paths
                    path_parts = sub_field.split(".")
                    array_path = ", ".join([f"'{part}'" for part in path_parts])

                    if direction == 1:  # Pop last element
                        set_parts.append(f"""
                            {safe_main_field} = jsonb_set(
                                COALESCE({safe_main_field}::jsonb, '{{}}'::jsonb),
                                ARRAY[{array_path}],
                                (
                                    SELECT CASE
                                        WHEN jsonb_array_length(
                                            COALESCE(
                                                jsonb_extract_path(
                                                    {safe_main_field}::jsonb, 
                                                    {", ".join([f"'{p}'" for p in path_parts])}
                                                ), 
                                                '[]'::jsonb
                                            )
                                        ) > 0
                                        THEN (
                                            SELECT jsonb_agg(value)
                                            FROM jsonb_array_elements(
                                                COALESCE(
                                                    jsonb_extract_path(
                                                        {safe_main_field}::jsonb, 
                                                        {", ".join([f"'{p}'" for p in path_parts])}
                                                    ), 
                                                    '[]'::jsonb
                                                )
                                            ) WITH ORDINALITY
                                            WHERE ordinality < jsonb_array_length(
                                                COALESCE(
                                                    jsonb_extract_path(
                                                        {safe_main_field}::jsonb, 
                                                        {", ".join([f"'{p}'" for p in path_parts])}
                                                    ), 
                                                    '[]'::jsonb
                                                )
                                            )
                                        )
                                        ELSE '[]'::jsonb
                                    END
                                ),
                                true
                            )
                        """)
                    elif direction == -1:  # Pop first element
                        set_parts.append(f"""
                            {safe_main_field} = jsonb_set(
                                COALESCE({safe_main_field}::jsonb, '{{}}'::jsonb),
                                ARRAY[{array_path}],
                                (
                                    SELECT CASE
                                        WHEN jsonb_array_length(
                                            COALESCE(
                                                jsonb_extract_path(
                                                    {safe_main_field}::jsonb, 
                                                    {", ".join([f"'{p}'" for p in path_parts])}
                                                ), 
                                                '[]'::jsonb
                                            )
                                        ) > 0
                                        THEN (
                                            SELECT jsonb_agg(value)
                                            FROM jsonb_array_elements(
                                                COALESCE(
                                                    jsonb_extract_path(
                                                        {safe_main_field}::jsonb, 
                                                        {", ".join([f"'{p}'" for p in path_parts])}
                                                    ), 
                                                    '[]'::jsonb
                                                )
                                            ) WITH ORDINALITY
                                            WHERE ordinality > 1
                                        )
                                        ELSE '[]'::jsonb
                                    END
                                ),
                                true
                            )
                        """)
                else:
                    safe_field = quote_identifier(field)
                    if direction == 1:  # Pop last element
                        set_parts.append(
                            f"{safe_field} = CASE WHEN array_length({safe_field}, 1) > 0 THEN {safe_field}[1:array_length({safe_field}, 1)-1] ELSE {safe_field} END")
                    elif direction == -1:  # Pop first element
                        set_parts.append(
                            f"{safe_field} = CASE WHEN array_length({safe_field}, 1) > 0 THEN {safe_field}[2:array_length({safe_field}, 1)] ELSE {safe_field} END")

        if not set_parts:
            raise ValueError("No valid update operations found in the Update object")

        return ", ".join(set_parts), params

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """
        Convert an entity to a dictionary suitable for PostgreSQL storage.
        """
        # Get the entity as a dictionary
        if hasattr(entity, "model_dump"):
            entity_dict = entity.model_dump()
        else:
            entity_dict = dict(entity.__dict__)

        # Handle ID fields
        if self._app_id_field in entity_dict and self._db_id_field not in entity_dict:
            entity_dict[self._db_id_field] = entity_dict[self._app_id_field]
        elif self._db_id_field in entity_dict and self._app_id_field not in entity_dict:
            entity_dict[self._app_id_field] = entity_dict[self._db_id_field]

        # Convert special types to storage-compatible formats
        entity_dict = prepare_for_storage(entity_dict)

        # Handle fields based on their types, not based on field names
        for key, value in list(entity_dict.items()):
            # For dictionary fields, convert to JSON string
            if isinstance(value, dict):
                entity_dict[key] = json.dumps(value)
            # For list fields, if they're homogeneous primitive types,
            # keep them as lists for PostgreSQL arrays
            elif isinstance(value, list):
                if len(value) == 0 or (all(isinstance(item, (str, int, float, bool)) for item in value) and
                                       all(isinstance(item, type(value[0])) for item in value)):
                    # Keep as a list for PostgreSQL array type
                    pass
                else:
                    # Convert to JSON for heterogeneous or complex lists
                    entity_dict[key] = json.dumps(value)

        return entity_dict

    def _row_to_entity(self, row: Record) -> T:
        """
        Convert a database row to an entity instance.
        """
        row_dict = dict(row)

        # Handle ID fields
        if self._db_id_field in row_dict:
            if row_dict.get(self._app_id_field) is None:
                row_dict[self._app_id_field] = row_dict[self._db_id_field]
            if self._app_id_field != self._db_id_field:
                del row_dict[self._db_id_field]

        # Parse any JSON strings to dictionaries/lists
        for key, value in row_dict.items():
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, (dict, list)):
                        row_dict[key] = parsed_value
                except (json.JSONDecodeError, ValueError):
                    pass

        return self._entity_cls(**row_dict)