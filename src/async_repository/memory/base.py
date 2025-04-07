import asyncio
import copy
import random
from logging import LoggerAdapter
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    get_origin,
    Union,
    get_args,
)

from async_repository.base.interfaces import Repository
from async_repository.base.exceptions import (
    ObjectNotFoundException,
    KeyAlreadyExistsException,
)
from async_repository.base.query import QueryOptions
from async_repository.base.update import Update  # custom update class

T = TypeVar("T")

import uuid


def _get_nested_value(entity_dict, nested_field):
    """
    Get a value from a nested field using dot notation.
    """
    parts = nested_field.split(".")
    curr = entity_dict
    for part in parts[:-1]:
        if part not in curr or not isinstance(curr[part], dict):
            return None
        curr = curr[part]
    if parts[-1] not in curr:
        return None
    return curr[parts[-1]]


def _set_nested_value(entity_dict, nested_field, value):
    """
    Set a value at a nested field using dot notation.
    """
    parts = nested_field.split(".")
    curr = entity_dict
    for part in parts[:-1]:
        if part not in curr:
            curr[part] = {}
        elif not isinstance(curr[part], dict):
            return False
        curr = curr[part]
    curr[parts[-1]] = value
    return True


def _unset_nested_value(entity_dict, nested_field):
    """
    Remove a value at a nested field using dot notation.
    """
    parts = nested_field.split(".")
    curr = entity_dict
    for part in parts[:-1]:
        if part not in curr or not isinstance(curr[part], dict):
            return False
        curr = curr[part]
    if parts[-1] in curr:
        del curr[parts[-1]]
        return True
    return False


async def update_one(
    self,
    options: QueryOptions,
    update: Update,
    logger: LoggerAdapter,
    timeout: Optional[float] = None,
    return_value: bool = False,
) -> Optional[T]:
    """
    Update specific fields of a single entity matching the provided QueryOptions.
    Returns the updated entity if return_value is True.
    """
    logger.debug(
        f"Updating one {self._entity_cls.__name__} with options: {options}, update: {update}"
    )
    await asyncio.sleep(0)
    if not options.expression:
        raise ValueError("QueryOptions must include an 'expression' for update.")
    filtered = await self._filter_entities(options)
    if not filtered:
        raise ObjectNotFoundException(
            f"{self._entity_cls.__name__} not found with criteria {options.expression}"
        )
    # Get first matching record.
    record = filtered[0]
    db_id = record.get(self._db_id_field) or record.get(self._app_id_field)
    if not db_id:
        raise ObjectNotFoundException(
            f"Could not determine identifier for record matching {options.expression}"
        )
    entity_dict = self._store[db_id]
    # Convert update object to dictionary.
    update_payload = update.build()
    # If the application ID is updated, update the index.
    if "$set" in update_payload and self._app_id_field in update_payload["$set"]:
        new_app_id = update_payload["$set"][self._app_id_field]
        old_app_id = entity_dict.get(self._app_id_field)
        if new_app_id != old_app_id:
            if (
                new_app_id in self._app_id_index
                and self._app_id_index[new_app_id] != db_id
            ):
                raise KeyAlreadyExistsException(
                    f"{self._entity_cls.__name__} with application ID {new_app_id} already exists"
                )
            if old_app_id in self._app_id_index:
                del self._app_id_index[old_app_id]
            self._app_id_index[new_app_id] = db_id
    # Process update operators.
    for op, data in update_payload.items():
        if op == "$set":
            for key, value in data.items():
                if "." in key:
                    _set_nested_value(entity_dict, key, value)
                else:
                    entity_dict[key] = value
        elif op == "$push":
            for key, value in data.items():
                if "." in key:
                    nested_array = _get_nested_value(entity_dict, key)
                    if nested_array is None or not isinstance(nested_array, list):
                        _set_nested_value(entity_dict, key, [value])
                    else:
                        nested_array.append(value)
                else:
                    if key not in entity_dict or not isinstance(entity_dict[key], list):
                        entity_dict[key] = [value]
                    else:
                        entity_dict[key].append(value)
        elif op == "$pop":
            for key, direction in data.items():
                if "." in key:
                    nested_array = _get_nested_value(entity_dict, key)
                    if (
                        nested_array is not None
                        and isinstance(nested_array, list)
                        and nested_array
                    ):
                        if direction == 1:
                            nested_array.pop()
                        elif direction == -1:
                            nested_array.pop(0)
                        else:
                            raise ValueError(
                                "Invalid pop direction; use 1 (pop last) or -1 (pop first)."
                            )
                else:
                    if (
                        key in entity_dict
                        and isinstance(entity_dict[key], list)
                        and entity_dict[key]
                    ):
                        if direction == 1:
                            entity_dict[key].pop()
                        elif direction == -1:
                            entity_dict[key].pop(0)
                        else:
                            raise ValueError(
                                "Invalid pop direction; use 1 (pop last) or -1 (pop first)."
                            )
        elif op == "$unset":
            for key in data.keys():
                if "." in key:
                    _unset_nested_value(entity_dict, key)
                elif key in entity_dict:
                    del entity_dict[key]
        elif op == "$pull":
            for key, value in data.items():
                if "." in key:
                    nested_array = _get_nested_value(entity_dict, key)
                    if nested_array is not None and isinstance(nested_array, list):
                        new_array = [x for x in nested_array if x != value]
                        _set_nested_value(entity_dict, key, new_array)
                elif key in entity_dict and isinstance(entity_dict[key], list):
                    entity_dict[key] = [x for x in entity_dict[key] if x != value]
        else:
            raise ValueError(f"Unsupported operator: {op}")
    if return_value:
        return self._dict_to_entity(entity_dict)
    return None


async def update_many(
    self,
    options: QueryOptions,
    update: Update,
    logger: LoggerAdapter,
    timeout: Optional[float] = None,
) -> int:
    """
    Update specific fields of entities matching the provided QueryOptions,
    respecting limit/offset/sort if provided.
    Returns the number of entities updated.
    """
    logger.debug(
        f"Updating many {self._entity_cls.__name__} with options: {options}, update: {update}"
    )
    await asyncio.sleep(0)
    if not options.expression:
        raise ValueError("QueryOptions must include an 'expression' for update.")
    filtered = await self._filter_entities(options)
    # Apply random ordering if requested; otherwise, apply sort_by if specified.
    if options.random_order:
        random.shuffle(filtered)
    elif options.sort_by:
        sort_field = options.sort_by
        if sort_field == "id" and self._app_id_field != "id":
            sort_field = self._app_id_field
        elif sort_field == self._db_id_field:
            sort_field = self._db_id_field
        filtered.sort(key=lambda x: x.get(sort_field, ""), reverse=options.sort_desc)
    if options.offset > 0:
        filtered = filtered[options.offset :]
    if options.limit > 0:
        filtered = filtered[: options.limit]
    if not filtered:
        return 0

    update_payload = update.build()
    count = 0
    for record in filtered:
        db_id = record.get(self._db_id_field) or record.get(self._app_id_field)
        if not db_id or db_id not in self._store:
            continue
        entity_dict = self._store[db_id]
        if "$set" in update_payload and self._app_id_field in update_payload["$set"]:
            new_app_id = update_payload["$set"][self._app_id_field]
            old_app_id = entity_dict.get(self._app_id_field)
            if new_app_id != old_app_id:
                if (
                    new_app_id in self._app_id_index
                    and self._app_id_index[new_app_id] != db_id
                ):
                    raise KeyAlreadyExistsException(
                        f"{self._entity_cls.__name__} with application ID {new_app_id} already exists"
                    )
                if old_app_id in self._app_id_index:
                    del self._app_id_index[old_app_id]
                self._app_id_index[new_app_id] = db_id
        for op, data in update_payload.items():
            if op == "$set":
                for key, value in data.items():
                    if "." in key:
                        _set_nested_value(entity_dict, key, value)
                    else:
                        entity_dict[key] = value
            elif op == "$push":
                for key, value in data.items():
                    if "." in key:
                        nested_array = _get_nested_value(entity_dict, key)
                        if nested_array is None or not isinstance(nested_array, list):
                            _set_nested_value(entity_dict, key, [value])
                        else:
                            nested_array.append(value)
                    else:
                        if key not in entity_dict or not isinstance(
                            entity_dict[key], list
                        ):
                            entity_dict[key] = [value]
                        else:
                            entity_dict[key].append(value)
            elif op == "$pop":
                for key, direction in data.items():
                    if "." in key:
                        nested_array = _get_nested_value(entity_dict, key)
                        if (
                            nested_array is not None
                            and isinstance(nested_array, list)
                            and nested_array
                        ):
                            if direction == 1:
                                nested_array.pop()
                            elif direction == -1:
                                nested_array.pop(0)
                            else:
                                raise ValueError(
                                    "Invalid pop direction; use 1 (pop last) or -1 (pop first)."
                                )
                    elif (
                        key in entity_dict
                        and isinstance(entity_dict[key], list)
                        and entity_dict[key]
                    ):
                        if direction == 1:
                            entity_dict[key].pop()
                        elif direction == -1:
                            entity_dict[key].pop(0)
                        else:
                            raise ValueError(
                                "Invalid pop direction; use 1 (pop last) or -1 (pop first)."
                            )
            elif op == "$unset":
                for key in data.keys():
                    if "." in key:
                        _unset_nested_value(entity_dict, key)
                    elif key in entity_dict:
                        del entity_dict[key]
            elif op == "$pull":
                for key, value in data.items():
                    if "." in key:
                        nested_array = _get_nested_value(entity_dict, key)
                        if nested_array is not None and isinstance(nested_array, list):
                            new_array = [x for x in nested_array if x != value]
                            _set_nested_value(entity_dict, key, new_array)
                    elif key in entity_dict and isinstance(entity_dict[key], list):
                        entity_dict[key] = [x for x in entity_dict[key] if x != value]
            else:
                raise ValueError(f"Unsupported operator: {op}")
        count += 1
    return count


class MemoryRepository(Repository[T], Generic[T]):
    """
    In-memory repository implementation using Python dictionaries.
    """

    def __init__(
        self, entity_cls: Type[T], app_id_field: str = "id", db_id_field: str = "_id"
    ):
        self._entity_cls = entity_cls
        self._app_id_field = app_id_field
        self._db_id_field = db_id_field
        self._store: Dict[Any, Dict[str, Any]] = {}
        self._app_id_index: Dict[Any, Any] = {}  # Maps application IDs to database IDs
        self._counter = 1  # For generating integer IDs

    def id_generator(self) -> Any:
        # Get the annotation for the ID field (defaulting to str if not present)
        expected_type = self._entity_cls.__annotations__.get(self._app_id_field, str)
        origin = get_origin(expected_type)
        if origin is Union:
            # Extract the types from the union
            args = get_args(expected_type)
            # Prefer int if it's one of the types
            if int in args:
                expected_type = int
            elif str in args:
                expected_type = str
            else:
                raise ValueError(f"Unsupported ID type: {expected_type}")
        if expected_type == int:
            current = self._counter
            self._counter += 1
            return current
        elif expected_type == str:
            return str(uuid.uuid4())
        else:
            raise ValueError(f"Unsupported ID type: {expected_type}")

    @property
    def entity_type(self) -> Type[T]:
        return self._entity_cls

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        field = self._db_id_field if use_db_id else self._app_id_field
        if use_db_id:
            if id not in self._store:
                raise ObjectNotFoundException(
                    f"{self._entity_cls.__name__} with database ID {id} not found"
                )
            entity_dict = self._store[id]
        else:
            if id not in self._app_id_index:
                raise ObjectNotFoundException(
                    f"{self._entity_cls.__name__} with application ID {id} not found"
                )
            db_id = self._app_id_index[id]
            entity_dict = self._store[db_id]
        return self._dict_to_entity(entity_dict)

    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        return await self.get(db_id, logger, timeout, use_db_id=True)

    async def store(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
        return_value: bool = False,
    ) -> Optional[T]:
        """
        Store an entity in the repository.
        If generate_app_id is True and the app_id is missing or None, a new ID will be generated.
        Similarly, if the db_id is missing or None, a new ID will be generated.
        Returns the stored entity if return_value is True.

        Raises:
            KeyAlreadyExistsException: If an entity with the same app_id or db_id already exists.
            ValueError: If the entity is not of the expected type.
        """
        self.validate_entity(entity)
        entity_dict = self._entity_to_dict(entity)

        # Generate app_id if needed.
        if generate_app_id and (
            self._app_id_field not in entity_dict
            or entity_dict.get(self._app_id_field) is None
        ):
            entity_dict[self._app_id_field] = self.id_generator()

        # Generate db_id if needed.
        if (
            self._db_id_field not in entity_dict
            or entity_dict.get(self._db_id_field) is None
        ):
            entity_dict[self._db_id_field] = self.id_generator()

        app_id = entity_dict.get(self._app_id_field)
        db_id = entity_dict.get(self._db_id_field)

        # Check for duplicate app_id only if it's not None.
        if app_id is not None and app_id in self._app_id_index:
            raise KeyAlreadyExistsException(
                f"{self._entity_cls.__name__} with application ID {app_id} already exists"
            )

        # Check for duplicate db_id.
        if db_id in self._store:
            raise KeyAlreadyExistsException(
                f"{self._entity_cls.__name__} with database ID {db_id} already exists"
            )

        # Add to the index.
        if app_id is not None:
            self._app_id_index[app_id] = db_id

        self._store[db_id] = copy.deepcopy(entity_dict)

        if return_value:
            return self._dict_to_entity(self._store[db_id])
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

        if generate_app_id and self._app_id_field not in entity_dict:
            entity_dict[self._app_id_field] = self.id_generator()

        if self._db_id_field not in entity_dict:
            entity_dict[self._db_id_field] = self.id_generator()

        old_db_id = None
        app_id = entity_dict.get(self._app_id_field)
        if app_id is not None and app_id in self._app_id_index:
            old_db_id = self._app_id_index[app_id]
            if old_db_id != entity_dict[self._db_id_field] and old_db_id in self._store:
                del self._store[old_db_id]
        if app_id is not None:
            self._app_id_index[app_id] = entity_dict[self._db_id_field]
        self._store[entity_dict[self._db_id_field]] = copy.deepcopy(entity_dict)

    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        # Use the previously defined update_one function.
        return await update_one(self, options, update, logger, timeout, return_value)

    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        # Use the previously defined update_many function.
        return await update_many(self, options, update, logger, timeout)

    async def delete_many(
        self,
        options: QueryOptions,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Delete entities matching the provided QueryOptions, respecting limit.
        Returns the number of entities deleted.
        """
        logger.debug(
            f"Deleting many {self._entity_cls.__name__} with options: {options}"
        )
        await asyncio.sleep(0)
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete.")
        filtered = await self._filter_entities(options)
        if options.random_order:
            random.shuffle(filtered)
        elif options.sort_by:
            sort_field = options.sort_by
            if sort_field == "id" and self._app_id_field != "id":
                sort_field = self._app_id_field
            elif sort_field == self._db_id_field:
                sort_field = self._db_id_field
            filtered.sort(
                key=lambda x: x.get(sort_field, ""), reverse=options.sort_desc
            )
        if options.offset > 0:
            filtered = filtered[options.offset :]
        if options.limit > 0:
            filtered = filtered[: options.limit]
        count = 0
        for record in filtered:
            db_id = record.get(self._db_id_field) or record.get(self._app_id_field)
            if not db_id or db_id not in self._store:
                continue
            app_id = self._store[db_id].get(self._app_id_field)
            if app_id in self._app_id_index:
                del self._app_id_index[app_id]
            del self._store[db_id]
            count += 1
        return count

    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        options = options or QueryOptions()
        filtered_entities = await self._filter_entities(options)
        sorted_entities = self._sort_entities(filtered_entities, options)
        paginated_entities = sorted_entities[
            options.offset : options.offset + options.limit
        ]
        for entity_dict in paginated_entities:
            yield self._dict_to_entity(entity_dict)

    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        options = options or QueryOptions()
        filtered_entities = await self._filter_entities(options)
        return len(filtered_entities)

    async def _filter_entities(self, options: QueryOptions) -> List[Dict[str, Any]]:
        filtered = []
        for entity_dict in self._store.values():
            if options.expression:
                if self._matches_expression(entity_dict, options.expression):
                    filtered.append(copy.deepcopy(entity_dict))
            else:
                filtered.append(copy.deepcopy(entity_dict))
        return filtered

    def _matches_expression(
        self, entity_dict: Dict[str, Any], expr: Dict[str, Any]
    ) -> bool:
        if "and" in expr:
            return all(
                self._matches_expression(entity_dict, sub_expr)
                for sub_expr in expr["and"]
            )
        if "or" in expr:
            return any(
                self._matches_expression(entity_dict, sub_expr)
                for sub_expr in expr["or"]
            )

        for field, condition in expr.items():
            # Get value considering potential nested fields
            entity_value = self._get_nested_field_value(entity_dict, field)

            if isinstance(condition, dict):
                operator = condition.get("operator", "eq")
                value = condition.get("value")
                if not self._check_operator(operator, entity_value, value):
                    return False
            else:
                if entity_value != condition:
                    return False
        return True

    def _check_operator(
        self, operator: str, entity_value: Any, filter_value: Any
    ) -> bool:
        if operator == "eq":
            return entity_value == filter_value
        elif operator == "ne":
            return entity_value != filter_value
        elif operator == "gt":
            return entity_value > filter_value
        elif operator in ("ge", "gte"):
            return entity_value >= filter_value
        elif operator == "lt":
            return entity_value < filter_value
        elif operator in ("le", "lte"):
            return entity_value <= filter_value
        elif operator == "in":
            return entity_value in filter_value
        elif operator == "nin":
            return entity_value not in filter_value
        elif operator == "contains":
            return isinstance(entity_value, list) and filter_value in entity_value
        elif operator == "startswith":
            return isinstance(entity_value, str) and entity_value.startswith(
                filter_value
            )
        elif operator == "endswith":
            return isinstance(entity_value, str) and entity_value.endswith(filter_value)
        elif operator == "exists":
            return (entity_value is not None) == filter_value
        elif operator == "regex":
            import re

            return isinstance(entity_value, str) and bool(
                re.search(filter_value, entity_value)
            )
        elif operator == "like":
            clean_filter = filter_value.replace("%", "")
            return (
                isinstance(entity_value, str)
                and clean_filter.lower() in entity_value.lower()
            )
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def _sort_entities(
        self, entities: List[Dict[str, Any]], options: QueryOptions
    ) -> List[Dict[str, Any]]:
        if not entities:
            return entities
        if options.random_order:
            # If random ordering is requested, simply shuffle the list.
            entities_copy = list(entities)
            random.shuffle(entities_copy)
            return entities_copy
        sort_field = options.sort_by
        if not sort_field:
            sort_field = (
                self._app_id_field
                if self._app_id_field in entities[0]
                else self._db_id_field
            )
        if sort_field == "id" and self._app_id_field != "id":
            sort_field = self._app_id_field
        elif sort_field == "_id" and self._db_id_field != "_id":
            sort_field = self._db_id_field
        return sorted(
            entities, key=lambda x: x.get(sort_field, None), reverse=options.sort_desc
        )

    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        if hasattr(entity, "model_dump"):
            entity_dict = entity.model_dump()
        else:
            entity_dict = dict(entity.__dict__)
        if self._app_id_field in entity_dict and self._db_id_field not in entity_dict:
            entity_dict[self._db_id_field] = entity_dict[self._app_id_field]
        elif self._db_id_field in entity_dict and self._app_id_field not in entity_dict:
            entity_dict[self._app_id_field] = entity_dict[self._db_id_field]
        return entity_dict

    def _dict_to_entity(self, entity_dict: Dict[str, Any]) -> T:
        entity_data = copy.deepcopy(entity_dict)
        if self._db_id_field in entity_data:
            if self._app_id_field not in entity_data:
                entity_data[self._app_id_field] = entity_data[self._db_id_field]
            if self._app_id_field != self._db_id_field:
                del entity_data[self._db_id_field]
        return self._entity_cls(**entity_data)

    def _get_nested_field_value(self, entity_dict: Dict[str, Any], field: str) -> Any:
        """Get a value from a nested field using dot notation."""
        if "." not in field:
            return entity_dict.get(field)

        parts = field.split(".")
        curr = entity_dict
        for part in parts:
            if not isinstance(curr, dict) or part not in curr:
                return None
            curr = curr[part]
        return curr
