# src/async_repository/backends/mongodb_repository.py

import json
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import LoggerAdapter
from typing import (Any, AsyncGenerator, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar)

# --- Motor Driver Import ---
from bson import ObjectId  # Often used for _id
from motor.motor_asyncio import (AsyncIOMotorClient, AsyncIOMotorCollection,
                                 AsyncIOMotorDatabase)
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid, DuplicateKeyError

# --- Framework Imports ---
from async_repository.base.exceptions import (KeyAlreadyExistsException,
                                              ObjectNotFoundException)
from async_repository.base.interfaces import Repository
from async_repository.base.model_validator import _is_none_type, get_type_hints
from async_repository.base.query import (QueryExpression, QueryFilter,
                                         QueryLogical, QueryOperator,
                                         QueryOptions)
from async_repository.base.update import (IncrementOperation, MaxOperation,
                                          MinOperation, MultiplyOperation,
                                          PopOperation, PullOperation,
                                          PushOperation, SetOperation,
                                          UnsetOperation, Update,
                                          UpdateOperation)
from async_repository.base.utils import prepare_for_storage

# --- Type Variables ---
T = TypeVar("T")
DB_CONNECTION_TYPE = AsyncIOMotorClient
DB_COLLECTION_TYPE = AsyncIOMotorCollection
DB_RECORD_TYPE = Dict[str, Any]

base_logger = logging.getLogger("async_repository.backends.mongodb_repository")


class MongoDBRepository(Repository[T], Generic[T]):
    """
    MongoDB repository implementation using Motor.

    Implements checking and explicit creation for schema (collection) and indexes.
    Expects external transaction management if operations need to span multiple
    repository calls.
    """

    def __init__(
        self,
        client: AsyncIOMotorClient,
        database_name: str,
        collection_name: str,
        entity_type: Type[T],
        app_id_field: str = "id",
        db_id_field: str = "_id",
    ):
        """
        Initialize the MongoDB repository. Does NOT perform schema/index creation or checks.
        Call check/create/initialize methods explicitly after creation if needed.

        Args:
            client: An instance of AsyncIOMotorClient.
            database_name: The name of the MongoDB database.
            collection_name: The name of the MongoDB collection.
            entity_type: The Python class representing the entity.
            app_id_field: The attribute name on the entity class for the application ID.
            db_id_field: The attribute name for the MongoDB document ID ('_id' usually).
        """
        if not isinstance(client, AsyncIOMotorClient):
            raise TypeError("client must be an instance of AsyncIOMotorClient")

        self._client = client
        self._database_name = database_name
        self._db: AsyncIOMotorDatabase = client[database_name]
        self._collection_name = collection_name
        self._collection: AsyncIOMotorCollection = self._db[collection_name]
        self._entity_type = entity_type
        self._app_id_field = app_id_field

        if db_id_field != "_id":
            base_logger.warning(
                f"MongoDB typically uses '_id' as the db_id_field, "
                f"but '{db_id_field}' was provided for {entity_type.__name__}."
            )
        self._db_id_field = db_id_field

        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}[{entity_type.__name__}]"
        )
        self._logger.info(
            f"Repository instance created for {entity_type.__name__} "
            f"(db: '{database_name}', collection: '{collection_name}'). "
            "Run check/create/initialize methods for setup."
        )

    @property
    def entity_type(self) -> Type[T]:
        return self._entity_type

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[AsyncIOMotorCollection, None]:
        """
        Provides the collection object directly. Lets operational exceptions
        propagate to the caller.
        """
        yield self._collection

    async def check_schema(self, logger: LoggerAdapter) -> bool:
        """Check if the MongoDB collection exists."""
        logger.info(f"Checking schema (collection '{self._collection_name}')...")
        try:
            collection_names = await self._db.list_collection_names(
                filter={"name": self._collection_name}
            )
            exists = len(collection_names) > 0
            if exists:
                logger.info(
                    f"Schema check PASSED for collection '{self._collection_name}'."
                )
            else:
                logger.warning(
                    f"Schema check FAILED: Collection '{self._collection_name}' not found."
                )
            return exists
        except Exception as e:
            logger.error(
                f"Error during schema check for '{self._collection_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"checking schema for {self._collection_name}")
            return False  # Should not be reached if _handle_db_error raises

    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """Check if essential indexes (e.g., unique on app_id_field) exist."""
        logger.info(
            f"Checking essential indexes for '{self._collection_name}'..."
        )
        target_field = self.app_id_field
        is_unique_needed = target_field != self.db_id_field

        try:
            async with self._get_session() as collection:
                index_info = await collection.index_information()

            found_essential = False
            for name, info in index_info.items():
                keys = info.get("key", [])
                if not keys or keys[0][0] != target_field:
                    continue
                if is_unique_needed:
                    if info.get("unique"):
                        found_essential = True
                        break
                else:  # Existence is enough if it's _id
                    found_essential = True
                    break

            if found_essential:
                logger.info(
                    f"Index check PASSED for '{self._collection_name}' "
                    f"(found index on '{target_field}')."
                )
            else:
                logger.warning(
                    f"Index check FAILED: Essential index on '{target_field}' "
                    f"for collection '{self._collection_name}' not found "
                    "or not unique (if required)."
                )
            return found_essential
        except Exception as e:
            logger.error(
                f"Error during index check for '{self._collection_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"checking indexes for {self._collection_name}")
            return False  # Should not be reached

    async def create_schema(self, logger: LoggerAdapter) -> None:
        """Explicitly create the collection. Idempotent."""
        logger.info(
            f"Attempting to create schema (collection '{self._collection_name}')..."
        )
        try:
            await self._db.create_collection(self._collection_name)
            logger.info(
                f"Collection '{self._collection_name}' ensured to exist."
            )
        except CollectionInvalid:
            logger.info(
                f"Collection '{self._collection_name}' already exists."
            )
        except Exception as e:
            logger.error(
                f"Failed to create collection '{self._collection_name}': {e}",
                exc_info=True,
            )
            self._handle_db_error(e, f"creating schema for {self._collection_name}")

    async def create_indexes(self, logger: LoggerAdapter) -> None:
        """Explicitly create necessary database indexes if they don't exist."""
        logger.info(
            f"Attempting to create indexes for '{self._collection_name}'..."
        )
        indexes_to_create = []

        # 1. Unique index on app_id_field if it's not the main _id
        if self.app_id_field != self.db_id_field:
            indexes_to_create.append(
                {
                    "keys": [(self.app_id_field, ASCENDING)],
                    "options": {
                        "name": f"{self.app_id_field}_unique_idx",
                        "unique": True,
                        "background": True,
                    },
                }
            )

        # 2. Add other desired indexes here
        # Example: indexes_to_create.append({"keys": [...], "options": {...}})

        if not indexes_to_create:
            logger.info(
                f"No explicit indexes defined for creation on '{self._collection_name}'."
            )
            return

        try:
            async with self._get_session() as collection:
                for index_spec in indexes_to_create:
                    keys = index_spec["keys"]
                    options = index_spec["options"]
                    name = options["name"]
                    logger.debug(
                        f"Ensuring index '{name}' on keys {keys} with options {options}"
                    )
                    await collection.create_index(keys, **options)
            logger.info(
                f"Index creation/verification complete for '{self._collection_name}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to create indexes for '{self._collection_name}': {e}",
                exc_info=True,
            )
            # Index creation errors can be complex, re-raise as runtime
            raise RuntimeError(
                f"Failed to create indexes for {self._collection_name}"
            ) from e

    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        field_to_match = self.db_id_field if use_db_id else self.app_id_field
        logger.debug(
            f"Getting {self.entity_type.__name__} by {field_to_match}='{id}'"
        )
        query_id: Any = id
        if field_to_match == "_id":
            try:
                query_id = ObjectId(id)
            except Exception:
                logger.warning(
                    f"Cannot convert id '{id}' to ObjectId for _id query. "
                    "Querying as string."
                )

        query = {field_to_match: query_id}
        record_data: Optional[DB_RECORD_TYPE] = None

        try:
            async with self._get_session() as collection:
                record_data = await collection.find_one(query)

            if record_data is None:
                id_type = (
                    "database (_id)"
                    if use_db_id
                    else f"application ({self.app_id_field})"
                )
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

        except ObjectNotFoundException:
            raise  # Let specific exception propagate
        except Exception as e:
            self._handle_db_error(e, f"getting entity ID {id}")
            # Ensure method raises if _handle_db_error doesn't (it should)
            raise RuntimeError("Unhandled error in get operation") from e

    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        logger.debug(
            f"Getting {self.entity_type.__name__} by DB ID '{db_id}'"
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
                f"Entity {self.entity_type.__name__} must have ID field "
                f"'{self.app_id_field}' set or generate_app_id must be True."
            )

        logger.debug(
            f"Storing new {self.entity_type.__name__} with {self.app_id_field} '{app_id}'"
        )
        db_doc = self._serialize_entity(entity)

        try:
            async with self._get_session() as collection:
                result = await collection.insert_one(db_doc)
            generated_db_id = result.inserted_id
            logger.info(
                f"Stored new {self.entity_type.__name__} with app_id '{app_id}', "
                f"db_id '{generated_db_id}'."
            )
            if return_value:
                if hasattr(entity, self.db_id_field):
                    setattr(entity, self.db_id_field, generated_db_id)
                return entity
            else:
                return None
        except DuplicateKeyError as e:
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
        db_doc = self._serialize_entity(entity)
        query_filter = {self.app_id_field: app_id}

        try:
            async with self._get_session() as collection:
                await collection.replace_one(query_filter, db_doc, upsert=True)
            logger.info(
                f"Upserted {self.entity_type.__name__} with ID '{app_id}'."
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
            f"Updating one {self.entity_type.__name__} matching: {options!r} "
            f"with update: {update!r}"
        )
        if not update:
            if return_value:
                return await self.find_one(logger=logger, options=options)
            return None
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update_one.")

        query_filter: Dict[str, Any] = {}
        update_doc: Dict[str, Any] = {}
        try:
            query_parts = self._translate_query_options(options, False)
            query_filter = query_parts.get("filter", {})
            update_doc = self._translate_update(update)
            logger.debug(
                f"MongoDB update_one filter: {query_filter}, update: {update_doc}"
            )
            if not query_filter:
                raise ValueError("Cannot update_one without a valid filter expression.")
            if not update_doc:
                raise ValueError("Cannot update_one without valid update operations.")
        except (ValueError, TypeError) as translation_error:
            logger.error(
                f"Failed to translate query/update for update_one: {translation_error}",
                exc_info=True,
            )
            raise translation_error

        try:
            from pymongo import ReturnDocument

            async with self._get_session() as collection:
                updated_doc = await collection.find_one_and_update(
                    query_filter,
                    update_doc,
                    return_document=(
                        ReturnDocument.AFTER
                        if return_value
                        else ReturnDocument.BEFORE
                    ),
                )
            if updated_doc is None:
                raise ObjectNotFoundException(
                    f"No {self.entity_type.__name__} found matching criteria for update."
                )
            logger.info(f"Updated one {self.entity_type.__name__} matching criteria.")
            return self._deserialize_record(updated_doc) if return_value else None
        except ObjectNotFoundException:
            raise
        except Exception as db_error:
            self._handle_db_error(db_error, "updating one entity")

    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        logger.debug(
            f"Updating many {self.entity_type.__name__} matching: {options!r} "
            f"with update: {update!r}"
        )
        if not update:
            return 0
        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update_many.")
        if options.limit or options.offset or options.sort_by or options.random_order:
            logger.warning(...)

        query_filter: Dict[str, Any] = {}
        update_doc: Dict[str, Any] = {}
        try:
            query_parts = self._translate_query_options(options, False)
            query_filter = query_parts.get("filter", {})
            update_doc = self._translate_update(update)
            logger.debug(
                f"MongoDB update_many filter: {query_filter}, update: {update_doc}"
            )
            if not query_filter:
                raise ValueError("Cannot update_many without a valid filter expression.")
            if not update_doc:
                raise ValueError("Cannot update_many without valid update operations.")
        except (ValueError, TypeError) as translation_error:
            logger.error(
                f"Failed to translate query/update for update_many: {translation_error}",
                exc_info=True,
            )
            raise translation_error
        try:
            async with self._get_session() as collection:
                result = await collection.update_many(query_filter, update_doc)
                affected_count = result.modified_count
            logger.info(f"Updated {affected_count} {self.entity_type.__name__}(s).")
            return affected_count
        except Exception as db_error:
            self._handle_db_error(db_error, "updating many entities")

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
            logger.warning(...)

        query_filter: Dict[str, Any] = {}
        try:
            query_parts = self._translate_query_options(options, False)
            query_filter = query_parts.get("filter", {})
            logger.debug(f"MongoDB delete_many filter: {query_filter}")
            if not query_filter:
                raise ValueError("Cannot delete_many without a valid filter expression.")
        except (ValueError, TypeError) as translation_error:
            logger.error(
                f"Failed to translate query for delete_many: {translation_error}",
                exc_info=True,
            )
            raise translation_error
        try:
            async with self._get_session() as collection:
                result = await collection.delete_many(query_filter)
                affected_count = result.deleted_count
            logger.info(f"Deleted {affected_count} {self.entity_type.__name__}(s).")
            return affected_count
        except Exception as db_error:
            self._handle_db_error(db_error, "deleting many entities")

    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        effective_options = options or QueryOptions()
        logger.debug(
            f"Listing {self.entity_type.__name__}(s) with options: {effective_options!r}"
        )
        query_parts: Dict[str, Any] = {}
        try:
            query_parts = self._translate_query_options(effective_options, True)
            logger.debug(f"MongoDB list query parts: {query_parts}")
        except (ValueError, TypeError) as translation_error:
            logger.error(
                f"Failed to translate query for list: {translation_error}",
                exc_info=True,
            )
            raise translation_error
        try:
            query_filter = query_parts.get("filter", {})
            async with self._get_session() as collection:
                if effective_options.random_order:
                    pipeline = []
                    if query_filter:
                        pipeline.append({"$match": query_filter})
                    sample_size = query_parts.get("limit", 0)
                    skip = query_parts.get("skip", 0)
                    limit = query_parts.get("limit", 0)
                    if skip > 0 and sample_size > 0:
                        sample_size += skip
                        logger.warning(...)
                    elif sample_size == 0:
                        logger.error(...)
                        raise ValueError(...)
                    pipeline.append({"$sample": {"size": sample_size}})
                    logger.debug(
                        f"Using aggregation pipeline for random order: {pipeline}"
                    )
                    mongo_cursor = collection.aggregate(pipeline)
                    items_yielded = 0
                    items_skipped = 0
                    async for record_data in mongo_cursor:
                        if items_skipped < skip: items_skipped += 1; continue
                        if limit > 0 and items_yielded >= limit: break
                        try:
                            yield self._deserialize_record(record_data)
                            items_yielded += 1
                        except Exception as de_err: logger.error(...); continue
                else:
                    mongo_cursor = collection.find(query_filter)
                    if query_parts.get("sort"):
                        mongo_cursor = mongo_cursor.sort(query_parts["sort"])
                    if query_parts.get("skip", 0) > 0:
                        mongo_cursor = mongo_cursor.skip(query_parts["skip"])
                    limit = query_parts.get("limit", 0)
                    if limit > 0:
                        mongo_cursor = mongo_cursor.limit(limit)
                    async for record_data in mongo_cursor:
                        try: yield self._deserialize_record(record_data)
                        except Exception as de_err: logger.error(...); continue
        except Exception as db_error:
            self._handle_db_error(db_error, "listing entities")

    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        effective_options = options or QueryOptions()
        logger.debug(
            f"Counting {self.entity_type.__name__}(s) with options: {effective_options!r}"
        )
        query_parts: Dict[str, Any] = {}
        try:
            query_parts = self._translate_query_options(effective_options, False)
            query_filter = query_parts.get("filter", {})
            logger.debug(f"MongoDB count filter: {query_filter}")
        except (ValueError, TypeError) as translation_error:
            logger.error(
                f"Failed to translate query for count: {translation_error}",
                exc_info=True,
            )
            raise translation_error
        try:
            async with self._get_session() as collection:
                count_val = await collection.count_documents(query_filter)
            logger.info(f"Counted {count_val} {self.entity_type.__name__}(s).")
            return int(count_val)
        except Exception as db_error:
            self._handle_db_error(db_error, "counting entities")

    # --- Helper Method Implementations ---
    def _serialize_entity(self, entity: T) -> Dict[str, Any]:
        # ... (implementation unchanged) ...
        if hasattr(entity, 'model_dump'):
             try: data = entity.model_dump(by_alias=False, mode='python')
             except TypeError: data = entity.model_dump(by_alias=False)
        elif hasattr(entity, 'dict'): data = entity.dict(by_alias=False)
        elif hasattr(entity, '__dict__'): from dataclasses import is_dataclass, asdict; data = asdict(entity) if is_dataclass(entity) else entity.__dict__
        else: raise TypeError(f"Cannot automatically serialize entity: {type(entity).__name__}")
        prepared_data = prepare_for_storage(data)
        if self.app_id_field not in prepared_data and hasattr(entity, self.app_id_field):
             app_id_value = getattr(entity, self.app_id_field);
             if app_id_value is not None: prepared_data[self.app_id_field] = app_id_value
        app_id_in_data = prepared_data.get(self.app_id_field)
        if self.app_id_field != self.db_id_field and app_id_in_data is not None: prepared_data[self.db_id_field] = app_id_in_data
        if self.db_id_field == "_id" and prepared_data.get("_id") is None: prepared_data.pop("_id", None)
        final_data = {k: v for k, v in prepared_data.items() if not k.startswith("_") or k == "_id"}
        return final_data


    def _deserialize_record(self, record_data: DB_RECORD_TYPE) -> T:
        """
        Converts a MongoDB document (dict) into an entity object T,
        ensuring retrieved datetimes are timezone-aware (UTC).
        """
        if record_data is None:
            raise ValueError("Cannot deserialize None record data.")

        raw_dict = record_data.copy()
        entity_init_kwargs: Dict[str, Any] = {}

        try:
            entity_fields = set(get_type_hints(self.entity_type).keys())
        except Exception:
            entity_fields = set(getattr(self.entity_type, '__annotations__', {}).keys())
            if hasattr(self.entity_type, '__slots__'):
                entity_fields.update(self.entity_type.__slots__)

        # Handle ID mapping
        db_id_value = raw_dict.get("_id")
        if db_id_value is not None:
            if self.db_id_field in entity_fields:
                entity_init_kwargs[self.db_id_field] = db_id_value
            if self.app_id_field != self.db_id_field and self.app_id_field in entity_fields:
                if self.app_id_field not in entity_init_kwargs:
                    entity_init_kwargs[self.app_id_field] = str(db_id_value)

        # Copy known fields and make datetimes timezone-aware
        for field_name in entity_fields:
            if field_name == self.db_id_field or field_name == self.app_id_field:
                continue # Skip IDs already handled

            if field_name in raw_dict:
                value = raw_dict[field_name]

                # --- Make naive datetimes UTC-aware ---
                if isinstance(value, datetime) and value.tzinfo is None:
                    entity_init_kwargs[field_name] = value.replace(tzinfo=timezone.utc)
                else:
                    entity_init_kwargs[field_name] = value
                # --- End datetime handling ---

        # Remove original _id if it's not an expected field
        if "_id" not in entity_fields:
            entity_init_kwargs.pop("_id", None) # Use pop with default None

        try:
            # Instantiate the entity using the prepared dictionary
            entity = self.entity_type(**entity_init_kwargs)
            return entity
        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                 self._logger.error(f"Failed to instantiate {self.entity_type.__name__} due to unexpected args: {e}. Filtered Data: {entity_init_kwargs!r}", exc_info=True)
                 raise ValueError(f"Mismatch between DB record and {self.entity_type.__name__} fields during deserialization.") from e
            else:
                 self._logger.error(f"TypeError during {self.entity_type.__name__} instantiation: {e}. Data: {entity_init_kwargs!r}", exc_info=True)
                 raise ValueError(f"Failed to deserialize record into {self.entity_type.__name__} due to type mismatch during initialization.") from e
        except Exception as e:
            self._logger.error(f"Failed to instantiate {self.entity_type.__name__}: {e}. Attempted kwargs: {list(entity_init_kwargs.keys())!r}", exc_info=True)
            raise ValueError(f"Failed to deserialize database record into {self.entity_type.__name__}") from e

    def _translate_query_options(
        self, options: QueryOptions, include_sorting_pagination: bool = False
    ) -> Dict[str, Any]:
        native_parts: Dict[str, Any] = {"filter": {}}
        if options.expression:
            native_parts["filter"] = self._translate_expression_recursive(
                options.expression
            )
        else:
            native_parts["filter"] = {}

        if include_sorting_pagination:
            if options.random_order:
                native_parts["sort"] = None  # Handled by $sample in list()
            elif options.sort_by:
                sort_field = options.sort_by
                if sort_field == self.app_id_field and self.db_id_field == '_id':
                    sort_field = self.db_id_field
                native_parts["sort"] = [
                    (sort_field, DESCENDING if options.sort_desc else ASCENDING)
                ]
            else:
                native_parts["sort"] = None

            native_parts["limit"] = options.limit if options.limit > 0 else 0
            native_parts["skip"] = options.offset

        self._logger.debug(
            f"Translated QueryOptions to MongoDB parts: {native_parts}"
        )
        return native_parts
    

    def _translate_expression_recursive(
        self, expression: QueryExpression
    ) -> Dict[str, Any]:
        self._logger.debug(f"Translating expression part: {expression!r}")
        translated_part = {}
        if isinstance(expression, QueryFilter):
            field = expression.field_path
            op = expression.operator
            val = expression.value
            # ... (ID mapping logic) ...
            if field == self.app_id_field and self.db_id_field == '_id': field = self.db_id_field
            if field == "_id" and isinstance(val, str):
                 try: val = ObjectId(val)
                 except Exception: self._logger.debug(f"Could not convert value '{val}' to ObjectId for _id query.")

            mongo_op_map = { # ... (mapping remains the same) ...
                QueryOperator.EQ: "$eq", QueryOperator.NE: "$ne", QueryOperator.GT: "$gt", QueryOperator.GTE: "$gte", QueryOperator.LT: "$lt", QueryOperator.LTE: "$lte", QueryOperator.IN: "$in", QueryOperator.NIN: "$nin", QueryOperator.EXISTS: "$exists", QueryOperator.CONTAINS: "$regex", QueryOperator.LIKE: "$regex", QueryOperator.STARTSWITH: "$regex", QueryOperator.ENDSWITH: "$regex"}
            mongo_op = mongo_op_map.get(op)

            if op == QueryOperator.EQ: translated_part = {field: val}
            elif op == QueryOperator.CONTAINS: pattern = f".*{re.escape(str(val))}.*"; translated_part = {field: {"$regex": pattern, "$options": "i"}}
            elif op == QueryOperator.LIKE: pattern = ".*".join(re.escape(part) for part in str(val).split("%")); translated_part = {field: {"$regex": pattern, "$options": "i"}}
            elif op == QueryOperator.STARTSWITH: pattern = f"^{re.escape(str(val))}"; translated_part = {field: {"$regex": pattern}}
            elif op == QueryOperator.ENDSWITH: pattern = f"{re.escape(str(val))}$"; translated_part = {field: {"$regex": pattern}}
            elif op == QueryOperator.EXISTS:
                if not isinstance(val, bool): val = bool(val)
                if val is True: translated_part = {field: {"$exists": True, "$ne": None}}
                else: translated_part = {field: {"$eq": None}}
            elif mongo_op:
                if op in (QueryOperator.IN, QueryOperator.NIN):
                    if not isinstance(val, list):
                        if isinstance(val, (tuple, set)): val = list(val)
                        else: raise TypeError(f"Value for MongoDB {mongo_op} must be a list.")
                translated_part = {field: {mongo_op: val}}
            else:
                # Log the specific error and raise ValueError with the correct message
                error_msg = f"Unsupported query operator for MongoDB: {op!r}" # Use repr(op) for clarity
                self._logger.error(
                    f"Encountered unhandled QueryOperator during MongoDB translation: {op!r}"
                )
                raise ValueError(error_msg)

        elif isinstance(expression, QueryLogical):
            translated = [self._translate_expression_recursive(cond) for cond in expression.conditions if cond]; filtered = [cond for cond in translated if cond]
            if not filtered: translated_part = {}
            elif len(filtered) == 1: translated_part = filtered[0]
            else: mongo_logic_op = "$and" if expression.operator == "and" else "$or"; translated_part = {mongo_logic_op: filtered}
        else:
            raise TypeError(f"Unknown QueryExpression type: {type(expression)}")

        self._logger.debug(f" -> Translated part: {translated_part}")
        return translated_part

    def _translate_update(self, update: Update) -> Dict[str, Any]:
        mongo_update_doc: Dict[str, Dict] = {}
        self._logger.debug(f"Translating Update object: {update!r}")
        for op in update.build():
            field = op.field_path; op_key: str = ""
            if field == self.app_id_field and self.db_id_field == '_id': field = self.db_id_field
            if isinstance(op, SetOperation): op_key = "$set"; mongo_update_doc.setdefault(op_key, {})[field] = op.value
            elif isinstance(op, UnsetOperation): op_key = "$unset"; mongo_update_doc.setdefault(op_key, {})[field] = ""
            elif isinstance(op, IncrementOperation): op_key = "$inc"; mongo_update_doc.setdefault(op_key, {})[field] = op.amount
            elif isinstance(op, MultiplyOperation): op_key = "$mul"; mongo_update_doc.setdefault(op_key, {})[field] = op.factor
            elif isinstance(op, MinOperation): op_key = "$min"; mongo_update_doc.setdefault(op_key, {})[field] = op.value
            elif isinstance(op, MaxOperation): op_key = "$max"; mongo_update_doc.setdefault(op_key, {})[field] = op.value
            elif isinstance(op, PushOperation): op_key = "$push"; mongo_update_doc.setdefault(op_key, {})[field] = {"$each": op.items}
            elif isinstance(op, PopOperation): op_key = "$pop"; mongo_update_doc.setdefault(op_key, {})[field] = op.position
            elif isinstance(op, PullOperation): op_key = "$pull"; mongo_update_doc.setdefault(op_key, {})[field] = op.value_or_condition
            else: raise TypeError(f"Unsupported UpdateOperation type: {type(op)}")
        self._logger.debug(f" -> Translated MongoDB update document: {mongo_update_doc}")
        return mongo_update_doc

    def _handle_db_error(self, error: Exception, context: str = "operation") -> None:
        if isinstance(error, ObjectNotFoundException):
            self._logger.warning(f"Object not found during {context}: {error}")
            raise error
        self._logger.error(
            f"MongoDB error during {context}: {error}", exc_info=True
        )
        if isinstance(error, DuplicateKeyError):
            match = re.search(r"index: (\S+).* dup key: ({.*?})", str(error))
            index = match.group(1) if match else "unknown"
            key = match.group(2) if match else "unknown"
            raise KeyAlreadyExistsException(
                f"Duplicate key error on index '{index}'. Key: {key}"
            ) from error
        else:
            raise RuntimeError(
                f"An unexpected MongoDB error occurred during {context}"
            ) from error