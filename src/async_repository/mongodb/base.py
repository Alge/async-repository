import logging
import re
from dataclasses import is_dataclass, asdict
from logging import LoggerAdapter
from typing import Any, AsyncGenerator, Dict, Generic, Optional, Type, TypeVar

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from repositories.base.interfaces import Repository
from repositories.base.exceptions import ObjectNotFoundException, KeyAlreadyExistsException
from repositories.base.query import QueryOptions
from repositories.base.update import Update
from repositories.base.utils import prepare_for_storage  # Import from centralized location

T = TypeVar('T')
base_logger = logging.getLogger(__name__)


class MongoDBRepository(Repository[T], Generic[T]):
    """
    Base MongoDB repository implementation.
    """

    def __init__(
            self,
            client: AsyncIOMotorClient,
            database_name: str,
            collection_name: str,
            entity_cls: Type[T],
            app_id_field: str = "id",
            db_id_field: str = "_id",
            auto_ensure_indexes: bool = False
    ):
        self._client = client
        self._db = client[database_name]
        self._collection: AsyncIOMotorCollection = self._db[collection_name]
        self._entity_cls = entity_cls
        self._app_id_field = app_id_field
        self._db_id_field = db_id_field
        if auto_ensure_indexes:
            self._ensure_indices()

        base_logger.debug("Initialized mongodb repository for class: %s, client: %s, database: %s, collection: %s",
                          entity_cls, client, database_name, collection_name)

    @property
    def entity_type(self) -> Type[T]:
        return self._entity_cls

    @property
    def app_id_field(self) -> str:
        return self._app_id_field

    @property
    def db_id_field(self) -> str:
        return self._db_id_field

    def _ensure_indices(self) -> None:
        # Override to create indices, e.g.:
        # await self._collection.create_index(self._app_id_field, unique=True, background=True)
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
        query = {field: id}
        doc = await self._collection.find_one(query)
        if not doc:
            id_type = "database" if use_db_id else "application"
            raise ObjectNotFoundException(f"{self._entity_cls.__name__} with {id_type} ID {id} not found")
        return self._document_to_entity(doc)

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
            generate_app_id: bool = True,
            return_value: bool = False
    ) -> Optional[T]:
        self.validate_entity(entity)
        doc = self._entity_to_document(entity)

        # If the app_id_field is missing or its value is None, generate a new ID.
        if generate_app_id and doc.get(self._app_id_field) is None:
            new_app_id = self.id_generator()
            doc[self._app_id_field] = new_app_id

        # Similarly for the db_id_field.
        if generate_app_id and doc.get(self._db_id_field) is None:
            new_db_id = self.id_generator()
            doc[self._db_id_field] = new_db_id

        try:
            await self._collection.insert_one(doc)
        except DuplicateKeyError:
            raise KeyAlreadyExistsException(
                f"{self._entity_cls.__name__} with ID {doc.get(self._app_id_field) or doc.get(self._db_id_field)} already exists"
            )

        if return_value:
            return self._document_to_entity(doc)
        else:
            return None

    async def upsert(
            self,
            entity: T,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            generate_app_id: bool = True
    ) -> None:
        self.validate_entity(entity)
        doc = self._entity_to_document(entity)
        if generate_app_id and self._app_id_field not in doc:
            doc[self._app_id_field] = self.id_generator()
        if self._db_id_field not in doc:
            doc[self._db_id_field] = self.id_generator()
        if "created_at" not in doc:
            pass
        query = {}
        if self._app_id_field in doc:
            query[self._app_id_field] = doc[self._app_id_field]
        elif self._db_id_field in doc:
            query[self._db_id_field] = doc[self._db_id_field]
        else:
            raise ValueError(f"Entity must have either {self._db_id_field} or {self._app_id_field}")
        await self._collection.replace_one(query, doc, upsert=True)

    def _parse_update_object(self, update: Update) -> Dict[str, Any]:
        """
        Parse the Update object and convert it to MongoDB update commands.
        """
        if not isinstance(update, Update):
            raise ValueError("update must be an instance of Update")

        update_operations = update.build()

        if not update_operations:
            raise ValueError("Update object doesn't contain any operations")

        # Convert update operations to MongoDB-compatible format
        update_operations = prepare_for_storage(update_operations)

        return update_operations

    async def update_one(
            self,
            options: QueryOptions,
            update: Update,
            logger: LoggerAdapter,
            timeout: Optional[float] = None,
            return_value: bool = False
    ) -> Optional[T]:
        logger.debug(
            f"Updating (one) {self._entity_cls.__name__} with criteria: {options.expression}, update: {update}")

        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        query = self._transform_expression(options.expression)
        update_operations = self._parse_update_object(update)

        result = await self._collection.update_one(query, update_operations)

        if result.matched_count == 0:
            raise ObjectNotFoundException(f"{self._entity_cls.__name__} not found with criteria {options.expression}")

        if return_value:
            updated_doc = await self._collection.find_one(query)
            if not updated_doc:
                raise ObjectNotFoundException(
                    f"{self._entity_cls.__name__} not found with criteria {options.expression} after update")
            return self._document_to_entity(updated_doc)
        else:
            return None

    async def update_many(
            self,
            options: QueryOptions,
            update: Update,
            logger: LoggerAdapter,
            timeout: Optional[float] = None
    ) -> int:
        logger.debug(
            f"Updating (many) {self._entity_cls.__name__} with criteria: {options.expression}, update: {update}")

        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for update.")

        query = self._transform_expression(options.expression)
        logger.debug("Find using query: %s", query)
        update_operations = self._parse_update_object(update)

        # If limit is specified, retrieve specific document IDs to update.
        if options.limit > 0:
            if options.random_order:
                sample_size = options.limit + options.offset
                pipeline = []
                if query:
                    pipeline.append({"$match": query})
                pipeline.append({"$sample": {"size": sample_size}})
                docs_to_update = await self._collection.aggregate(pipeline).to_list(length=sample_size)
            else:
                sort_direction = DESCENDING if options.sort_desc else ASCENDING
                sort_field = self._app_id_field if not options.sort_by else options.sort_by
                if sort_field == "id" and self._app_id_field != "id":
                    sort_field = self._app_id_field
                elif sort_field == "_id" and self._db_id_field != "_id":
                    sort_field = self._db_id_field
                docs_to_update = await self._collection.find(query).sort(sort_field, sort_direction) \
                    .skip(options.offset).limit(options.limit).to_list(length=options.limit)
            doc_ids = [doc[self._db_id_field] for doc in docs_to_update]
            if not doc_ids:
                return 0
            query = {self._db_id_field: {"$in": doc_ids}}

        result = await self._collection.update_many(query, update_operations)
        return result.modified_count

    async def delete_many(
            self,
            options: QueryOptions,
            logger: LoggerAdapter,
            timeout: Optional[float] = None
    ) -> int:
        logger.debug(f"Deleting many {self._entity_cls.__name__} with options: {options}")

        if not options.expression:
            raise ValueError("QueryOptions must include an 'expression' for delete.")

        # If limit, offset, or sort is specified, get specific document IDs.
        if options.limit > 0 or options.offset > 0 or options.sort_by:
            if options.random_order:
                sample_size = (options.limit if options.limit > 0 else 1000) + options.offset
                pipeline = []
                query = self._build_query(options)
                if query:
                    pipeline.append({"$match": query})
                pipeline.append({"$sample": {"size": sample_size}})
                docs_to_delete = await self._collection.aggregate(pipeline).to_list(length=sample_size)
                docs_to_delete = docs_to_delete[options.offset:(options.offset + options.limit)
                if options.limit > 0 else None]
            else:
                sort_direction = DESCENDING if options.sort_desc else ASCENDING
                sort_field = self._app_id_field if not options.sort_by else options.sort_by
                if sort_field == "id" and self._app_id_field != "id":
                    sort_field = self._app_id_field
                elif sort_field == "_id" and self._db_id_field != "_id":
                    sort_field = self._db_id_field
                query = self._build_query(options)
                docs_to_delete = await self._collection.find(query).sort(sort_field, sort_direction) \
                    .skip(options.offset).limit(options.limit).to_list(length=options.limit)
            doc_ids = [doc[self._db_id_field] for doc in docs_to_delete]
            if not doc_ids:
                return 0
            query = {self._db_id_field: {"$in": doc_ids}}
        else:
            query = self._build_query(options)

        result = await self._collection.delete_many(query)
        return result.deleted_count

    async def list(
            self,
            logger: LoggerAdapter,
            options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        options = options or QueryOptions()
        query = self._build_query(options)
        logger.debug("List using query: %s", query)
        if options.random_order:
            sample_size = options.limit + options.offset
            pipeline = []
            if query:
                pipeline.append({"$match": query})
            pipeline.append({"$sample": {"size": sample_size}})
            docs = await self._collection.aggregate(pipeline).to_list(length=sample_size)
            docs = docs[options.offset: options.offset + options.limit]
        else:
            sort_direction = DESCENDING if options.sort_desc else ASCENDING
            sort_field = self._app_id_field if not options.sort_by else options.sort_by
            if sort_field == "id" and self._app_id_field != "id":
                sort_field = self._app_id_field
            elif sort_field == "_id" and self._db_id_field != "_id":
                sort_field = self._db_id_field
            cursor = self._collection.find(query).sort(sort_field, sort_direction) \
                .skip(options.offset).limit(options.limit)
            docs = await cursor.to_list(length=options.limit)
        for doc in docs:
            yield self._document_to_entity(doc)

    async def count(
            self,
            logger: LoggerAdapter,
            options: Optional[QueryOptions] = None
    ) -> int:
        options = options or QueryOptions()
        logger.debug(f"Counting {self._entity_cls.__name__} with options: {options.__dict__}")
        query = self._build_query(options)
        return await self._collection.count_documents(query)

    def _build_query(self, options: QueryOptions) -> Dict[str, Any]:
        if options.expression:
            return self._transform_expression(options.expression)
        return {}

    def _transform_expression(self, expr: Dict[str, Any]) -> Dict[str, Any]:
        if "and" in expr:
            return {"$and": [self._transform_expression(sub_expr) for sub_expr in expr["and"]]}
        if "or" in expr:
            return {"$or": [self._transform_expression(sub_expr) for sub_expr in expr["or"]]}
        result = {}
        for field, condition in expr.items():
            if isinstance(condition, dict):
                operator = condition.get("operator", "eq")
                value = condition.get("value")
                if operator == "eq":
                    result[field] = value
                elif operator == "ne":
                    result[field] = {"$ne": value}
                elif operator == "gt":
                    result[field] = {"$gt": value}
                elif operator in ("ge", "gte"):
                    result[field] = {"$gte": value}
                elif operator == "lt":
                    result[field] = {"$lt": value}
                elif operator in ("le", "lte"):
                    result[field] = {"$lte": value}
                elif operator == "in":
                    result[field] = {"$in": value}
                elif operator == "nin":
                    result[field] = {"$nin": value}
                elif operator == "contains":
                    # Check if value is an element in the array field
                    result[field] = value
                elif operator == "startswith":
                    result[field] = {"$regex": f"^{re.escape(value)}", "$options": "i"}
                elif operator == "endswith":
                    result[field] = {"$regex": f"{re.escape(value)}$", "$options": "i"}
                elif operator == "exists":
                    result[field] = {"$exists": value}
                elif operator == "regex":
                    result[field] = {"$regex": value, "$options": "i"}
                elif operator == "like":
                    if "%" in value:
                        pattern = ".*".join(re.escape(part) for part in value.split("%"))
                    else:
                        pattern = f".*{re.escape(value)}.*"
                    result[field] = {"$regex": pattern, "$options": "i"}
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
            else:
                result[field] = condition

        base_logger.debug(f"Built query: {result}")
        return result

    def _entity_to_document(self, entity: T) -> Dict[str, Any]:
        """Convert entity to MongoDB document format."""
        if hasattr(entity, "model_dump") and callable(getattr(entity, "model_dump")):
            try:
                # Try model_dump with json mode and by_alias=True (Pydantic v2)
                doc = entity.model_dump(mode="json", by_alias=True)
            except Exception:
                # Fallback to regular model_dump or dict
                try:
                    doc = entity.model_dump(by_alias=True)
                except Exception:
                    if is_dataclass(entity):
                        doc = asdict(entity)
                    else:
                        doc = dict(entity.__dict__)
        elif is_dataclass(entity):
            # Handle dataclasses
            doc = asdict(entity)
        else:
            # Handle any other object
            doc = dict(entity.__dict__)

        # CRITICAL FIX: Always apply prepare_for_storage to the entire document
        # This must be outside all the if/else blocks to ensure it runs in all cases
        doc = prepare_for_storage(doc)

        if self._app_id_field != self._db_id_field and self._app_id_field in doc:
            if self._db_id_field not in doc:
                doc[self._db_id_field] = doc[self._app_id_field]
        return doc

    def _document_to_entity(self, doc: Dict[str, Any]) -> T:
        entity_dict = doc.copy()

        # If app_id_field and db_id_field are different, ensure app_id is populated
        if self._app_id_field != self._db_id_field:
            if self._db_id_field in entity_dict and not entity_dict.get(self._app_id_field):
                entity_dict[self._app_id_field] = entity_dict[self._db_id_field]

        # Remove any fields that shouldn't be passed to the entity constructor
        # This includes db_id_field (if it's different from app_id_field)
        # and any MongoDB-specific fields like '_id' that aren't explicitly used as db_id_field
        if '_id' in entity_dict and '_id' != self._db_id_field:
            del entity_dict['_id']

        if self._db_id_field in entity_dict:
            del entity_dict[self._db_id_field]

        return self._entity_cls(**entity_dict)