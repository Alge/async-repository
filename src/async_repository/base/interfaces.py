# src/async_repository/base/interfaces.py

import uuid
from abc import ABC, abstractmethod
from logging import LoggerAdapter
from typing import (Any, AsyncGenerator, Callable, Dict, Generic, Optional,
                    Type, TypeVar)

# Import local exceptions and types
from async_repository.base.exceptions import (KeyAlreadyExistsException,
                                              ObjectNotFoundException)
from async_repository.base.query import QueryBuilder, QueryOptions
from async_repository.base.update import Update

# Type variable for any entity
T = TypeVar("T")


def generate_id() -> str:
    """Generate a new unique ID for entities."""
    return str(uuid.uuid4())


class Repository(Generic[T], ABC):
    """
    Base repository interface for CRUD operations, schema/index checking,
    and optional explicit creation.

    Provides common operations like get, store, update, delete, list, count,
    and explicit methods (`ensure_schema`, `ensure_indexes`) for setup.
    The `initialize` method orchestrates the explicit creation steps.
    """

    def __init__(self, skip_custom_validation: bool = False):
        """
        Initialize repository with validation settings.

        Args:
            skip_custom_validation: If True, custom validators are skipped when
                                   deserializing entities from the database.
        """
        self._skip_custom_validation = skip_custom_validation

    @property
    @abstractmethod
    def entity_type(self) -> Type[T]:
        """The entity type this repository manages."""
        pass

    @property
    @abstractmethod
    def app_id_field(self) -> str:
        """The field name for the application ID within the entity model."""
        pass

    @property
    @abstractmethod
    def db_id_field(self) -> str:
        """The field name for the database-specific ID used internally."""
        pass

    @property
    def id_generator(self) -> Callable[[], str]:
        """Function to generate new application IDs. Can be overridden."""
        return generate_id

    # --- Initialization and Schema/Index Management ---

    async def initialize(
        self,
        logger: LoggerAdapter,
        create_schema_if_needed: bool = False,
        create_indexes_if_needed: bool = False,
    ) -> None:
        """
        Orchestrates optional, explicit repository setup by calling create_schema
        and/or create_indexes based on the provided flags.

        Call `check_schema` and `check_indexes` separately after __init__ if you
        only want to verify existing structures without modifying them.

        Args:
            logger: Logger adapter for recording initialization steps.
            create_schema_if_needed: If True, call `self.create_schema`.
            create_indexes_if_needed: If True, call `self.create_indexes`.
        """
        logger.info(
            f"Initializing repository setup for {self.entity_type.__name__} "
            f"(Create Schema: {create_schema_if_needed}, Create Indexes: {create_indexes_if_needed})"
        )
        if create_schema_if_needed:
            await self.create_schema(logger)
        if create_indexes_if_needed:
            await self.create_indexes(logger)
        logger.info(
            f"Repository explicit setup complete for {self.entity_type.__name__}."
        )

    @abstractmethod
    async def check_schema(self, logger: LoggerAdapter) -> bool:
        """
        Check if the database schema (table/collection) exists and appears
        compatible with the repository's entity type. Should be non-destructive.
        Intended to be called after __init__ for verification.

        Args:
            logger: Logger adapter for recording check steps.

        Returns:
            True if the schema seems to exist and is compatible, False otherwise.
            May raise exceptions on connection errors.
        """
        pass

    @abstractmethod
    async def check_indexes(self, logger: LoggerAdapter) -> bool:
        """
        Check if the essential database indexes (e.g., on app_id_field) seem
        to exist. Should be non-destructive.
        Intended to be called after __init__ for verification.

        Args:
            logger: Logger adapter for recording check steps.

        Returns:
            True if essential indexes seem to exist, False otherwise.
            May raise exceptions on connection errors.
        """
        pass

    @abstractmethod
    async def create_schema(self, logger: LoggerAdapter) -> None:
        """
        Explicitly create the database schema (table/collection) if it doesn't
        already exist. Should be idempotent.

        WARNING: Depending on implementation, this *could* potentially modify
        existing schemas if not carefully implemented with "IF NOT EXISTS" checks.

        Args:
            logger: Logger adapter for recording schema creation steps.
        """
        pass

    @abstractmethod
    async def create_indexes(self, logger: LoggerAdapter) -> None:
        """
        Explicitly create necessary database indexes (e.g., unique index on
        app_id_field) if they don't already exist. Should be idempotent.

        Args:
            logger: Logger adapter for recording index creation steps.
        """
        pass

    # --- Core CRUD Methods ---

    @abstractmethod
    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        """
        Retrieve an entity by its application or database ID.

        Args:
            id: The identifier (application or database) of the entity.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            use_db_id: If True, treat 'id' as the database-specific ID.

        Returns:
            The retrieved entity instance.

        Raises:
            ObjectNotFoundException: If the entity with the specified ID is not found.
        """
        pass

    @abstractmethod
    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        """
        Retrieve an entity specifically by its database-specific ID.

        Args:
            db_id: The database-specific identifier of the entity.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.

        Returns:
            The retrieved entity instance.

        Raises:
            ObjectNotFoundException: If the entity with the specified database ID is not found.
        """
        pass

    @abstractmethod
    async def store(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
        return_value: bool = False,
    ) -> Optional[T]:
        """
        Store a new entity in the repository.

        Args:
            entity: The entity instance to store.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            generate_app_id: If True, generates a new application ID using `id_generator`
                             if the `app_id_field` is not already set on the entity.
            return_value: If True, returns the newly stored entity (potentially updated
                          with generated IDs or database defaults); otherwise, returns None.

        Returns:
            The stored entity if `return_value` is True, otherwise None.

        Raises:
            ValueError: If the entity instance is invalid (e.g., wrong type).
            KeyAlreadyExistsException: If an entity with the same ID already exists.
        """
        pass

    @abstractmethod
    async def upsert(
        self,
        entity: T,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        generate_app_id: bool = True,
    ) -> None:
        """
        Insert or update an entity based on its ID.

        If an entity with the same ID (determined by `app_id_field` or potentially
        `db_id_field` depending on implementation) does not exist, it is inserted.
        If it already exists, it is updated (typically a full replacement).
        The `generate_app_id` logic applies similarly to `store`.

        Args:
            entity: The entity instance to upsert.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            generate_app_id: If True, generates a new application ID if one doesn't exist
                             and the operation results in an insert.

        Raises:
            ValueError: If the entity instance is invalid.
        """
        pass

    @abstractmethod
    async def update_one(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        return_value: bool = False,
    ) -> Optional[T]:
        """
        Update specific fields of a single entity matching the query options.

        Args:
            options: QueryOptions instance (potentially built using QueryBuilder)
                     containing the filter expression to identify the entity.
                     Implementations should respect options like timeout if provided here.
            update: An Update object (built using the Update builder) describing the
                    modifications (e.g. set, push, increment).
            logger: Logger adapter for recording operations.
            timeout: Optional operation-specific timeout override.
            return_value: If True, returns the updated entity after modifications;
                          otherwise, returns None.

        Returns:
            The updated entity if `return_value` is True, otherwise None.

        Raises:
            ObjectNotFoundException: If no entity is found matching the criteria in `options`.
            ValueError: If the update operations are invalid for the matched entity.
        """
        pass

    @abstractmethod
    async def update_many(
        self,
        options: QueryOptions,
        update: Update,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Update specific fields of all entities matching the query options.

        Args:
            options: QueryOptions instance (potentially built using QueryBuilder)
                     containing the filter expression to identify entities.
                     Implementations should respect options like timeout if provided here.
            update: An Update object (built using the Update builder) describing the
                    modifications to apply to all matched entities.
            logger: Logger adapter for recording operations.
            timeout: Optional operation-specific timeout override.

        Returns:
            The number of entities that were successfully updated.

        Raises:
            ValueError: If the update operations are invalid.
        """
        pass

    async def delete_one(
        self,
        identifier: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> None:
        """
        Delete a single entity specified by its application or database ID.

        This default implementation reuses `delete_many`.

        Args:
            identifier: The unique identifier (application or database) of the entity to delete.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            use_db_id: If True, treat 'identifier' as the database-specific ID.

        Raises:
            ObjectNotFoundException: If no entity with the given ID exists.
        """
        field_to_match = self.db_id_field if use_db_id else self.app_id_field
        # Using QueryBuilder internally for consistency
        qb = QueryBuilder(self.entity_type)
        options = (
            qb.filter(getattr(qb.fields, field_to_match) == identifier)
            .limit(1)
            .build()
        )

        count_deleted = await self.delete_many(options, logger, timeout)

        if count_deleted == 0:
            raise ObjectNotFoundException(
                f"{self.entity_type.__name__} with ID '{identifier}' not found for deletion."
            )
        elif count_deleted > 1:
            logger.warning(
                f"delete_one attempted for ID '{identifier}' but {count_deleted} entities were deleted."
            )

    @abstractmethod
    async def delete_many(
        self,
        options: QueryOptions,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Delete all entities matching the provided query options.

        Args:
            options: QueryOptions instance containing the filter expression
                     for selecting entities to delete.
            logger: Logger adapter for recording operations.
            timeout: Optional operation-specific timeout override.

        Returns:
            The number of entities that were successfully deleted.
        """
        pass

    @abstractmethod
    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        """
        List entities matching the provided query options.

        Args:
            logger: Logger adapter for recording operations.
            options: Optional QueryOptions instance for filtering, sorting,
                     and pagination. If None, list all (or default limit).

        Yields:
            Entity instances (`T`) matching the query criteria.
        """
        # Abstract method requires yield, but it won't be executed.
        if False:  # pragma: no cover
            yield

    @abstractmethod
    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        """
        Count entities matching the provided query options.

        Args:
            logger: Logger adapter for recording operations.
            options: Optional QueryOptions instance containing the filter
                     expression. If None, counts all entities.

        Returns:
            The total number of entities matching the query criteria.
        """
        pass

    # --- Helper Methods ---

    def validate_entity(self, entity: T) -> None:
        """
        Basic validation that an entity instance is of the expected type.
        Concrete implementations might add more specific validation.

        Args:
            entity: The entity instance to validate.

        Raises:
            ValueError: If the entity is not an instance of `self.entity_type`.
        """
        if not isinstance(entity, self.entity_type):
            raise ValueError(
                f"Entity must be of type {self.entity_type.__name__}, "
                f"but received {type(entity).__name__}"
            )

    async def find_one(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> T:
        """
        Find a single entity matching the provided query options.

        This default implementation modifies the query options to ensure only one
        result is returned and then uses `list()` to fetch it.

        Args:
            logger: Logger adapter for recording operations.
            options: Optional QueryOptions instance for filtering and sorting.
                     Pagination (`limit`, `offset`) will be overridden.

        Returns:
            The first entity matching the query options.

        Raises:
            ObjectNotFoundException: If no entity is found matching the criteria.
        """
        # Ensure QueryOptions has a copy method (add to query.py if missing)
        # Create a new options object or copy the existing one
        if options and hasattr(options, "copy"):
            query_options = options.copy()
        else:
            # If no options or no copy method, create new default options
            # and potentially copy the expression if it exists
            query_options = QueryOptions()
            if options and options.expression:
                query_options.expression = options.expression # Shallow copy expression
            # Copy other relevant fields if needed, e.g., sort_by

        query_options.limit = 1
        query_options.offset = 0

        async for entity in self.list(logger, query_options):
            return entity  # Return the first (and only) entity found

        # If the loop finishes without returning, no entity was found
        raise ObjectNotFoundException(
            f"No {self.entity_type.__name__} found matching the provided criteria."
        )