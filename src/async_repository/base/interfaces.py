from abc import ABC, abstractmethod
import uuid
from logging import LoggerAdapter
from typing import AsyncGenerator, Dict, Generic, Optional, Type, TypeVar, Any, Callable

from async_repository.base.exceptions import ObjectNotFoundException
from async_repository.base.query import QueryOptions
from async_repository.base.update import Update  # New custom Update class

# Type variable for any entity
T = TypeVar("T")


def generate_id() -> str:
    """Generate a new unique ID for entities."""
    return str(uuid.uuid4())


class Repository(Generic[T], ABC):
    """
    Base repository interface for CRUD operations on a specific entity type.

    Provides common operations like get, store, update, delete, list, and count.
    The repository handles both application IDs (exposed to users) and database IDs
    (internal to the storage system).
    """

    @property
    @abstractmethod
    def entity_type(self) -> Type[T]:
        """The entity type this repository manages."""
        pass

    @property
    @abstractmethod
    def app_id_field(self) -> str:
        """The field name for the application ID."""
        pass

    @property
    @abstractmethod
    def db_id_field(self) -> str:
        """The field name for the database-specific ID."""
        pass

    @property
    def id_generator(self) -> Callable[[], str]:
        """Function to generate new application IDs. Can be overridden."""
        return generate_id

    @abstractmethod
    async def get(
        self,
        id: str,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
        use_db_id: bool = False,
    ) -> T:
        """
        Retrieve an entity by its ID.
        """
        pass

    @abstractmethod
    async def get_by_db_id(
        self, db_id: Any, logger: LoggerAdapter, timeout: Optional[float] = None
    ) -> T:
        """
        Retrieve an entity by its database-specific ID.
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
            entity: The entity to store.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            generate_app_id: If True, generates a new application ID if one doesn't exist.
            return_value: If True, returns the newly stored entity; otherwise, returns None.

        Raises:
            ValueError: If the entity is invalid.
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
        Insert or update an entity in the repository.

        If the entity does not exist, it is inserted.
        If it already exists, its fields are updated.
        This method uses the entity itself (and its identifiers) to perform the upsert
        without relying on external DSL criteria.

        Args:
            entity: The entity to upsert.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            generate_app_id: If True, generates a new application ID if one doesn't exist.
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
        Update specific fields of a single entity matching the provided filter options.
        The update instructions should be built using the Update builder.

        Args:
            filter_options: QueryOptions instance containing a DSL expression for filtering the entity.
            update: An Update object describing the operations (e.g. set, push, pop, unset, pull).
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            return_value: If True, returns the updated entity; otherwise, returns None.

        Raises:
            ObjectNotFoundException: If no entity is found matching the criteria.
            ValueError: If the update is invalid.
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
        Update specific fields of all entities matching the provided filter options.
        The update instructions should be built using the Update builder.

        Args:
            filter_options: QueryOptions instance containing a DSL expression for filtering the entities.
            update: An Update object describing the operations.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.

        Returns:
            The number of entities updated.

        Raises:
            ValueError: If the update is invalid.
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
        Delete a single entity from the repository by reusing delete_many.

        Args:
            identifier: The unique identifier of the entity to delete.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.
            use_db_id: If True, the ID is treated as a database-specific ID instead of application ID.

        Raises:
            ObjectNotFoundException: If no entity with the given ID exists.
        """
        # Determine the field to filter by.
        field = self.db_id_field if use_db_id else self.app_id_field

        # Construct filter options with a limit of 1.
        filter_opts = QueryOptions(
            expression={field: {"operator": "eq", "value": identifier}},
            limit=1,
            offset=0,
        )

        # Call delete_many and expect exactly one deletion.
        count_deleted = await self.delete_many(filter_opts, logger, timeout)

        if count_deleted == 0:
            raise ObjectNotFoundException(
                f"{self.entity_type.__name__} with ID {identifier} not found"
            )
        elif count_deleted > 1:
            logger.warning(
                f"delete_one: More than one document deleted for ID {identifier}"
            )

    @abstractmethod
    async def delete_many(
        self,
        filter_options: QueryOptions,
        logger: LoggerAdapter,
        timeout: Optional[float] = None,
    ) -> int:
        """
        Delete all entities matching the provided filter options.

        Args:
            filter_options: QueryOptions instance containing a DSL expression for filtering the entities to delete.
            logger: Logger adapter for recording operations.
            timeout: Optional timeout for the operation.

        Returns:
            The number of entities deleted.
        """
        pass

    @abstractmethod
    async def list(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> AsyncGenerator[T, None]:
        """
        List entities matching the provided options.

        Args:
            logger: Logger adapter for recording operations.
            options: Query options for filtering, sorting, and pagination.

        Yields:
            Entities matching the query options.
        """
        yield

    @abstractmethod
    async def count(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> int:
        """
        Count entities matching the provided options.

        Args:
            logger: Logger adapter for recording operations.
            options: Query options for filtering.

        Returns:
            The number of entities matching the query options.
        """
        pass

    # Helper method to validate an entity has required fields.
    def validate_entity(self, entity: T) -> None:
        """
        Validate that an entity is valid for this repository.

        Args:
            entity: The entity to validate.

        Raises:
            ValueError: If the entity is invalid.
        """
        if not isinstance(entity, self.entity_type):
            raise ValueError(
                f"Entity must be of type {self.entity_type.__name__}, but is of type: {type(entity)}"
            )

    async def find_one(
        self, logger: LoggerAdapter, options: Optional[QueryOptions] = None
    ) -> T:
        """
        Find a single entity matching the provided query options.

        This method modifies the given QueryOptions (or creates new ones if none
        are provided) so that at most one entity is retrieved. It then uses the
        list() method to iterate over matching entities and returns the first one found.
        If no matching entity is found, it raises an ObjectNotFoundException.

        Args:
            logger: Logger adapter for recording operations.
            options: Optional QueryOptions for filtering, sorting, and pagination.

        Returns:
            The first entity matching the query options.

        Raises:
            ObjectNotFoundException: If no entity is found matching the criteria.
        """
        # Create or modify the options to limit the query to 1 result.
        if options is None:
            options = QueryOptions(limit=1, offset=0)
        else:
            options.limit = 1

        async for entity in self.list(logger, options):
            return entity

        raise ObjectNotFoundException(
            f"{self.entity_type.__name__} not found with the provided criteria"
        )
