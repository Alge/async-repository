# tests/database_implementations/test_basic_crud.py
import pytest
from dataclasses import asdict
from async_repository.base.exceptions import (
    KeyAlreadyExistsException,
    ObjectNotFoundException,
)
# Make sure QueryBuilder is imported
from async_repository.base.query import QueryOptions, QueryBuilder
from tests.conftest import Entity, REPOSITORY_IMPLEMENTATIONS

# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.usefixtures("initialized_repository")


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_and_get(initialized_repository, test_entity, logger): # Changed fixture
    """Test storing and retrieving an entity."""
    repo = initialized_repository # Use the initialized repo
    await repo.store(test_entity, logger)
    retrieved = await repo.get(test_entity.id, logger)
    assert retrieved.id == test_entity.id
    assert retrieved.name == test_entity.name


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_duplicate_key_insert(initialized_repository, test_entity, logger): # Changed fixture
    """Test storing an entity twice triggers a duplicate key exception."""
    repo = initialized_repository # Use the initialized repo
    await repo.store(test_entity, logger)
    with pytest.raises(KeyAlreadyExistsException):
        await repo.store(test_entity, logger)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_return_value(initialized_repository, test_entity, logger): # Changed fixture
    """
    Test that store returns the stored entity when return_value is True,
    and returns None when return_value is False.
    """
    repo = initialized_repository # Use the initialized repo
    # Test with return_value True
    result = await repo.store(test_entity, logger, return_value=True)
    assert result is not None
    assert result.id == test_entity.id
    assert result.name == test_entity.name

    # For a different entity, verify that return_value False returns None.
    another_entity = type(test_entity)()  # Create a new entity instance
    result_none = await repo.store(another_entity, logger, return_value=False)
    assert result_none is None


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_list_entities(initialized_repository, logger): # Changed fixture
    """Test listing entities returns all stored entities."""
    repo = initialized_repository # Use the initialized repo
    entities = [Entity(), Entity(), Entity()]
    stored_ids = {ent.id for ent in entities} # Get ids before storing
    for ent in entities:
        await repo.store(ent, logger)

    listed = [item async for item in repo.list(logger)]
    listed_ids = {item.id for item in listed}

    # Check if all stored IDs are present in the listed IDs
    assert stored_ids.issubset(listed_ids)
    # Optionally, check if the count matches if no other entities exist
    assert len(listed) >= len(entities)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_list_entities_random_order(initialized_repository, logger, get_repo_type): # Changed fixture
    """
    Test that listing entities with random ordering returns results
    in an order different from natural (non-random) ordering and that
    two random queries yield different orders.
    NOTE: Skipped for mongomock as its $sample may not be truly random.
    """
    repo = initialized_repository # Use the initialized repo
    repo_type = get_repo_type

    # Create a number of entities
    num_entities = 20
    entities = [Entity() for _ in range(num_entities)]
    for ent in entities:
        await repo.store(ent, logger)

    async def get_ids(options: QueryOptions) -> list:
        ids = []
        async for item in repo.list(logger, options):
            ids.append(item.id)
        return ids

    # Get natural ordering (default sort or sort by ID if needed)
    qb = QueryBuilder(Entity)
    options_ordered = qb.limit(num_entities).sort_by(qb.fields.id).build() # Explicit sort
    options_random = qb.limit(num_entities).random_order().build()

    ordered_ids = await get_ids(options_ordered)
    assert len(ordered_ids) == num_entities, f"Expected {num_entities} ordered results"

    # Get two random orderings.
    random_ids_1 = await get_ids(options_random)
    random_ids_2 = await get_ids(options_random)

    assert len(random_ids_1) == num_entities, f"Expected {num_entities} random results (1)"
    assert len(random_ids_2) == num_entities, f"Expected {num_entities} random results (2)"

    # Assertions (only run if not skipped)
    assert (
        ordered_ids != random_ids_1 or ordered_ids != random_ids_2
    ), "Random ordering should yield a different order than the natural ordering."
    assert (
        random_ids_1 != random_ids_2
    ), "Two random queries should ideally yield different orders."


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_count_entities(initialized_repository, logger): # Changed fixture
    """Test counting entities matches the number of entities stored."""
    repo = initialized_repository # Use the initialized repo
    initial_count = await repo.count(logger)
    entities = [Entity(), Entity(), Entity()]
    for ent in entities:
        await repo.store(ent, logger)
    count_after = await repo.count(logger)
    assert count_after == initial_count + len(entities)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_returns_existing_entity(initialized_repository, logger): # Changed fixture
    """
    Test that find_one returns an entity when one exists matching the criteria.
    """
    repo = initialized_repository # Use the initialized repo
    alice = Entity(name="Alice", value=100)
    await repo.store(alice, logger)

    # Use QueryBuilder to create options
    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "Alice").limit(10).offset(0).build()

    found = await repo.find_one(logger, options)
    assert found is not None # find_one should not return None on success
    assert found.id == alice.id
    assert found.name == "Alice"
    assert found.value == 100


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_with_multiple_matches_returns_first(initialized_repository, logger): # Changed fixture
    """
    Test that when multiple entities match the criteria, find_one returns only one.
    """
    repo = initialized_repository # Use the initialized repo
    bob1 = Entity(name="Bob", value=200)
    bob2 = Entity(name="Bob", value=300)
    await repo.store(bob1, logger)
    await repo.store(bob2, logger)

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    # Add a sort to make the "first" predictable if needed, otherwise DB decides
    options = qb.filter(qb.fields.name == "Bob").sort_by(qb.fields.value).limit(50).build()

    found = await repo.find_one(logger, options)
    assert found is not None
    assert found.name == "Bob"
    # With sorting by value asc, it should be bob1
    assert found.id == bob1.id
    assert found.value == 200


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_raises_when_not_found(initialized_repository, logger): # Changed fixture
    """
    Test that find_one raises ObjectNotFoundException when no entity matches.
    """
    repo = initialized_repository # Use the initialized repo
    charlie = Entity(name="Charlie", value=400)
    await repo.store(charlie, logger)

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "NonExistent").build()

    with pytest.raises(ObjectNotFoundException):
        await repo.find_one(logger, options)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
def test_validate_entity_valid(initialized_repository, test_entity): # Changed fixture
    """Test that validate_entity accepts a valid entity."""
    repo = initialized_repository # Use the initialized repo
    # No logger needed for validate_entity
    repo.validate_entity(test_entity)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
def test_validate_entity_invalid(initialized_repository): # Changed fixture
    """Test that validate_entity raises ValueError when passed an invalid type."""
    repo = initialized_repository # Use the initialized repo
    # No logger needed
    with pytest.raises(ValueError, match="Entity must be of type Entity"):
        repo.validate_entity("not an entity") # Pass string