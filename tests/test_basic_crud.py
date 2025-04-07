import pytest
from async_repository.base.exceptions import (
    KeyAlreadyExistsException,
    ObjectNotFoundException,
)
from async_repository.base.query import QueryOptions
from tests.conftest import Entity
from tests.conftest import REPOSITORY_IMPLEMENTATIONS


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_and_get(repository_factory, test_entity, logger):
    """Test storing and retrieving an entity."""
    repo = repository_factory(type(test_entity))
    await repo.store(test_entity, logger)
    retrieved = await repo.get(test_entity.id, logger)
    assert retrieved.id == test_entity.id
    assert retrieved.name == test_entity.name


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_duplicate_key_insert(repository_factory, test_entity, logger):
    """Test storing an entity twice triggers a duplicate key exception."""
    repo = repository_factory(type(test_entity))
    await repo.store(test_entity, logger)
    with pytest.raises(KeyAlreadyExistsException):
        await repo.store(test_entity, logger)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_return_value(repository_factory, test_entity, logger):
    """
    Test that store returns the stored entity when return_value is True,
    and returns None when return_value is False.
    """
    repo = repository_factory(type(test_entity))
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
async def test_list_entities(repository_factory, logger):
    """Test listing entities returns all stored entities."""
    repo = repository_factory(Entity)
    entities = [Entity(), Entity(), Entity()]
    for ent in entities:
        await repo.store(ent, logger)
    listed = []
    async for item in repo.list(logger):
        listed.append(item)
    stored_ids = {ent.id for ent in entities}
    listed_ids = {item.id for item in listed}
    assert stored_ids.issubset(listed_ids)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_list_entities_random_order(repository_factory, logger):
    """
    Test that listing entities with random ordering returns results
    in an order different from natural (non-random) ordering and that
    two random queries yield different orders.
    """
    repo = repository_factory(Entity)
    # Create a number of entities so that ordering differences are less likely to be coincidental.
    entities = [Entity() for _ in range(20)]
    for ent in entities:
        await repo.store(ent, logger)

    async def get_ids(options: QueryOptions) -> list:
        ids = []
        async for item in repo.list(logger, options):
            ids.append(item.id)
        return ids

    # Get natural ordering (non-random) for comparison.
    ordered_ids = await get_ids(QueryOptions(limit=20, offset=0, random_order=False))
    # Get two random orderings.
    random_ids_1 = await get_ids(QueryOptions(limit=20, offset=0, random_order=True))
    random_ids_2 = await get_ids(QueryOptions(limit=20, offset=0, random_order=True))

    # At least one random ordering should differ from the natural ordering.
    assert (
        ordered_ids != random_ids_1 or ordered_ids != random_ids_2
    ), "Random ordering should yield a different order than the natural ordering."
    # Two random orderings should also be different.
    assert (
        random_ids_1 != random_ids_2
    ), "Two random queries should yield different orders."


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_count_entities(repository_factory, logger):
    """Test counting entities matches the number of entities stored."""
    repo = repository_factory(Entity)
    initial_count = await repo.count(logger)
    entities = [Entity(), Entity(), Entity()]
    for ent in entities:
        await repo.store(ent, logger)
    count_after = await repo.count(logger)
    assert count_after == initial_count + len(entities)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_returns_existing_entity(repository_factory, logger):
    """
    Test that find_one returns an entity when one exists matching the criteria.
    """
    repo = repository_factory(Entity)
    # Create an entity with name "Alice" and value 100.
    alice = Entity(name="Alice", value=100)
    await repo.store(alice, logger)
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Alice"}}, limit=10, offset=0
    )
    found = await repo.find_one(logger, options)
    assert found.name == "Alice"
    assert found.value == 100


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_with_multiple_matches_returns_first(repository_factory, logger):
    """
    Test that when multiple entities match the criteria, find_one returns only one.
    """
    repo = repository_factory(Entity)
    # Create two entities with the same name "Bob" but different values.
    bob1 = Entity(name="Bob", value=200)
    bob2 = Entity(name="Bob", value=300)
    await repo.store(bob1, logger)
    await repo.store(bob2, logger)
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Bob"}}, limit=50, offset=0
    )
    found = await repo.find_one(logger, options)
    assert found.name == "Bob"
    assert found.value in (200, 300)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_raises_when_not_found(repository_factory, logger):
    """
    Test that find_one raises ObjectNotFoundException when no entity matches the criteria.
    """
    repo = repository_factory(Entity)
    # Store an entity with a different name.
    charlie = Entity(name="Charlie", value=400)
    await repo.store(charlie, logger)
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "NonExistent"}},
        limit=50,
        offset=0,
    )
    with pytest.raises(ObjectNotFoundException):
        await repo.find_one(logger, options)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
def test_validate_entity_valid(repository_factory, test_entity, logger):
    """Test that validate_entity accepts a valid entity."""
    repo = repository_factory(type(test_entity))
    repo.validate_entity(test_entity)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
def test_validate_entity_invalid(repository_factory, test_entity, logger):
    """Test that validate_entity raises ValueError when passed an invalid type."""
    repo = repository_factory(type(test_entity))
    with pytest.raises(ValueError):
        repo.validate_entity("not an entity")
