import pytest
from repositories.base.exceptions import ObjectNotFoundException
from repositories.base.query import QueryOptions
from tests.conftest import Entity
from tests.testlib import REPOSITORY_IMPLEMENTATIONS


# =============================================================================
# Tests for delete_one method
# =============================================================================

@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_entity(repository_factory, test_entity, logger):
    """Test deleting an existing entity."""
    repo = repository_factory(type(test_entity))
    await repo.store(test_entity, logger)
    await repo.delete_one(test_entity.id, logger)
    with pytest.raises(ObjectNotFoundException):
        await repo.get(test_entity.id, logger)


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_non_existent(repository_factory, test_entity, logger):
    """Test deleting a non-existent entity raises an exception."""
    repo = repository_factory(type(test_entity))
    non_existent_id = "non-existent-id"
    with pytest.raises(ObjectNotFoundException):
        await repo.delete_one(non_existent_id, logger)


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_one_by_db_id(repository_factory, test_entity, logger):
    """
    Test deleting a single entity by database ID using delete_one.
    """
    repo = repository_factory(type(test_entity))
    stored = await repo.store(test_entity, logger, return_value=True)

    # Get the DB ID (implementation specific)
    if hasattr(stored, '_id'):
        db_id = getattr(stored, '_id')

        # Delete the entity by DB ID
        await repo.delete_one(db_id, logger, use_db_id=True)

        # Verify entity no longer exists
        with pytest.raises(ObjectNotFoundException):
            await repo.get(test_entity.id, logger)
    else:
        # If DB ID is not directly accessible, we'll skip this test
        pytest.skip("DB ID not directly accessible in this implementation")


# =============================================================================
# Tests for delete_many method
# =============================================================================
@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_simple(repository_factory, logger):
    """
    Test deleting multiple entities with delete_many.
    """
    repo = repository_factory(Entity)
    # Create entities with the same owner
    entities = [
        Entity(name="Delete1", owner="ToDelete"),
        Entity(name="Delete2", owner="ToDelete"),
        Entity(name="Keep", owner="Keep")
    ]
    for e in entities:
        await repo.store(e, logger)

    # Delete all entities with owner="ToDelete"
    options = QueryOptions(expression={"owner": {"operator": "eq", "value": "ToDelete"}})
    delete_count = await repo.delete_many(options, logger)

    # Verify delete count
    assert delete_count == 2

    # Verify "ToDelete" entities no longer exist
    all_entities = []
    async for entity in repo.list(logger):
        all_entities.append(entity)

    assert len(all_entities) == 1
    assert all_entities[0].owner == "Keep"


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_with_limit(repository_factory, logger):
    """
    Test delete_many with a limit to delete only a subset of matching entities.
    """
    repo = repository_factory(Entity)
    # Create several entities with the same property
    entities = [
        Entity(name="ToDelete", value=100),
        Entity(name="ToDelete", value=200),
        Entity(name="ToDelete", value=300)
    ]
    for e in entities:
        await repo.store(e, logger)

    # Delete entities with name="ToDelete" but limit to 2, sorted by value
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "ToDelete"}},
        limit=2,
        sort_by="value",
        sort_desc=False  # Sort ascending by value, so the 2 smallest values should be deleted
    )

    delete_count = await repo.delete_many(options, logger)

    # Verify delete count
    assert delete_count == 2

    # Verify only the entity with value 300 remains
    remaining = []
    async for entity in repo.list(logger):
        remaining.append(entity)

    assert len(remaining) == 1
    assert remaining[0].value == 300


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_with_offset(repository_factory, logger):
    """
    Test delete_many with offset to skip the first N matching entities.
    """
    repo = repository_factory(Entity)
    # Create several entities for testing
    entities = [
        Entity(name="OffsetDelete", value=10),
        Entity(name="OffsetDelete", value=20),
        Entity(name="OffsetDelete", value=30),
        Entity(name="OffsetDelete", value=40)
    ]
    for e in entities:
        await repo.store(e, logger)

    # Delete OffsetDelete entities, but skip the first 2 when sorted by value
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "OffsetDelete"}},
        offset=2,
        sort_by="value",
        sort_desc=False  # Sort ascending, so skip values 10 and 20
    )

    delete_count = await repo.delete_many(options, logger)

    # Verify the correct number of entities were deleted
    assert delete_count == 2

    # Verify only entities with values 10 and 20 remain
    remaining_values = []
    async for entity in repo.list(logger):
        remaining_values.append(entity.value)

    assert sorted(remaining_values) == [10, 20]


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_with_complex_expression(repository_factory, logger):
    """
    Test delete_many with a complex expression involving AND, OR conditions.
    """
    repo = repository_factory(Entity)
    # Create test entities with various properties
    entities = [
        Entity(name="Alice", value=100, active=True, owner="X"),
        Entity(name="Bob", value=200, active=True, owner="Y"),
        Entity(name="Charlie", value=150, active=False, owner="X"),
        Entity(name="Dave", value=250, active=True, owner="Y"),
        Entity(name="Eve", value=300, active=False, owner="X")
    ]
    for e in entities:
        await repo.store(e, logger)

    # Complex expression: (active=False) OR (value > 200 AND owner="Y")
    expr = {
        "or": [
            {"active": {"operator": "eq", "value": False}},
            {"and": [
                {"value": {"operator": "gt", "value": 200}},
                {"owner": {"operator": "eq", "value": "Y"}}
            ]}
        ]
    }
    options = QueryOptions(expression=expr)

    # Delete all matching entities
    delete_count = await repo.delete_many(options, logger)

    # Should match: Charlie (active=False), Eve (active=False), Dave (value>200, owner="Y")
    assert delete_count == 3

    # Verify only Alice and Bob remain
    remaining_names = []
    async for entity in repo.list(logger):
        remaining_names.append(entity.name)

    assert sorted(remaining_names) == ["Alice", "Bob"]


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_with_no_matches(repository_factory, logger):
    """
    Test delete_many when no entities match the expression.
    """
    repo = repository_factory(Entity)
    # Create some test entities
    entities = [Entity(name="Alice"), Entity(name="Bob")]
    for e in entities:
        await repo.store(e, logger)

    # Try to delete entities with a non-existent name
    options = QueryOptions(expression={"name": {"operator": "eq", "value": "NonExistent"}})
    delete_count = await repo.delete_many(options, logger)

    # Should not have deleted any entities
    assert delete_count == 0

    # Verify all original entities still exist
    count = await repo.count(logger)
    assert count == 2


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_delete_many_all_with_no_expression(repository_factory, logger):
    """
    Test that delete_many raises ValueError when no expression is provided.
    """
    repo = repository_factory(Entity)
    entities = [Entity(), Entity()]
    for e in entities:
        await repo.store(e, logger)

    # Try to delete with an empty expression
    options = QueryOptions()  # No expression
    with pytest.raises(ValueError, match="must include an 'expression'"):
        await repo.delete_many(options, logger)