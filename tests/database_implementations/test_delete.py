# tests/database_implementations/test_delete.py

import pytest

from async_repository.base.exceptions import ObjectNotFoundException
from async_repository.base.query import QueryBuilder, QueryOptions
from tests.conftest import Entity, REPOSITORY_IMPLEMENTATIONS

# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)


# =============================================================================
# Tests for delete_one method
# =============================================================================


@pytest.mark.asyncio
async def test_delete_entity(initialized_repository, test_entity, logger):
    """Test deleting an existing entity using delete_one."""
    repo = initialized_repository
    await repo.store(test_entity, logger)
    # Verify it exists before delete
    assert await repo.get(test_entity.id, logger) is not None

    await repo.delete_one(test_entity.id, logger)

    # Verify it's gone
    with pytest.raises(ObjectNotFoundException):
        await repo.get(test_entity.id, logger)


@pytest.mark.asyncio
async def test_delete_non_existent(initialized_repository, logger):
    """Test deleting a non-existent entity raises ObjectNotFoundException."""
    repo = initialized_repository
    non_existent_id = "non-existent-id-for-delete"
    with pytest.raises(ObjectNotFoundException):
        await repo.delete_one(non_existent_id, logger)


@pytest.mark.asyncio
async def test_delete_one_by_db_id(initialized_repository, test_entity, logger):
    """Test deleting a single entity by database ID using delete_one."""
    repo = initialized_repository
    # Ensure the entity is stored and we can get its db_id
    # Note: return_value=True might not guarantee db_id is set on the object
    # depending on implementation. Re-fetch might be needed in some cases.
    await repo.store(test_entity, logger)
    stored = await repo.get(test_entity.id, logger) # Fetch to ensure we have the ID state

    # Get the DB ID (implementation specific attribute access)
    db_id = None
    if hasattr(stored, repo.db_id_field):
        db_id = getattr(stored, repo.db_id_field)
    elif repo.db_id_field == repo.app_id_field: # If they are the same
         db_id = stored.id

    if db_id is None:
        pytest.skip(
            f"Could not retrieve DB ID ('{repo.db_id_field}') for entity in this implementation."
        )

    # Delete the entity by DB ID
    await repo.delete_one(str(db_id), logger, use_db_id=True) # Ensure ID is string

    # Verify entity no longer exists using app ID
    with pytest.raises(ObjectNotFoundException):
        await repo.get(test_entity.id, logger)


# =============================================================================
# Tests for delete_many method
# =============================================================================
@pytest.mark.asyncio
async def test_delete_many_simple(initialized_repository, logger):
    """Test deleting multiple entities matching criteria."""
    repo = initialized_repository
    e1 = Entity(name="Delete1", owner="ToDelete")
    e2 = Entity(name="Delete2", owner="ToDelete")
    e3 = Entity(name="Keep", owner="Keep")
    for e in [e1, e2, e3]:
        await repo.store(e, logger)

    # Delete all entities with owner="ToDelete" using QueryBuilder
    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.owner == "ToDelete").build()

    delete_count = await repo.delete_many(options, logger)
    assert delete_count == 2

    # Verify only "Keep" entity remains
    all_entities = [item async for item in repo.list(logger)]
    assert len(all_entities) == 1
    assert all_entities[0].owner == "Keep"
    assert all_entities[0].name == "Keep"


# Note: delete_many with limit/offset/sort is often not directly supported
# by backend DELETE commands. These tests might be unreliable across DBs
# unless the implementation specifically fetches IDs first. Skipping them
# is safer for general cross-backend testing.

# @pytest.mark.asyncio
# async def test_delete_many_with_limit(...): pytest.skip(...)
# @pytest.mark.asyncio
# async def test_delete_many_with_offset(...): pytest.skip(...)


@pytest.mark.asyncio
async def test_delete_many_with_complex_expression(initialized_repository, logger):
    """Test delete_many with a complex expression using QueryBuilder."""
    repo = initialized_repository
    e1 = Entity(name="Alice", value=100, active=True, owner="X") # Keep
    e2 = Entity(name="Bob", value=200, active=True, owner="Y") # Keep
    e3 = Entity(name="Charlie", value=150, active=False, owner="X") # Delete (active=False)
    e4 = Entity(name="Dave", value=250, active=True, owner="Y") # Delete (value>200, owner="Y")
    e5 = Entity(name="Eve", value=300, active=False, owner="X") # Delete (active=False)
    for e in [e1, e2, e3, e4, e5]:
        await repo.store(e, logger)

    # Complex expression: (active=False) OR (value > 200 AND owner="Y")
    qb = QueryBuilder(Entity)
    expr = (qb.fields.active == False) | (
        (qb.fields.value > 200) & (qb.fields.owner == "Y")
    )
    options = qb.filter(expr).build()

    delete_count = await repo.delete_many(options, logger)
    assert delete_count == 3

    # Verify only Alice and Bob remain
    remaining_entities = [item async for item in repo.list(logger)]
    remaining_names = {item.name for item in remaining_entities}
    assert len(remaining_entities) == 2
    assert remaining_names == {"Alice", "Bob"}


@pytest.mark.asyncio
async def test_delete_many_with_no_matches(initialized_repository, logger):
    """Test delete_many when no entities match the expression."""
    repo = initialized_repository
    e1 = Entity(name="Alice")
    e2 = Entity(name="Bob")
    for e in [e1, e2]:
        await repo.store(e, logger)

    # Try to delete entities with a non-existent name
    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "NonExistent").build()

    delete_count = await repo.delete_many(options, logger)
    assert delete_count == 0

    # Verify all original entities still exist
    count = await repo.count(logger)
    assert count == 2


@pytest.mark.asyncio
async def test_delete_many_requires_expression(initialized_repository, logger):
    """Test that delete_many raises ValueError when no expression is provided."""
    repo = initialized_repository
    await repo.store(Entity(), logger) # Store one entity

    options = QueryOptions()  # No expression
    with pytest.raises(ValueError, match="must include an 'expression'"):
        await repo.delete_many(options, logger)