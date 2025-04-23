# tests/database_implementations/test_list_contains.py

import pytest

# Import QueryBuilder and Entity/ProfileData from conftest
from async_repository.base.query import QueryBuilder
from tests.conftest import (Entity, ProfileData, REPOSITORY_IMPLEMENTATIONS,
                            initialized_repository, logger)

# Apply initialized_repository and parametrization to all tests in this module
pytestmark = [
    pytest.mark.parametrize(
        "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
    ),
    pytest.mark.usefixtures("initialized_repository"),
]


@pytest.mark.asyncio
async def test_contains_on_direct_list_match(initialized_repository, logger):
    """Test 'contains' finds entities where a direct list field contains the value."""
    repo = initialized_repository
    e1 = Entity(name="Entity A", tags=["A", "B", "C"])
    e2 = Entity(name="Entity B", tags=["B", "D"])
    e3 = Entity(name="Entity C", tags=["A", "D"])
    await repo.store(e1, logger)
    await repo.store(e2, logger)
    await repo.store(e3, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.tags.contains("B")).build()

    found_entities = [item async for item in repo.list(logger, options)]
    found_ids = {e.id for e in found_entities}
    expected_ids = {e1.id, e2.id}

    assert len(found_entities) == 2
    assert found_ids == expected_ids


@pytest.mark.asyncio
async def test_contains_on_direct_list_no_match(initialized_repository, logger):
    """Test 'contains' doesn't find entities when the value is not in the direct list."""
    repo = initialized_repository
    e1 = Entity(name="Entity A", tags=["A", "C"])
    e2 = Entity(name="Entity B", tags=["D", "E"])
    await repo.store(e1, logger)
    await repo.store(e2, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.tags.contains("X")).build() # "X" is not present

    found_entities = [item async for item in repo.list(logger, options)]

    assert len(found_entities) == 0


@pytest.mark.asyncio
async def test_contains_on_empty_direct_list(initialized_repository, logger):
    """Test 'contains' doesn't match when the direct list field is empty."""
    repo = initialized_repository
    e1 = Entity(name="Entity Empty", tags=[])
    await repo.store(e1, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.tags.contains("A")).build()

    found_entities = [item async for item in repo.list(logger, options)]

    assert len(found_entities) == 0


@pytest.mark.asyncio
async def test_contains_on_nested_list_match(initialized_repository, logger):
    """Test 'contains' finds entities where a nested list field contains the value."""
    repo = initialized_repository
    e1 = Entity(name="Profile A", profile=ProfileData(emails=["a@x.com", "b@x.com"]))
    e2 = Entity(name="Profile B", profile=ProfileData(emails=["c@x.com"]))
    e3 = Entity(name="Profile C", profile=ProfileData(emails=["b@x.com", "d@x.com"]))
    await repo.store(e1, logger)
    await repo.store(e2, logger)
    await repo.store(e3, logger)

    qb = QueryBuilder(Entity)
    # Use dot notation to access nested list
    options = qb.filter(qb.fields.profile.emails.contains("b@x.com")).build()

    found_entities = [item async for item in repo.list(logger, options)]
    found_ids = {e.id for e in found_entities}
    expected_ids = {e1.id, e3.id}

    assert len(found_entities) == 2
    assert found_ids == expected_ids


@pytest.mark.asyncio
async def test_contains_on_nested_list_no_match(initialized_repository, logger):
    """Test 'contains' doesn't find entities when the value is not in the nested list."""
    repo = initialized_repository
    e1 = Entity(name="Profile A", profile=ProfileData(emails=["a@x.com"]))
    e2 = Entity(name="Profile B", profile=ProfileData(emails=["c@x.com"]))
    await repo.store(e1, logger)
    await repo.store(e2, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.profile.emails.contains("not_found@x.com")).build()

    found_entities = [item async for item in repo.list(logger, options)]

    assert len(found_entities) == 0


@pytest.mark.asyncio
async def test_contains_on_empty_nested_list(initialized_repository, logger):
    """Test 'contains' doesn't match when the nested list field is empty."""
    repo = initialized_repository
    # ProfileData defaults emails to []
    e1 = Entity(name="Profile Empty", profile=ProfileData())
    await repo.store(e1, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.profile.emails.contains("a@x.com")).build()

    found_entities = [item async for item in repo.list(logger, options)]

    assert len(found_entities) == 0