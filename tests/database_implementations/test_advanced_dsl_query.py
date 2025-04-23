# tests/database_implementations/test_advanced_dsl_query.py

import pytest

# Import QueryBuilder directly
from async_repository.base.query import QueryBuilder, QueryOptions, QueryFilter
from tests.conftest import Entity # Use Entity from conftest

# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.usefixtures("initialized_repository")


async def test_dsl_deeply_nested_expression(initialized_repository, logger):
    """
    Test a deeply nested expression using QueryBuilder:
    ((name == "Alice") | (value > 150)) & (active == True)
    Expected: Active entities that either have name 'Alice' or value > 150.
    """
    repo = initialized_repository
    e1 = Entity(name="Alice", value=100, active=True) # Matches (name + active)
    e2 = Entity(name="Bob", value=200, active=True) # Matches (value + active)
    e3 = Entity(name="Charlie", value=100, active=True) # No match
    e4 = Entity(name="Alice", value=50, active=False) # Matches name, but not active
    e5 = Entity(name="Dave", value=180, active=False) # Matches value, but not active
    for ent in [e1, e2, e3, e4, e5]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    expr = ((qb.fields.name == "Alice") | (qb.fields.value > 150)) & (qb.fields.active == True)
    options = qb.filter(expr).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    expected_ids = {e1.id, e2.id}
    returned_ids = {item.id for item in listed}

    assert len(listed) == 2
    assert returned_ids == expected_ids


# This test might not be valid anymore if the translation layer now handles
# unknown operators gracefully by raising ValueError during translation/building.
# If QueryBuilder itself prevents invalid operators, this test case changes.
# Assuming the translator raises ValueError:
async def test_dsl_invalid_operator_in_translation(initialized_repository, logger):
    """
    Test that using an expression containing an operator unknown to the
    backend's translator raises an error during query execution (like list/count).
    Note: QueryBuilder might allow creating the condition, but execution fails.
    """
    repo = initialized_repository
    await repo.store(Entity(name="Alice"), logger)

    # Manually create QueryOptions with a bad operator string if QueryBuilder prevents it
    # This simulates a potentially corrupted or manually crafted QueryOptions
    bad_expr = QueryFilter(field_path="name", operator="foo", value="Alice") # type: ignore - Deliberately using bad operator
    options = QueryOptions(expression=bad_expr)

    # The error should occur when the backend tries to translate "foo"
    with pytest.raises(ValueError, match="Unsupported query operator"):
        # Need to consume the generator to trigger execution
        [_ async for _ in repo.list(logger, options)]

    with pytest.raises(ValueError, match="Unsupported query operator"):
        await repo.count(logger, options)


async def test_dsl_no_match(initialized_repository, logger):
    """Test expression matching no entities using QueryBuilder."""
    repo = initialized_repository
    for ent in [Entity(name="Alice"), Entity(name="Bob")]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "NonExistent").limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    assert len(listed) == 0
    count = await repo.count(logger, options)
    assert count == 0


async def test_dsl_in_empty(initialized_repository, logger):
    """Test 'in_' operator with an empty list using QueryBuilder."""
    repo = initialized_repository
    for ent in [Entity(name="Alice"), Entity(name="Bob")]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name.in_([])).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    assert len(listed) == 0
    count = await repo.count(logger, options)
    assert count == 0