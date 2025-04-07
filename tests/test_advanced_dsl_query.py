import pytest
from repositories.base.exceptions import ObjectNotFoundException
from repositories.base.query import QueryOptions
from tests.conftest import Entity
from tests.testlib import REPOSITORY_IMPLEMENTATIONS


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_dsl_deeply_nested_expression(repository_factory, logger):
    """
    Test a deeply nested DSL expression:
      {
        "and": [
          {"or": [
              {"name": {"operator": "eq", "value": "Alice"}},
              {"value": {"operator": "gt", "value": 150}}
           ]},
          {"active": {"operator": "eq", "value": True}}
        ]
      }
    Expected: Active entities that either have name 'Alice' or value > 150.
    """
    repo = repository_factory(Entity)
    e1 = Entity(name="Alice", value=100, active=True)
    e2 = Entity(name="Bob", value=200, active=True)
    e3 = Entity(name="Charlie", value=100, active=True)
    e4 = Entity(name="Alice", value=200, active=False)
    for ent in [e1, e2, e3, e4]:
        await repo.store(ent, logger)
    expr = {
        "and": [
            {"or": [
                {"name": {"operator": "eq", "value": "Alice"}},
                {"value": {"operator": "gt", "value": 150}}
            ]},
            {"active": {"operator": "eq", "value": True}}
        ]
    }
    options = QueryOptions(expression=expr, limit=100, offset=0)
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    expected_ids = {e1.id, e2.id}
    returned_ids = {item.id for item in listed}
    assert returned_ids == expected_ids


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_dsl_invalid_operator(repository_factory, logger):
    """
    Test DSL expression with an unsupported operator.
    For unsupported operators, the implementation should raise a ValueError.
    """
    repo = repository_factory(Entity)
    e1 = Entity(name="Alice")
    e2 = Entity(name="Bob")
    await repo.store(e1, logger)
    await repo.store(e2, logger)
    expr = {"name": {"operator": "foo", "value": "Alice"}}
    options = QueryOptions(expression=expr, limit=100, offset=0)
    with pytest.raises(ValueError, match="Unsupported operator: foo"):
        [item async for item in repo.list(logger, options)]


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_dsl_no_match(repository_factory, logger):
    """Test DSL expression that matches no entities."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Bob")]:
        await repo.store(ent, logger)
    expr = {"name": {"operator": "eq", "value": "NonExistent"}}
    options = QueryOptions(expression=expr, limit=100, offset=0)
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 0
    count = await repo.count(logger, options)
    assert count == 0


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_dsl_in_empty(repository_factory, logger):
    """
    Test DSL 'in' operator with an empty list.
    Expected: No entities match.
    """
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Bob")]:
        await repo.store(ent, logger)
    expr = {"name": {"operator": "in", "value": []}}
    options = QueryOptions(expression=expr, limit=100, offset=0)
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 0