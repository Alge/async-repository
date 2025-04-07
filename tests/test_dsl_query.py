import logging
import pytest
from async_repository.base.query import QueryOptions, QueryBuilder
from tests.conftest import Entity
from tests.conftest import REPOSITORY_IMPLEMENTATIONS


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_overloading_query(repository_factory, logger):
    """
    Test a DSL query built via operator overloading using the Entity class.

    Query built:
      (((qb.value > 21) & (qb.active == True)) | (qb.name.like("%Alice%"))) & (qb.owner == "org1")

    Expected: Only entities with owner "org1" that either have value > 21 and active True,
              or have a name matching "%Alice%".
    """
    repo = repository_factory(Entity)
    qb = QueryBuilder(Entity)

    # Create test entities:
    # e1: value 25 (>21), active True, name "Bob", owner "org1"
    e1 = Entity(value=25, active=True, name="Bob", owner="org1")
    # e2: value 20 (<21), active True, name "Alice", owner "org1" (matches by name)
    e2 = Entity(value=20, active=True, name="Alice", owner="org1")
    # e3: value 30 (>21), active False, name "Alice", owner "org1" (matches by name)
    e3 = Entity(value=30, active=False, name="Alice", owner="org1")
    # e4: Should not match because owner is "org2"
    e4 = Entity(value=30, active=True, name="Bob", owner="org2")

    for ent in [e1, e2, e3, e4]:
        await repo.store(ent, logger)

    # Build DSL expression using operator overloading:
    # ((qb.value > 21) & (qb.active == True)) OR (qb.name.like("%Alice%"))
    # Then AND with (qb.owner == "org1")
    expr = (((qb.value > 21) & (qb.active == True)) | (qb.name.like("%Alice%"))) & (
        qb.owner == "org1"
    )

    options = qb.filter(expr).build()

    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)

    returned_ids = {item.id for item in listed}
    expected_ids = {e1.id, e2.id, e3.id}

    logging.info("Found IDs: %s", returned_ids)
    logging.info("")
    assert (
        returned_ids == expected_ids
    ), f"Expected IDs {expected_ids}, got {returned_ids}"

    count = await repo.count(logger, options)
    assert count == len(
        listed
    ), f"Count {count} does not match listed length {len(listed)}"


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_filter_equal(repository_factory, logger):
    """Test DSL filter: retrieve only entities where name equals 'Alice'."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Bob"), Entity(name="Alice")]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Alice"}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert all(item.name == "Alice" for item in listed)
    count = await repo.count(logger, options)
    assert count == len(listed)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_complex_filter(repository_factory, logger):
    """
    Test DSL complex filter: ((value > 100 AND value < 200) OR (name eq 'Alice')).
    Expect entities with name 'Alice' or with value between 100 and 200.
    """
    repo = repository_factory(Entity)
    for ent in [
        Entity(name="Alice", value=50),
        Entity(name="Bob", value=150),
        Entity(name="Charlie", value=250),
    ]:
        await repo.store(ent, logger)
    expr = {
        "or": [
            {
                "and": [
                    {"value": {"operator": "gt", "value": 100}},
                    {"value": {"operator": "lt", "value": 200}},
                ]
            },
            {"name": {"operator": "eq", "value": "Alice"}},
        ]
    }
    options = QueryOptions(expression=expr, limit=100, offset=0)
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    names = {item.name for item in listed}
    assert names == {"Alice", "Bob"}
    count = await repo.count(logger, options)
    assert count == len(listed)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_string_filters(repository_factory, logger):
    """Test DSL string filters for 'startswith', 'endswith', and 'like'."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alpha"), Entity(name="Beta"), Entity(name="Gamma")]:
        await repo.store(ent, logger)
    # Test 'startswith'
    options = QueryOptions(
        expression={"name": {"operator": "startswith", "value": "Al"}},
        limit=100,
        offset=0,
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 1
    assert listed[0].name == "Alpha"
    # Test 'endswith'
    options = QueryOptions(
        expression={"name": {"operator": "endswith", "value": "ta"}},
        limit=100,
        offset=0,
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 1
    assert listed[0].name == "Beta"
    # Test 'like' (substring search)
    options = QueryOptions(
        expression={"name": {"operator": "like", "value": "am"}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 1
    assert listed[0].name == "Gamma"


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_array_contains(repository_factory, logger):
    """Test DSL operator 'contains': retrieve entities where tags array contains a specific value."""
    repo = repository_factory(Entity)
    for ent in [
        Entity(name="Alice", tags=["user", "admin"]),
        Entity(name="Bob", tags=["user"]),
        Entity(name="Charlie", tags=["guest"]),
    ]:
        await repo.store(ent, logger)

    # Test that 'contains' checks if a value is in a list
    options = QueryOptions(
        expression={"tags": {"operator": "contains", "value": "admin"}},
        limit=100,
        offset=0,
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert len(listed) == 1
    assert listed[0].name == "Alice"

    # Test another value
    options = QueryOptions(
        expression={"tags": {"operator": "contains", "value": "user"}},
        limit=100,
        offset=0,
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    names = {item.name for item in listed}
    assert names == {"Alice", "Bob"}


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_sorting(repository_factory, logger):
    """Test that sorting via QueryOptions works correctly using DSL."""
    repo = repository_factory(Entity)
    for ent in [
        Entity(name="A", value=300),
        Entity(name="B", value=100),
        Entity(name="C", value=200),
    ]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={}, sort_by="value", sort_desc=False, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    values = [item.value for item in listed]
    assert values == sorted(values)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_gt(repository_factory, logger):
    """Test DSL operator 'gt': retrieve entities where value > 100."""
    repo = repository_factory(Entity)
    for ent in [Entity(value=50), Entity(value=150), Entity(value=250)]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"value": {"operator": "gt", "value": 100}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    assert all(item.value > 100 for item in listed)
    count = await repo.count(logger, options)
    assert count == len(listed)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_in(repository_factory, logger):
    """Test DSL operator 'in': retrieve entities where name is in a given list."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Bob"), Entity(name="Charlie")]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"name": {"operator": "in", "value": ["Alice", "Charlie"]}},
        limit=100,
        offset=0,
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    names = {item.name for item in listed}
    assert names == {"Alice", "Charlie"}


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_nin(repository_factory, logger):
    """Test DSL operator 'nin': retrieve entities where name is not in a given list."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Bob"), Entity(name="Charlie")]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"name": {"operator": "nin", "value": ["Bob"]}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    names = {item.name for item in listed}
    assert names == {"Alice", "Charlie"}


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_exists(repository_factory, logger):
    """Test DSL operator 'exists': retrieve entities where owner exists (is not null)."""
    repo = repository_factory(Entity)
    for ent in [Entity(owner="org1"), Entity(owner=None), Entity(owner="org2")]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"owner": {"operator": "exists", "value": True}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    owners = {item.owner for item in listed if item.owner is not None}
    assert owners == {"org1", "org2"}


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_operator_regex(repository_factory, logger):
    """Test DSL operator 'regex': retrieve entities whose name matches a given regex."""
    repo = repository_factory(Entity)
    for ent in [Entity(name="Alice"), Entity(name="Alicia"), Entity(name="Bob")]:
        await repo.store(ent, logger)
    options = QueryOptions(
        expression={"name": {"operator": "regex", "value": "^Ali"}}, limit=100, offset=0
    )
    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)
    names = {item.name for item in listed}
    assert names == {"Alice", "Alicia"}


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_dsl_array_contains_with_builder(repository_factory, logger):
    """Test using QueryBuilder with 'contains' operator on array fields."""
    repo = repository_factory(Entity)
    qb = QueryBuilder(Entity)

    for ent in [
        Entity(name="Alice", tags=["user", "admin"]),
        Entity(name="Bob", tags=["user"]),
        Entity(name="Charlie", tags=["guest"]),
    ]:
        await repo.store(ent, logger)

    # Use QueryBuilder to build a 'contains' expression
    expr = qb.tags.contains("admin")
    options = qb.filter(expr).build()

    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)

    assert len(listed) == 1
    assert listed[0].name == "Alice"

    # Test simple query for users with the "user" tag
    qb = QueryBuilder(Entity)  # Reset query builder
    expr = qb.tags.contains("user")
    options = qb.filter(expr).build()

    listed = []
    async for item in repo.list(logger, options):
        listed.append(item)

    assert len(listed) == 2
    names = {item.name for item in listed}
    assert names == {"Alice", "Bob"}
