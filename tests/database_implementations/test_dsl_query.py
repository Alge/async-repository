# tests/database_implementations/test_dsl_query.py

import logging
import pytest

# Import QueryBuilder and QueryOptions directly
from async_repository.base.query import QueryBuilder, QueryOptions
from tests.conftest import Entity # Use Entity from conftest

# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.usefixtures("initialized_repository")


async def test_dsl_filter_equal(initialized_repository, logger):
    """Test retrieve only entities where name equals 'Alice' using QueryBuilder."""
    repo = initialized_repository
    alice1 = Entity(name="Alice", value=100)
    bob = Entity(name="Bob", value=150)
    alice2 = Entity(name="Alice", value=200)
    for ent in [alice1, bob, alice2]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "Alice").limit(100).build()

    listed = [item async for item in repo.list(logger, options)]

    assert len(listed) == 2
    assert all(item.name == "Alice" for item in listed)
    # Verify count matches
    count = await repo.count(logger, options)
    assert count == 2


async def test_dsl_complex_filter(initialized_repository, logger):
    """
    Test complex filter: ((value > 100 AND value < 200) OR (name eq 'Alice')).
    Using QueryBuilder.
    """
    repo = initialized_repository
    alice = Entity(name="Alice", value=50) # Matches by name
    bob = Entity(name="Bob", value=150) # Matches by value range
    charlie = Entity(name="Charlie", value=250) # No match
    for ent in [alice, bob, charlie]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    expr = ((qb.fields.value > 100) & (qb.fields.value < 200)) | (qb.fields.name == "Alice")
    options = qb.filter(expr).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    names = {item.name for item in listed}

    assert len(listed) == 2
    assert names == {"Alice", "Bob"}
    count = await repo.count(logger, options)
    assert count == 2



@pytest.mark.asyncio
async def test_dsl_string_filters(initialized_repository, logger, get_repo_type): # Added get_repo_type
    """Test string filters 'startswith', 'endswith', 'like' using QueryBuilder."""
    repo = initialized_repository
    repo_type = get_repo_type # Get backend type

    alpha = Entity(name="Alpha")
    beta = Entity(name="Beta")
    gamma = Entity(name="Gamma")
    delta = Entity(name="Delta") # Also ends with 'ta'
    for ent in [alpha, beta, gamma, delta]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)

    # Test 'startswith'
    options_start = qb.filter(qb.fields.name.startswith("Al")).build()
    listed_start = [item async for item in repo.list(logger, options_start)]
    assert len(listed_start) == 1
    assert listed_start[0].name == "Alpha"

    # Test 'endswith'
    qb = QueryBuilder(Entity) # Reset
    options_end = qb.filter(qb.fields.name.endswith("ta")).build()
    listed_end = [item async for item in repo.list(logger, options_end)]

    # --- CORRECTED ASSERTION for endswith ---
    expected_names_end = {"Beta", "Delta"}
    actual_names_end = {e.name for e in listed_end}
    assert len(listed_end) == 2, f"Expected 2 ending with 'ta', got {actual_names_end}"
    assert actual_names_end == expected_names_end
    # --- End CORRECTED ASSERTION ---

    # Test 'like' (substring search) - using '%' wildcards (case-insensitive by default)
    qb = QueryBuilder(Entity) # Reset
    # Using %amm% should match Gamma (case-insensitive)
    options_like = qb.filter(qb.fields.name.like("%amm%")).build()
    listed_like = [item async for item in repo.list(logger, options_like)]
    assert len(listed_like) == 1
    assert listed_like[0].name == "Gamma"

    # Test 'contains' (case-insensitive by default in our translation)
    qb = QueryBuilder(Entity) # Reset
    options_contains = qb.filter(qb.fields.name.contains("elt")).build()
    listed_contains = [item async for item in repo.list(logger, options_contains)]
    assert len(listed_contains) == 1
    assert listed_contains[0].name == "Delta"


async def test_dsl_array_contains(initialized_repository, logger):
    """Test 'contains' on array field using QueryBuilder."""
    repo = initialized_repository
    alice = Entity(name="Alice", tags=["user", "admin"])
    bob = Entity(name="Bob", tags=["user"])
    charlie = Entity(name="Charlie", tags=["guest"])
    for ent in [alice, bob, charlie]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)

    # Test contains 'admin'
    options_admin = qb.filter(qb.fields.tags.contains("admin")).build()
    listed_admin = [item async for item in repo.list(logger, options_admin)]
    assert len(listed_admin) == 1
    assert listed_admin[0].name == "Alice"

    # Test contains 'user'
    qb = QueryBuilder(Entity) # Reset
    options_user = qb.filter(qb.fields.tags.contains("user")).build()
    listed_user = [item async for item in repo.list(logger, options_user)]
    names = {item.name for item in listed_user}
    assert len(listed_user) == 2
    assert names == {"Alice", "Bob"}


async def test_dsl_sorting(initialized_repository, logger):
    """Test sorting via QueryBuilder works correctly."""
    repo = initialized_repository
    e1 = Entity(name="A", value=300)
    e2 = Entity(name="B", value=100)
    e3 = Entity(name="C", value=200)
    for ent in [e1, e2, e3]:
        await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    # Sort ascending by value
    options_asc = qb.sort_by(qb.fields.value).limit(100).build()
    listed_asc = [item async for item in repo.list(logger, options_asc)]
    values_asc = [item.value for item in listed_asc]
    assert values_asc == [100, 200, 300]

    # Sort descending by value
    options_desc = qb.sort_by(qb.fields.value, descending=True).limit(100).build()
    listed_desc = [item async for item in repo.list(logger, options_desc)]
    values_desc = [item.value for item in listed_desc]
    assert values_desc == [300, 200, 100]


async def test_dsl_operator_gt(initialized_repository, logger):
    """Test operator '>' using QueryBuilder."""
    repo = initialized_repository
    e1 = Entity(value=50)
    e2 = Entity(value=150)
    e3 = Entity(value=250)
    for ent in [e1, e2, e3]: await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.value > 100).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    assert len(listed) == 2
    assert all(item.value > 100 for item in listed)
    count = await repo.count(logger, options)
    assert count == 2


async def test_dsl_operator_in(initialized_repository, logger):
    """Test operator 'in_' using QueryBuilder."""
    repo = initialized_repository
    alice = Entity(name="Alice")
    bob = Entity(name="Bob")
    charlie = Entity(name="Charlie")
    for ent in [alice, bob, charlie]: await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name.in_(["Alice", "Charlie"])).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    names = {item.name for item in listed}
    assert len(listed) == 2
    assert names == {"Alice", "Charlie"}


async def test_dsl_operator_nin(initialized_repository, logger):
    """Test operator 'nin' using QueryBuilder."""
    repo = initialized_repository
    alice = Entity(name="Alice")
    bob = Entity(name="Bob")
    charlie = Entity(name="Charlie")
    for ent in [alice, bob, charlie]: await repo.store(ent, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name.nin(["Bob"])).limit(100).build()

    listed = [item async for item in repo.list(logger, options)]
    names = {item.name for item in listed}
    assert len(listed) == 2
    assert names == {"Alice", "Charlie"}


async def test_dsl_operator_exists(initialized_repository, logger, get_repo_type):
    """Test operator 'exists' using QueryBuilder."""
    repo = initialized_repository
    repo_type = get_repo_type
    e1 = Entity(owner="org1")
    e2 = Entity(owner=None) # Explicitly None
    e3 = Entity(owner="org2")
    for ent in [e1, e2, e3]: await repo.store(ent, logger)

    qb = QueryBuilder(Entity)

    # Test exists=True (Should return e1 and e3)
    options_true = qb.filter(qb.fields.owner.exists(True)).limit(100).build()
    listed_true = [item async for item in repo.list(logger, options_true)]
    owners_true = {item.owner for item in listed_true}
    assert len(listed_true) == 2
    assert owners_true == {"org1", "org2"}

    # Test exists=False (Should return e2 where owner is explicitly None)
    qb = QueryBuilder(Entity) # Create a new builder instance
    options_false = qb.filter(qb.fields.owner.exists(False)).limit(100).build()

    listed_false = [item async for item in repo.list(logger, options_false)]

    # Expect exactly 1 result (the one explicitly set to None)
    assert len(listed_false) == 1, f"Expected 1 with owner=null, got {len(listed_false)}"
    if listed_false: assert listed_false[0].owner is None