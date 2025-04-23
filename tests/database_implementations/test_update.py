# tests/database_implementations/test_update.py

from datetime import datetime, timezone # Added timezone

import pytest
from pydantic import AnyHttpUrl, BaseModel, Field

from async_repository.base.exceptions import ObjectNotFoundException
# Import QueryBuilder to construct options where needed
from async_repository.base.query import QueryBuilder, QueryOptions
from async_repository.base.update import Update
from tests.conftest import Entity  # Use Entity from conftest


# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.usefixtures("initialized_repository")


# =============================================================================
# Tests for update_one method (using Update builder)
# =============================================================================

async def test_update_set_fields(initialized_repository, test_entity, logger):
    """Test updating fields using set."""
    repo = initialized_repository # Use the initialized repo
    await repo.store(test_entity, logger)

    new_name = "Updated Name"
    new_value = 200
    now = datetime.now(timezone.utc) # Use timezone-aware datetime

    # Use QueryBuilder for options
    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == test_entity.id).build()
    update = Update(Entity).set("name", new_name).set("value", new_value).set("updated_at", now)

    await repo.update_one(options, update, logger)
    updated = await repo.get(test_entity.id, logger)

    assert updated.name == new_name
    assert updated.value == new_value
    assert updated.updated_at is not None
    # Compare timestamps carefully (allow for slight difference)
    time_diff = abs((updated.updated_at - now).total_seconds())
    assert time_diff < 1.0


async def test_update_non_existent(initialized_repository, test_entity, logger):
    """Test updating a non-existent entity raises ObjectNotFoundException."""
    repo = initialized_repository
    non_existent_id = "non-existent-id"

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == non_existent_id).build()
    update = Update(Entity).set("name", "Should Fail")

    with pytest.raises(ObjectNotFoundException):
        await repo.update_one(options, update, logger)


async def test_update_return_value(initialized_repository, test_entity, logger):
    """Test update_one return value."""
    repo = initialized_repository
    await repo.store(test_entity, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == test_entity.id).build()
    update = Update(Entity).set("name", "Updated For Return").set("value", 300)

    # Test return_value=True
    result = await repo.update_one(options, update, logger, return_value=True)
    assert result is not None
    assert result.id == test_entity.id
    assert result.name == "Updated For Return"
    assert result.value == 300

    # Test return_value=False
    result_none = await repo.update_one(
        options, Update(Entity).set("value", 400), logger, return_value=False
    )
    assert result_none is None

    # Verify the second update still happened
    final = await repo.get(test_entity.id, logger)
    assert final.value == 400


async def test_update_one_with_complex_expression(initialized_repository, logger):
    """Test update_one with QueryBuilder expression."""
    repo = initialized_repository
    e1 = Entity(name="Alice", value=100, active=True)
    e2 = Entity(name="Bob", value=200, active=True)
    e3 = Entity(name="Alice", value=300, active=False) # Should not be updated
    for e in [e1, e2, e3]:
        await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    # Find active Alice
    options = qb.filter((qb.fields.active == True) & (qb.fields.name == "Alice")).build()
    update = Update(Entity).set("value", 150)

    await repo.update_one(options, update, logger)

    # Verify only e1 was updated
    updated_alice = await repo.get(e1.id, logger)
    assert updated_alice.value == 150
    unchanged_bob = await repo.get(e2.id, logger)
    assert unchanged_bob.value == 200
    inactive_alice = await repo.get(e3.id, logger)
    assert inactive_alice.value == 300


# =============================================================================
# Tests for update_many method (using Update builder)
# =============================================================================


@pytest.mark.asyncio
async def test_update_many_simple(initialized_repository, logger):
    """Test updating multiple entities with update_many."""
    repo = initialized_repository
    entities = [
        Entity(name="Target", value=100, active=False),
        Entity(name="Target", value=200, active=False),
        Entity(name="Other", value=300, active=False),
    ]
    for e in entities:
        await repo.store(e, logger)

    # --- Query/Update for 'Target' entities ---
    qb_target = QueryBuilder(Entity) # Use a specific builder instance
    options_target = qb_target.filter(qb_target.fields.name == "Target").build()
    update = Update(Entity).set("active", True)

    update_count = await repo.update_many(options_target, update, logger)
    assert update_count == 2

    # Verify changes using the same options object
    listed = [item async for item in repo.list(logger, options_target)]
    assert len(listed) == 2
    assert all(item.active is True for item in listed)

    # --- Query for 'Other' entity ---
    qb_other = QueryBuilder(Entity) # Create a NEW builder instance
    options_other = qb_other.filter(qb_other.fields.name == "Other").build()
    other_entity = await repo.find_one(logger, options_other) # Should now find the entity

    assert other_entity is not None
    assert other_entity.name == "Other"
    assert other_entity.active is False # Verify it wasn't updated

# Note: update_many with limit/offset is tricky and behaviour differs between DBs.
# SQLite/Postgres typically ignore limit/offset in UPDATE statements.
# MongoDB's update_many also ignores them.
# The previous tests for limit/offset might have relied on non-standard behaviour
# or the complex find-ids-then-update logic which was removed for MongoDB/SQLite.
# We will skip testing limit/offset with update_many for now, as it's generally
# not a reliable cross-database feature for the UPDATE command itself.
# If specific limited updates are needed, usually a fetch-then-update loop is used.
# async def test_update_many_with_limit(...): pytest.skip(...)
# async def test_update_many_with_offset(...): pytest.skip(...)


async def test_update_many_with_complex_expression(initialized_repository, logger):
    """Test update_many with a complex expression."""
    repo = initialized_repository
    entities = [
        Entity(name="Alice", value=100, active=True, owner="A", updated_at=None),
        Entity(name="Bob", value=200, active=True, owner="B", updated_at=None),
        Entity(name="Charlie", value=150, active=False, owner="A", updated_at=None),
        Entity(name="Dave", value=250, active=True, owner="B", updated_at=None),
        Entity(name="Eve", value=300, active=False, owner="A", updated_at=None),
    ]
    for e in entities:
        await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    # Update if (active=True AND owner="B") OR value > 200
    expr = ((qb.fields.active == True) & (qb.fields.owner == "B")) | (qb.fields.value > 200)
    options = qb.filter(expr).build()

    update_timestamp = datetime.now(timezone.utc)
    update = Update(Entity).set("updated_at", update_timestamp).set("value", 1000)

    update_count = await repo.update_many(options, update, logger)
    # Should match Bob (active, B), Dave (active, B), Eve (value > 200)
    assert update_count == 3

    # Verification
    all_entities = {e.name: e async for e in repo.list(logger)}
    assert all_entities["Bob"].value == 1000
    assert all_entities["Dave"].value == 1000
    assert all_entities["Eve"].value == 1000
    assert all_entities["Alice"].value == 100 # Unchanged
    assert all_entities["Charlie"].value == 150 # Unchanged

    assert all_entities["Bob"].updated_at is not None
    assert all_entities["Dave"].updated_at is not None
    assert all_entities["Eve"].updated_at is not None
    assert all_entities["Alice"].updated_at is None
    assert all_entities["Charlie"].updated_at is None


async def test_update_many_with_no_matches(initialized_repository, logger):
    """Test update_many when no entities match."""
    repo = initialized_repository
    entities = [Entity(name="Alice"), Entity(name="Bob")]
    for e in entities: await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "NonExistent").build()
    update = Update(Entity).set("active", False)

    update_count = await repo.update_many(options, update, logger)
    assert update_count == 0

    # Verify active status didn't change
    async for entity in repo.list(logger):
        assert entity.active is True


async def test_update_many_requires_expression(initialized_repository, logger):
    """Test update_many raises ValueError if no expression is provided."""
    repo = initialized_repository
    await repo.store(Entity(), logger) # Store one entity

    options = QueryOptions() # No expression
    update = Update(Entity).set("active", False)

    with pytest.raises(ValueError, match="must include an 'expression'"):
        await repo.update_many(options, update, logger)


# =============================================================================
# Tests for Upsert Method (Kept from previous file)
# =============================================================================

async def test_upsert_inserts_new_entity(initialized_repository, logger):
    """Test upsert inserts a new entity."""
    repo = initialized_repository
    entity = Entity(name="Upsert New", value=123)
    await repo.upsert(entity, logger)
    retrieved = await repo.get(entity.id, logger)
    assert retrieved.name == entity.name
    assert retrieved.value == entity.value


async def test_upsert_updates_existing_entity(initialized_repository, logger):
    """Test upsert updates an existing entity."""
    repo = initialized_repository
    entity = Entity(name="Upsert Update", value=100)
    await repo.store(entity, logger) # Store first

    entity.name = "Upsert Updated" # Modify
    entity.value = 250
    await repo.upsert(entity, logger) # Upsert modified

    retrieved = await repo.get(entity.id, logger)
    assert retrieved.name == "Upsert Updated"
    assert retrieved.value == 250


# =============================================================================
# Tests for push, pop, unset, pull, inc, dec, min, max, mul operations
# =============================================================================

async def test_update_push_operation(initialized_repository, logger):
    """Test push operation: append an item to a list field."""
    repo = initialized_repository
    e = Entity(name="PushTest", tags=["initial"])
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).push("tags", "new_tag")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["initial", "new_tag"]


async def test_update_pop_operation(initialized_repository, logger):
    """Test pop operation: remove last (-1: first) element."""
    repo = initialized_repository
    e = Entity(name="PopTest", tags=["a", "b", "c"])
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Pop last element (position=1)
    update_last = Update(Entity).pop("tags", 1)
    await repo.update_one(options, update_last, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["a", "b"]

    # Pop first element (position=-1)
    update_first = Update(Entity).pop("tags", -1)
    await repo.update_one(options, update_first, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["b"]


async def test_update_unset_operation(initialized_repository, logger):
    """Test unset operation: remove a field or set to default."""
    # Note: Behavior might differ. SQL might set to NULL, Mongo removes field.
    # Test assumes field removal or setting to None if optional.
    repo = initialized_repository
    e = Entity(name="UnsetTest", metadata={"note": "important", "extra": 1}, owner="owner1")
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    # Unset an optional field and a field within metadata
    update = Update(Entity).unset("owner").unset("metadata.note")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert updated.owner is None
    assert updated.metadata is not None # Metadata dict itself shouldn't be removed
    assert "note" not in updated.metadata # Key 'note' should be gone
    assert updated.metadata.get("extra") == 1 # Other keys remain


async def test_update_pull_operation(initialized_repository, logger):
    """Test pull operation: remove a specific item from a list."""
    repo = initialized_repository
    e = Entity(name="PullTest", tags=["x", "y", "z", "y"])
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    # Pull all occurrences of "y"
    update = Update(Entity).pull("tags", "y")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert updated.tags == ["x", "z"]


async def test_update_increment_operation(initialized_repository, logger):
    """Test increment operation."""
    repo = initialized_repository
    e = Entity(name="IncTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).increment("value", 10) # Increment by 10

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 110

    # Test default increment (by 1)
    update_default = Update(Entity).increment("value")
    await repo.update_one(options, update_default, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 111


async def test_update_decrement_operation(initialized_repository, logger):
    """Test decrement operation."""
    repo = initialized_repository
    e = Entity(name="DecTest", value=50)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).decrement("value", 5) # Decrement by 5

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 45

    # Test default decrement (by 1)
    update_default = Update(Entity).decrement("value")
    await repo.update_one(options, update_default, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 44


async def test_update_min_operation(initialized_repository, logger):
    """Test min operation."""
    repo = initialized_repository
    e = Entity(name="MinTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Update with value lower than current
    update_lower = Update(Entity).min("value", 50)
    await repo.update_one(options, update_lower, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 50 # Should update to 50

    # Update with value higher than current
    update_higher = Update(Entity).min("value", 75)
    await repo.update_one(options, update_higher, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 50 # Should remain 50


async def test_update_max_operation(initialized_repository, logger):
    """Test max operation."""
    repo = initialized_repository
    e = Entity(name="MaxTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Update with value higher than current
    update_higher = Update(Entity).max("value", 150)
    await repo.update_one(options, update_higher, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 150 # Should update to 150

    # Update with value lower than current
    update_lower = Update(Entity).max("value", 120)
    await repo.update_one(options, update_lower, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 150 # Should remain 150


async def test_update_multiply_operation(initialized_repository, logger):
    """Test multiply operation."""
    repo = initialized_repository
    e = Entity(name="MulTest", value=10)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).mul("value", 2.5) # Multiply by 2.5

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 25.0 # 10 * 2.5


# =============================================================================
# Nested Update Tests (Using Update builder)
# =============================================================================

async def test_update_nested_push_operation(initialized_repository, logger):
    """Test push operation on a nested list field."""
    repo = initialized_repository
    e = Entity(name="NestedPushTest", profile={"emails": ["initial@example.com"], "settings": {}})
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    # Use dot notation with Update builder
    update = Update(Entity).push("profile.emails", "new@example.com")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert "new@example.com" in updated.profile["emails"]
    assert updated.profile["emails"] == ["initial@example.com", "new@example.com"]


async def test_update_nested_pop_operation(initialized_repository, logger):
    """Test pop operation on a nested list field."""
    repo = initialized_repository
    initial_emails = ["a@example.com", "b@example.com", "c@example.com"]
    e = Entity(name="NestedPopTest", profile={"emails": initial_emails.copy()})
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Pop last email (position=1)
    update_pop_last = Update(Entity).pop("profile.emails", 1)
    await repo.update_one(options, update_pop_last, logger)
    updated = await repo.get(e.id, logger)
    assert updated.profile["emails"] == ["a@example.com", "b@example.com"]

    # Pop first email (position=-1)
    update_pop_first = Update(Entity).pop("profile.emails", -1)
    await repo.update_one(options, update_pop_first, logger)
    updated = await repo.get(e.id, logger)
    assert updated.profile["emails"] == ["b@example.com"]


async def test_update_nested_unset_operation(initialized_repository, logger):
    """Test unset operation on a nested field."""
    repo = initialized_repository
    e = Entity(
        name="NestedUnsetTest",
        profile={"emails": ["x@example.com"], "phone": "1234"},
        metadata={"keep": "this"}
    )
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).unset("profile.phone") # Unset phone inside profile

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert "phone" not in updated.profile
    assert "emails" in updated.profile # Other nested fields remain
    assert updated.metadata == {"keep": "this"} # Other top-level fields remain


async def test_update_nested_pull_operation(initialized_repository, logger):
    """Test pull operation on a nested list field."""
    repo = initialized_repository
    e = Entity(
        name="NestedPullTest",
        profile={"emails": ["x@example.com", "y@example.com", "z@example.com", "y@example.com"]}
    )
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).pull("profile.emails", "y@example.com")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert updated.profile["emails"] == ["x@example.com", "z@example.com"]


# =============================================================================
# Test updating with Pydantic Model Field (Copied from test_update.py)
# =============================================================================
async def test_update_with_pydantic_model_field_agnostic(initialized_repository, logger):
    """Test updating metadata with a Pydantic model (agnostic)."""
    class TestModel(BaseModel):
        value: str = Field(alias="alias_value")
        url: AnyHttpUrl = Field(alias="alias_url")
        class Config: populate_by_name = True

    repo = initialized_repository
    e = Entity(name="GenericTest", metadata={})
    await repo.store(e, logger)

    test_model_instance = TestModel(value="sample", url="https://example.com/sample.json") # type: ignore

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).set("metadata", test_model_instance)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert isinstance(updated.metadata, dict)
    # Check against alias because prepare_for_storage uses model_dump(by_alias=True)
    assert "alias_value" in updated.metadata
    assert "alias_url" in updated.metadata
    assert updated.metadata["alias_value"] == "sample"
    assert updated.metadata["alias_url"].rstrip("/") == "https://example.com/sample.json".rstrip("/")