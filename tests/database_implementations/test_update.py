# tests/database_implementations/test_update.py

from datetime import datetime, timezone

import pytest
from pydantic import AnyHttpUrl, BaseModel, Field

from async_repository.base.exceptions import ObjectNotFoundException
from async_repository.base.query import QueryBuilder, QueryOptions
from async_repository.base.update import Update
from tests.conftest import Entity


pytestmark = pytest.mark.usefixtures("initialized_repository")


# Basic CRUD operations with Update
async def test_update_set_fields(initialized_repository, test_entity, logger):
    """Test updating fields using set operation."""
    repo = initialized_repository
    await repo.store(test_entity, logger)

    new_name = "Updated Name"
    new_value = 200
    now = datetime.now(timezone.utc)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == test_entity.id).build()
    update = Update(Entity).set("name", new_name).set("value", new_value).set("updated_at", now)

    await repo.update_one(options, update, logger)
    updated = await repo.get(test_entity.id, logger)

    assert updated.name == new_name
    assert updated.value == new_value
    assert updated.updated_at is not None
    assert abs((updated.updated_at - now).total_seconds()) < 1.0


async def test_update_non_existent(initialized_repository, logger):
    """Test updating a non-existent entity raises exception."""
    repo = initialized_repository
    non_existent_id = "non-existent-id"

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == non_existent_id).build()
    update = Update(Entity).set("name", "Should Fail")

    with pytest.raises(ObjectNotFoundException):
        await repo.update_one(options, update, logger)


async def test_update_return_value(initialized_repository, test_entity, logger):
    """Test update_one return value behavior."""
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

    # Verify update still happened
    updated = await repo.get(test_entity.id, logger)
    assert updated.value == 400


async def test_update_one_with_complex_expression(initialized_repository, logger):
    """Test update_one with complex query expressions."""
    repo = initialized_repository
    e1 = Entity(name="Alice", value=100, active=True)
    e2 = Entity(name="Bob", value=200, active=True)
    e3 = Entity(name="Alice", value=300, active=False)
    for e in [e1, e2, e3]:
        await repo.store(e, logger)

    qb = QueryBuilder(Entity)
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


# Update Many Tests
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

    qb_target = QueryBuilder(Entity)
    options_target = qb_target.filter(qb_target.fields.name == "Target").build()
    update = Update(Entity).set("active", True)

    update_count = await repo.update_many(options_target, update, logger)
    assert update_count == 2

    # Verify changes
    listed = [item async for item in repo.list(logger, options_target)]
    assert len(listed) == 2
    assert all(item.active is True for item in listed)

    # Verify other entity wasn't updated
    qb_other = QueryBuilder(Entity)
    options_other = qb_other.filter(qb_other.fields.name == "Other").build()
    other_entity = await repo.find_one(logger, options_other)

    assert other_entity is not None
    assert other_entity.name == "Other"
    assert other_entity.active is False


async def test_update_many_with_complex_expression(initialized_repository, logger):
    """Test update_many with complex query expressions."""
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
    assert all_entities["Alice"].value == 100
    assert all_entities["Charlie"].value == 150

    assert all_entities["Bob"].updated_at is not None
    assert all_entities["Dave"].updated_at is not None
    assert all_entities["Eve"].updated_at is not None
    assert all_entities["Alice"].updated_at is None
    assert all_entities["Charlie"].updated_at is None


async def test_update_many_with_no_matches(initialized_repository, logger):
    """Test update_many when no entities match the query."""
    repo = initialized_repository
    entities = [Entity(name="Alice"), Entity(name="Bob")]
    for e in entities:
        await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.name == "NonExistent").build()
    update = Update(Entity).set("active", False)

    update_count = await repo.update_many(options, update, logger)
    assert update_count == 0

    # Verify entities were not changed
    async for entity in repo.list(logger):
        assert entity.active is True


async def test_update_many_requires_expression(initialized_repository, logger):
    """Test update_many raises when no query expression is provided."""
    repo = initialized_repository
    await repo.store(Entity(), logger)

    options = QueryOptions()
    update = Update(Entity).set("active", False)

    with pytest.raises(ValueError, match="must include an 'expression'"):
        await repo.update_many(options, update, logger)


# Upsert Tests
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
    await repo.store(entity, logger)

    entity.name = "Upsert Updated"
    entity.value = 250
    await repo.upsert(entity, logger)

    retrieved = await repo.get(entity.id, logger)
    assert retrieved.name == "Upsert Updated"
    assert retrieved.value == 250


# Array Operation Tests
async def test_update_push_operation(initialized_repository, logger):
    """Test push operation on an array field."""
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
    """Test pop operation on an array field."""
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


async def test_update_pull_operation(initialized_repository, logger):
    """Test pull operation on an array field."""
    repo = initialized_repository
    e = Entity(name="PullTest", tags=["x", "y", "z", "y"])
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).pull("tags", "y")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["x", "z"]


async def test_update_unset_operation(initialized_repository, logger):
    """Test unset operation on fields."""
    repo = initialized_repository
    e = Entity(name="UnsetTest", metadata={"note": "important", "extra": 1}, owner="owner1")
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).unset("owner").unset("metadata.note")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert updated.owner is None
    assert updated.metadata is not None
    assert "note" not in updated.metadata
    assert updated.metadata.get("extra") == 1


# Numeric Operation Tests
async def test_update_increment_operation(initialized_repository, logger):
    """Test increment operation on numeric fields."""
    repo = initialized_repository
    e = Entity(name="IncTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).increment("value", 10)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 110

    # Test default increment (by 1)
    update_default = Update(Entity).increment("value")
    await repo.update_one(options, update_default, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 111


async def test_update_decrement_operation(initialized_repository, logger):
    """Test decrement operation on numeric fields."""
    repo = initialized_repository
    e = Entity(name="DecTest", value=50)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).decrement("value", 5)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 45

    # Test default decrement (by 1)
    update_default = Update(Entity).decrement("value")
    await repo.update_one(options, update_default, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 44


async def test_update_min_operation(initialized_repository, logger):
    """Test min operation on numeric fields."""
    repo = initialized_repository
    e = Entity(name="MinTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Update with value lower than current
    update_lower = Update(Entity).min("value", 50)
    await repo.update_one(options, update_lower, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 50

    # Update with value higher than current
    update_higher = Update(Entity).min("value", 75)
    await repo.update_one(options, update_higher, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 50


async def test_update_max_operation(initialized_repository, logger):
    """Test max operation on numeric fields."""
    repo = initialized_repository
    e = Entity(name="MaxTest", value=100)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    # Update with value higher than current
    update_higher = Update(Entity).max("value", 150)
    await repo.update_one(options, update_higher, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 150

    # Update with value lower than current
    update_lower = Update(Entity).max("value", 120)
    await repo.update_one(options, update_lower, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 150


async def test_update_multiply_operation_int(initialized_repository, logger):
    """Test multiply operation on integer fields."""
    repo = initialized_repository
    e = Entity(name="MulTest", value=10)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).mul("value", 4)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.value == 40


async def test_update_multiply_operation_float(initialized_repository, logger):
    """Test multiply operation on float fields."""
    repo = initialized_repository
    e = Entity(name="MulTest", float_value=10.0, value=999)
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).mul("float_value", 2.5)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.float_value == 25.0


# Nested Path Tests
async def test_update_nested_push_operation(initialized_repository, logger):
    """Test push operation on a nested array field."""
    repo = initialized_repository
    e = Entity(name="NestedPushTest", profile={"emails": ["initial@example.com"], "settings": {}})
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).push("profile.emails", "new@example.com")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert "new@example.com" in updated.profile["emails"]
    assert updated.profile["emails"] == ["initial@example.com", "new@example.com"]


async def test_update_nested_pop_operation(initialized_repository, logger):
    """Test pop operation on a nested array field."""
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
    update = Update(Entity).unset("profile.phone")

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert "phone" not in updated.profile
    assert "emails" in updated.profile
    assert updated.metadata == {"keep": "this"}


async def test_update_nested_pull_operation(initialized_repository, logger):
    """Test pull operation on a nested array field."""
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


async def test_update_nested_increment(initialized_repository, logger):
    """Test increment operation on a nested numeric field."""

    repo = initialized_repository
    e = Entity(name="NestedIncrementTest")
    e.metadata = {"stats": {"visits": 10, "actions": 5}}
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).increment("metadata.stats.visits", 3)

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["stats"]["visits"] == 13
    assert updated.metadata["stats"]["actions"] == 5


async def test_update_nested_decrement(initialized_repository, logger):
    """Test decrement operation on a nested numeric field."""

    repo = initialized_repository
    e = Entity(name="NestedDecrementTest")
    e.metadata = {"stats": {"clicks": 20, "impressions": 100}}
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).decrement("metadata.stats.clicks", 5)

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["stats"]["clicks"] == 15
    assert updated.metadata["stats"]["impressions"] == 100


async def test_update_nested_multiply(initialized_repository, logger):
    """Test multiply operation on a nested numeric field."""

    repo = initialized_repository
    e = Entity(name="NestedMultiplyTest")
    e.metadata = {"rates": {"base": 10, "premium": 5}}
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).mul("metadata.rates.base", 2.5)

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["rates"]["base"] == 25
    assert updated.metadata["rates"]["premium"] == 5


async def test_update_nested_min_max(initialized_repository, logger):
    """Test min/max operations on nested numeric fields."""

    repo = initialized_repository
    e = Entity(name="NestedMinMaxTest")
    e.metadata = {"limits": {"floor": 50, "ceiling": 80}}
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    update_min = Update(Entity).min("metadata.limits.floor", 30)
    await repo.update_one(options, update_min, logger)

    update_max = Update(Entity).max("metadata.limits.ceiling", 100)
    await repo.update_one(options, update_max, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["limits"]["floor"] == 30
    assert updated.metadata["limits"]["ceiling"] == 100


async def test_update_nested_set(initialized_repository, logger):
    """Test set operation on a deeply nested field."""
    repo = initialized_repository
    e = Entity(name="NestedSetTest")
    e.metadata = {
        "user": {
            "preferences": {
                "theme": "dark",
                "notifications": True
            }
        }
    }
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).set("metadata.user.preferences.theme", "light")

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["user"]["preferences"]["theme"] == "light"
    assert updated.metadata["user"]["preferences"]["notifications"] is True


async def test_update_multiple_nested_operations(initialized_repository, logger):
    """Test multiple nested operations in a single update."""

    repo = initialized_repository
    e = Entity(name="MultiNestedTest")
    e.metadata = {
        "stats": {"views": 100, "likes": 20},
        "tags": ["initial", "test"],
        "config": {"enabled": True}
    }
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()

    update = (Update(Entity)
              .increment("metadata.stats.views", 50)
              .push("metadata.tags", "updated")
              .set("metadata.config.enabled", False))

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert updated.metadata["stats"]["views"] == 150
    assert updated.metadata["stats"]["likes"] == 20
    assert "updated" in updated.metadata["tags"]
    assert not updated.metadata["config"]["enabled"]


async def test_update_pull_with_dict_criteria(initialized_repository, logger):
    """Test pull operation with dictionary criteria."""

    repo = initialized_repository
    e = Entity(name="PullDictCriteriaTest")
    e.metadata = {
        "items": [
            {"id": 1, "type": "A", "active": True},
            {"id": 2, "type": "B", "active": True},
            {"id": 3, "type": "A", "active": False}
        ]
    }
    await repo.store(e, logger)

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).pull("metadata.items", {"type": "A", "active": True})

    await repo.update_one(options, update, logger)

    updated = await repo.get(e.id, logger)
    assert len(updated.metadata["items"]) == 2
    remaining_types = [item["type"] for item in updated.metadata["items"]]
    assert "B" in remaining_types
    assert any(item["type"] == "A" and not item["active"]
               for item in updated.metadata["items"])


# Advanced Tests
async def test_update_with_pydantic_model_field(initialized_repository, logger):
    """Test updating with a Pydantic model field."""
    class TestModel(BaseModel):
        value: str = Field(alias="alias_value")
        url: AnyHttpUrl = Field(alias="alias_url")
        class Config: populate_by_name = True

    repo = initialized_repository
    e = Entity(name="GenericTest", metadata={})
    await repo.store(e, logger)

    test_model_instance = TestModel(
        value="sample", 
        url="https://example.com/sample.json"
    )

    qb = QueryBuilder(Entity)
    options = qb.filter(qb.fields.id == e.id).build()
    update = Update(Entity).set("metadata", test_model_instance)

    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    assert isinstance(updated.metadata, dict)
    assert "alias_value" in updated.metadata
    assert "alias_url" in updated.metadata
    assert updated.metadata["alias_value"] == "sample"
    assert updated.metadata["alias_url"].rstrip("/") == "https://example.com/sample.json".rstrip("/")