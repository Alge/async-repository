from datetime import datetime
from typing import Optional

import pytest
from pydantic import AnyHttpUrl, BaseModel, Field

from async_repository.base.exceptions import ObjectNotFoundException
from async_repository.base.query import QueryOptions
from async_repository.base.update import Update
from tests.conftest import Entity
from tests.conftest import REPOSITORY_IMPLEMENTATIONS


# =============================================================================
# Tests for update_one method (using Update builder)
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_entity(repository_factory, test_entity, logger):
    """
    Test updating an entity's fields via update_one using Update.
    """
    repo = repository_factory(type(test_entity))
    await repo.store(test_entity, logger)
    new_name = "Updated Name"
    new_value = 200
    # Create filter options matching the entity by ID.
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": test_entity.id}},
        limit=1,
        offset=0,
    )
    update = Update().set("name", new_name).set("value", new_value)
    await repo.update_one(options, update, logger)
    updated = await repo.get(test_entity.id, logger)
    assert updated.name == new_name
    assert updated.value == new_value


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_non_existent(repository_factory, test_entity, logger):
    """
    Test updating a non-existent entity raises ObjectNotFoundException.
    """
    repo = repository_factory(type(test_entity))
    non_existent_id = "non-existent-id"
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": non_existent_id}},
        limit=1,
        offset=0,
    )
    update = Update().set("name", "Should Fail")
    with pytest.raises(ObjectNotFoundException):
        await repo.update_one(options, update, logger)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_return_value(repository_factory, test_entity, logger):
    """
    Test that update_one returns the updated entity when return_value is True,
    and returns None when return_value is False.
    """
    repo = repository_factory(type(test_entity))
    await repo.store(test_entity, logger)
    new_fields = Update().set("name", "Updated Name").set("value", 200)
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": test_entity.id}},
        limit=1,
        offset=0,
    )
    result = await repo.update_one(options, new_fields, logger, return_value=True)
    assert result is not None
    assert result.name == "Updated Name"
    assert result.value == 200

    result_none = await repo.update_one(
        options, Update().set("name", "Another Update"), logger, return_value=False
    )
    assert result_none is None


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_one_with_complex_expression(repository_factory, logger):
    """
    Test update_one with a complex expression that matches exactly one entity.
    """
    repo = repository_factory(Entity)
    e1 = Entity(name="Alice", value=100, active=True)
    e2 = Entity(name="Bob", value=200, active=True)
    e3 = Entity(name="Alice", value=300, active=False)

    for e in [e1, e2, e3]:
        await repo.store(e, logger)

    expr = {
        "and": [
            {"active": {"operator": "eq", "value": True}},
            {"name": {"operator": "eq", "value": "Alice"}},
        ]
    }
    options = QueryOptions(expression=expr, limit=1, offset=0)
    update = Update().set("value", 150)
    await repo.update_one(options, update, logger)

    updated_alice = await repo.get(e1.id, logger)
    assert updated_alice.value == 150
    unchanged_bob = await repo.get(e2.id, logger)
    assert unchanged_bob.value == 200
    inactive_alice = await repo.get(e3.id, logger)
    assert inactive_alice.value == 300


# =============================================================================
# Tests for update_many method (using Update builder)
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_simple(repository_factory, logger):
    """
    Test updating multiple entities with update_many.
    """
    repo = repository_factory(Entity)
    entities = [
        Entity(name="Target", value=100, active=False),
        Entity(name="Target", value=200, active=False),
        Entity(name="Other", value=300, active=False),
    ]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions(expression={"name": {"operator": "eq", "value": "Target"}})
    update = Update().set("active", True)
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 2

    listed = []
    filter_options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Target"}}
    )
    async for item in repo.list(logger, filter_options):
        listed.append(item)
    assert len(listed) == 2
    assert all(item.active is True for item in listed)

    other_options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Other"}}
    )
    other_entity = await repo.find_one(logger, other_options)
    assert other_entity.active is False


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_with_limit(repository_factory, logger):
    """
    Test update_many with a limit to update only a subset of matching entities.
    """
    repo = repository_factory(Entity)
    entities = [
        Entity(name="UpdateMe", value=100, active=False),
        Entity(name="UpdateMe", value=200, active=False),
        Entity(name="UpdateMe", value=300, active=False),
    ]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "UpdateMe"}},
        limit=2,
        sort_by="value",
        sort_desc=False,
    )
    update = Update().set("active", True)
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 2

    async for entity in repo.list(logger):
        if entity.value in [100, 200]:
            assert entity.active is True
        elif entity.value == 300:
            assert entity.active is False


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_with_offset(repository_factory, logger):
    """
    Test update_many with offset to skip the first N matching entities.
    """
    repo = repository_factory(Entity)
    entities = [
        Entity(name="OffsetTest", value=10, active=False),
        Entity(name="OffsetTest", value=20, active=False),
        Entity(name="OffsetTest", value=30, active=False),
        Entity(name="OffsetTest", value=40, active=False),
    ]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "OffsetTest"}},
        offset=2,
        sort_by="value",
        sort_desc=False,
    )
    update = Update().set("active", True)
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 2

    async for entity in repo.list(logger):
        if entity.value in [30, 40]:
            assert entity.active is True
        elif entity.value in [10, 20]:
            assert entity.active is False


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_with_complex_expression(repository_factory, logger):
    """
    Test update_many with a complex expression involving AND, OR conditions.
    """
    repo = repository_factory(Entity)
    entities = [
        Entity(name="Alice", value=100, active=True, owner="A", updated_at=None),
        Entity(name="Bob", value=200, active=True, owner="B", updated_at=None),
        Entity(name="Charlie", value=150, active=False, owner="A", updated_at=None),
        Entity(name="Dave", value=250, active=True, owner="B", updated_at=None),
        Entity(name="Eve", value=300, active=False, owner="A", updated_at=None),
    ]
    for e in entities:
        await repo.store(e, logger)

    expr = {
        "or": [
            {
                "and": [
                    {"active": {"operator": "eq", "value": True}},
                    {"owner": {"operator": "eq", "value": "B"}},
                ]
            },
            {"value": {"operator": "gt", "value": 200}},
        ]
    }
    options = QueryOptions(expression=expr)
    update_timestamp = datetime.utcnow()
    update = Update().set("updated_at", update_timestamp).set("value", 1000)
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 3

    async for entity in repo.list(logger):
        if entity.name in ["Bob", "Dave", "Eve"]:
            assert entity.updated_at is not None
            time_diff = abs((entity.updated_at - update_timestamp).total_seconds())
            assert time_diff < 1.0
            assert entity.value == 1000
        else:
            assert entity.updated_at is None
            assert entity.value != 1000


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_with_sort(repository_factory, logger):
    """
    Test update_many with sorting and limit to ensure the correct subset is updated.
    """
    repo = repository_factory(Entity)
    entities = [
        Entity(name="Player", value=85, active=False),
        Entity(name="Player", value=92, active=False),
        Entity(name="Player", value=78, active=False),
        Entity(name="Player", value=100, active=False),
        Entity(name="Player", value=65, active=False),
    ]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "Player"}},
        limit=2,
        sort_by="value",
        sort_desc=True,
    )
    update = Update().set("active", True)
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 2

    all_players = []
    async for player in repo.list(
        logger, QueryOptions(expression={"name": {"operator": "eq", "value": "Player"}})
    ):
        all_players.append(player)

    active_players = [p for p in all_players if p.active]
    assert len(active_players) == 2
    top_values = [p.value for p in active_players]
    assert sorted(top_values, reverse=True) == [100, 92]


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_with_no_matches(repository_factory, logger):
    """
    Test update_many when no entities match the expression.
    """
    repo = repository_factory(Entity)
    entities = [Entity(name="Alice"), Entity(name="Bob")]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions(
        expression={"name": {"operator": "eq", "value": "NonExistent"}}
    )
    update = Update().set("active", False)  # Use an existing field
    update_count = await repo.update_many(options, update, logger)
    assert update_count == 0

    async for entity in repo.list(logger):
        assert entity.active is True  # Default is True


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_all_with_no_expression(repository_factory, logger):
    """
    Test that update_many raises ValueError when no expression is provided.
    """
    repo = repository_factory(Entity)
    entities = [Entity(), Entity()]
    for e in entities:
        await repo.store(e, logger)

    options = QueryOptions()  # No expression
    with pytest.raises(ValueError, match="must include an 'expression'"):
        await repo.update_many(options, Update().set("active", False), logger)


# =============================================================================
# Tests for Upsert Method
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_upsert_inserts_new_entity(repository_factory, logger):
    """
    Test that upsert inserts a new entity when it does not exist.
    """
    repo = repository_factory(Entity)
    entity = Entity(name="Upsert New", value=123)
    await repo.upsert(entity, logger)
    retrieved = await repo.get(entity.id, logger)
    assert retrieved.name == entity.name
    assert retrieved.value == entity.value


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_upsert_updates_existing_entity(repository_factory, logger):
    """
    Test that upsert updates an existing entity.
    """
    repo = repository_factory(Entity)
    entity = Entity(name="Upsert Update", value=100)
    await repo.store(entity, logger)
    entity.name = "Upsert Updated"
    entity.value = 250
    await repo.upsert(entity, logger)
    retrieved = await repo.get(entity.id, logger)
    assert retrieved.name == "Upsert Updated"
    assert retrieved.value == 250


# =============================================================================
# New Tests for push, pop, unset, and pull operations via Update builder
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_push_operation(repository_factory, logger):
    """
    Test push operation: append an item to a list field.
    """
    repo = repository_factory(Entity)
    # Entity has a field 'tags' which is a list.
    e = Entity(name="PushTest", tags=["initial"])
    await repo.store(e, logger)
    update = Update().push("tags", "new_tag")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert "new_tag" in updated.tags


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_pop_operation(repository_factory, logger):
    """
    Test pop operation: remove the last element (or first, if specified) from a list field.
    """
    repo = repository_factory(Entity)
    e = Entity(name="PopTest", tags=["a", "b", "c"])
    await repo.store(e, logger)
    # Pop last element.
    update = Update().pop("tags", 1)
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["a", "b"]

    # Pop first element.
    update = Update().pop("tags", -1)
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert updated.tags == ["b"]


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_unset_operation(repository_factory, logger):
    """
    Test unset operation: remove a field by setting it to its default value.
    For fields with not-null constraints, this means setting to an empty value rather than null.
    """
    repo = repository_factory(Entity)
    # Create an entity with a non-empty metadata field
    e = Entity(name="UnsetTest", metadata={"note": "to be removed"})
    await repo.store(e, logger)
    # Use metadata.note instead of the whole metadata field to avoid not-null constraint
    update = Update().unset("metadata.note")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    # The note field should be removed from metadata
    assert "note" not in updated.metadata
    # Metadata itself should still exist as at least an empty dict
    assert updated.metadata is not None


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_pull_operation(repository_factory, logger):
    """
    Test pull operation: remove a specific item from a list field.
    """
    repo = repository_factory(Entity)
    e = Entity(name="PullTest", tags=["x", "y", "z", "y"])
    await repo.store(e, logger)
    update = Update().pull("tags", "y")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    # Both occurrences of "y" should be removed.
    assert "y" not in updated.tags
    # Other tags remain.
    assert set(updated.tags) == {"x", "z"}


# =============================================================================
# Tests for Nested Update Operations via Update Builder
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_nested_push_operation(repository_factory, logger):
    """
    Test push operation on a nested field using dot notation.
    For example, push a new email into profile.emails.
    """
    repo = repository_factory(Entity)
    # Create an entity with a nested profile field that has emails.
    e = Entity(name="NestedPushTest", profile={"emails": ["initial@example.com"]})
    await repo.store(e, logger)
    # Use the Update builder to push a new email.
    update = Update().push("profile.emails", "new@example.com")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    assert "new@example.com" in updated.profile["emails"]


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_nested_pop_operation(repository_factory, logger):
    """
    Test pop operation on a nested field using dot notation.
    First, push an extra email then pop from the nested list.
    """
    repo = repository_factory(Entity)
    initial_emails = ["a@example.com", "b@example.com", "c@example.com"]
    e = Entity(name="NestedPopTest", profile={"emails": initial_emails.copy()})
    await repo.store(e, logger)
    # Push an extra email.
    update_push = Update().push("profile.emails", "d@example.com")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update_push, logger)
    # Now, pop the last email (direction=1)
    update_pop = Update().pop("profile.emails", 1)
    await repo.update_one(options, update_pop, logger)
    updated = await repo.get(e.id, logger)
    # The list should revert to the initial emails.
    assert updated.profile["emails"] == initial_emails


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_nested_unset_operation(repository_factory, logger):
    """
    Test unset operation on a nested field.
    For example, remove the 'emails' key from profile.
    """
    repo = repository_factory(Entity)
    e = Entity(
        name="NestedUnsetTest",
        profile={"emails": ["x@example.com", "y@example.com"], "phone": "1234"},
    )
    await repo.store(e, logger)
    update = Update().unset("profile.emails")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    # Depending on implementation, 'emails' may be removed or set to None.
    profile = updated.profile
    assert "emails" not in profile or profile.get("emails") is None


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_nested_pull_operation(repository_factory, logger):
    """
    Test pull operation on a nested field.
    For example, remove a specific email from profile.emails.
    """
    repo = repository_factory(Entity)
    e = Entity(
        name="NestedPullTest",
        profile={
            "emails": [
                "x@example.com",
                "y@example.com",
                "z@example.com",
                "y@example.com",
            ]
        },
    )
    await repo.store(e, logger)
    update = Update().pull("profile.emails", "y@example.com")
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)
    # Both occurrences of "y@example.com" should be removed.
    assert "y@example.com" not in updated.profile["emails"]
    # Other emails remain.
    assert set(updated.profile["emails"]) == {"x@example.com", "z@example.com"}


# =============================================================================
# New Test: Updating with a Generic Pydantic Model Field (Agnostic Implementation)
# =============================================================================
@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_with_pydantic_model_field_agnostic(repository_factory, logger):
    """
    Test updating an entity's field (using an already existing field, 'metadata') with a generic
    Pydantic model value. This verifies that the Update builder correctly serializes the model
    using model_dump(mode="json", by_alias=True) so that BSON serialization errors are avoided.
    """

    # Use Pydantic v2 style field aliases with Field(alias=...)
    class TestModel(BaseModel):
        value: str = Field(alias="alias_value")
        url: AnyHttpUrl = Field(alias="alias_url")

        class Config:
            populate_by_name = (
                True  # Pydantic v2 equivalent of allow_population_by_field_name
            )

    repo = repository_factory(Entity)
    # Create an entity with an empty metadata dict.
    e = Entity(name="GenericTest", metadata={})
    await repo.store(e, logger)

    # Create an instance of the generic Pydantic model.
    test_model_instance = TestModel(
        value="sample", url="https://example.com/sample.json"
    )

    # Build an update that sets the 'metadata' field with the Pydantic model instance.
    update = Update().set("metadata", test_model_instance)
    options = QueryOptions(
        expression={"id": {"operator": "eq", "value": e.id}}, limit=1, offset=0
    )

    # This update should now succeed without BSON serialization errors.
    await repo.update_one(options, update, logger)
    updated = await repo.get(e.id, logger)

    # Verify that the updated 'metadata' field is a dict containing the alias keys.
    assert isinstance(updated.metadata, dict)
    assert "alias_value" in updated.metadata
    assert "alias_url" in updated.metadata
    assert updated.metadata["alias_value"] == "sample"
    assert updated.metadata["alias_url"].rstrip(
        "/"
    ) == "https://example.com/sample.json".rstrip("/")
