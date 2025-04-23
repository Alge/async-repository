# tests/database_implementations/test_serializing.py

import pytest
from pydantic import AnyHttpUrl, AnyWebsocketUrl, BaseModel, Field
from typing import Any

# Import QueryBuilder and QueryOptions correctly
from async_repository.base.query import QueryBuilder, QueryOptions
from async_repository.base.update import Update
from tests.conftest import (Entity, REPOSITORY_IMPLEMENTATIONS,
                            initialized_repository, logger)  # Use fixtures

# Apply initialized_repository to all tests in this module
pytestmark = pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)


@pytest.mark.asyncio
async def test_store_and_get_with_url_types(initialized_repository, logger):
    """Test storing and retrieving an entity with URL types."""
    repo = initialized_repository

    entity = Entity(name="URL Store Test")
    entity.metadata = {
        "stream_url": AnyHttpUrl("https://media.example.com/stream"),
        "config": {"websocket": AnyWebsocketUrl("ws://example.com:8765/")},
    }
    entity.profile = {
        "emails": ["test@example.com"],
        "services": {
            "notification_url": AnyHttpUrl("https://notify.example.com/user1")
        },
    }

    await repo.store(entity, logger)
    retrieved = await repo.get(entity.id, logger)

    # prepare_for_storage converts URLs to strings
    assert retrieved.id == entity.id
    assert (
        retrieved.metadata["stream_url"] == "https://media.example.com/stream"
    )
    assert (
        retrieved.metadata["config"]["websocket"] == "ws://example.com:8765/"
    )
    assert (
        retrieved.profile["services"]["notification_url"]
        == "https://notify.example.com/user1"
    )


@pytest.mark.asyncio
async def test_update_with_url_types(initialized_repository, logger):
    """Test updating an entity with URL types in metadata."""
    repo = initialized_repository
    entity = Entity(name="Update URL Test")
    await repo.store(entity, logger)

    update_data = {
        "api_url": AnyHttpUrl("https://api.example.com/v1"),
        "connection": {
            "socket_url": AnyWebsocketUrl("ws://socket.example.com:9000/")
        },
    }
    update = Update(Entity).set("metadata", update_data)

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    query_options = qb.filter(qb.fields.id == entity.id).build()

    await repo.update_one(query_options, update, logger)
    updated = await repo.get(entity.id, logger)

    # Verify stored values are strings
    assert updated.name == "Update URL Test"
    assert updated.metadata["api_url"] == "https://api.example.com/v1"
    assert (
        updated.metadata["connection"]["socket_url"]
        == "ws://socket.example.com:9000/"
    )


@pytest.mark.asyncio
async def test_update_nested_url_field(initialized_repository, logger):
    """Test updating a nested URL field directly."""
    repo = initialized_repository
    entity = Entity(name="Nested URL Test")
    entity.metadata = {"api": {"url": "https://old-api.example.com"}}
    await repo.store(entity, logger)

    update = Update(Entity).set(
        "metadata.api.url", AnyHttpUrl("https://new-api.example.com")
    )

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    query_options = qb.filter(qb.fields.id == entity.id).build()

    await repo.update_one(query_options, update, logger)
    updated = await repo.get(entity.id, logger)

    # Verify the stored value is a string
    assert updated.metadata["api"]["url"] == "https://new-api.example.com/"


@pytest.mark.asyncio
async def test_find_entities_with_url_query(initialized_repository, logger):
    """Test finding entities based on URL string values."""
    repo = initialized_repository
    entity1 = Entity(name="URL Query Test 1")
    entity1.metadata = {"service_url": "https://service-a.example.com"}
    entity2 = Entity(name="URL Query Test 2")
    entity2.metadata = {"service_url": "https://service-b.example.com"}
    await repo.store(entity1, logger)
    await repo.store(entity2, logger)

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    # Access nested field using dot notation
    # Assuming 'metadata' allows attribute access or is defined as a nested model/dataclass
    # If 'metadata' is just Dict[str, Any], __getitem__ might be needed:
    # query_options = qb.filter(qb.fields.metadata["service_url"] == "https://service-a.example.com").build() # type: ignore
    query_options = qb.filter(
        qb.fields.metadata.service_url == "https://service-a.example.com"
    ).build()

    found_entities = [item async for item in repo.list(logger, query_options)]

    assert len(found_entities) == 1
    assert found_entities[0].id == entity1.id
    assert found_entities[0].name == "URL Query Test 1"
    assert (
        found_entities[0].metadata["service_url"]
        == "https://service-a.example.com"
    )


@pytest.mark.asyncio
async def test_upsert_with_url_types(initialized_repository, logger):
    """Test upserting an entity with URL types."""
    repo = initialized_repository
    entity = Entity(name="Upsert URL Test")
    entity.metadata = {
        "endpoint": AnyHttpUrl("https://original.example.com/api"),
        "websocket": AnyWebsocketUrl("ws://original.example.com:8080"),
    }
    await repo.upsert(entity, logger)  # First upsert (insert)

    # Modify and upsert again (update)
    entity.metadata["endpoint"] = AnyHttpUrl("https://updated.example.com/api")
    entity.metadata["status"] = "updated"
    await repo.upsert(entity, logger)

    retrieved = await repo.get(entity.id, logger)

    # Verify stored values are strings and update occurred
    assert retrieved.name == "Upsert URL Test"
    assert retrieved.metadata["endpoint"] == "https://updated.example.com/api"
    assert (
        retrieved.metadata["websocket"] == "ws://original.example.com:8080/"
    )  # Unchanged part
    assert retrieved.metadata["status"] == "updated"  # Added part


@pytest.mark.asyncio
async def test_update_many_with_url_types(initialized_repository, logger):
    """Test updating multiple entities with URL types."""
    repo = initialized_repository
    ids_to_update = set()
    for i in range(3):
        entity = Entity(name=f"Multi URL Test {i}")
        entity.metadata = {"api": {"url": f"https://old-{i}.example.com/api"}}
        await repo.store(entity, logger)
        ids_to_update.add(entity.id)

    # Update builder handles URL serialization
    update = Update(Entity).set(
        "metadata.api.url", AnyHttpUrl("https://common-updated.example.com/api")
    )

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    query_options = qb.filter(qb.fields.name.startswith("Multi URL Test")).build()

    updated_count = await repo.update_many(query_options, update, logger)
    assert updated_count == 3

    # Verify updates
    list_options = query_options  # Reuse the same filter
    updated_entities = [item async for item in repo.list(logger, list_options)]

    assert len(updated_entities) == 3
    for entity in updated_entities:
        assert (
            entity.metadata["api"]["url"]
            == "https://common-updated.example.com/api"
        )
        assert entity.id in ids_to_update


@pytest.mark.asyncio
async def test_error_scenario_reproduction(initialized_repository, logger):
    """Reproduce and test the specific error scenario from logs."""

    class TestModel(BaseModel): # Define model used in original scenario
        value: str = Field(alias="alias_value")
        url: AnyHttpUrl = Field(alias="alias_url")
        class Config: populate_by_name = True

    repo = initialized_repository
    entity = Entity(name="Error Scenario Test")
    await repo.store(entity, logger)

    update_data = {
        "next_action": {
            "type": "play_url",
            "url": AnyHttpUrl("https://http-live.sr.se/p1-mp3-192"),
            "mode": "stream",
        },
        "type": "external_media_websocket",
        "host": AnyWebsocketUrl("ws://alge.se:8765/"),
        "codec": "pcm",
        # Add the Pydantic model instance that might have caused issues if not serialized
        "pyd_nested": TestModel(value="sample", url="https://example.com/pyd.json") # type: ignore
    }
    # Use dot notation for nested field update
    update = Update(Entity).set("metadata.on_outgoing_call", update_data)

    # Use QueryBuilder
    qb = QueryBuilder(Entity)
    query_options = qb.filter(
        (qb.fields.id == entity.id) & (qb.fields.name == "Error Scenario Test")
    ).build()

    # This update should succeed
    await repo.update_one(query_options, update, logger)
    updated = await repo.get(entity.id, logger)

    # Verify update (values should be stored as strings or simple types)
    meta_call = updated.metadata["on_outgoing_call"]
    assert meta_call["type"] == "external_media_websocket"
    assert meta_call["host"] == "ws://alge.se:8765/"
    assert meta_call["codec"] == "pcm"
    assert meta_call["next_action"]["type"] == "play_url"
    assert (
        meta_call["next_action"]["url"] == "https://http-live.sr.se/p1-mp3-192"
    )
    assert meta_call["next_action"]["mode"] == "stream"
    # Verify nested Pydantic model was serialized correctly (with aliases)
    assert isinstance(meta_call["pyd_nested"], dict)
    assert meta_call["pyd_nested"]["alias_value"] == "sample"
    assert meta_call["pyd_nested"]["alias_url"] == "https://example.com/pyd.json"