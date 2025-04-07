import pytest
from pydantic import AnyHttpUrl, AnyWebsocketUrl
from dataclasses import asdict
from typing import Dict, Any

from repositories.base.query import QueryOptions
from repositories.base.update import Update
from repositories.base.exceptions import ObjectNotFoundException
from tests.testlib import REPOSITORY_IMPLEMENTATIONS
from tests.conftest import Entity  # Use the existing Entity class


# Helper function for the tests
def prepare_for_mongodb(data: Any) -> Any:
    """Test helper to convert Pydantic URL types to strings"""
    if data is None:
        return None
    elif isinstance(data, dict):
        return {k: prepare_for_mongodb(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [prepare_for_mongodb(item) for item in data]
    elif hasattr(data, '__class__') and data.__class__.__module__ == 'pydantic.networks':
        return str(data)
    return data


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_store_and_get_with_url_types(repository_factory, logger):
    """Test storing and retrieving an entity with URL types in metadata."""
    repo = repository_factory(Entity)

    # Create test entity with URL types in metadata and profile
    entity = Entity()
    entity.metadata = {
        "stream_url": AnyHttpUrl("https://media.example.com/stream"),
        "config": {
            "websocket": AnyWebsocketUrl("ws://example.com:8765/")
        }
    }
    entity.profile = {
        "emails": ["test@example.com"],
        "services": {
            "notification_url": AnyHttpUrl("https://notify.example.com/user1")
        }
    }

    # Store the entity
    await repo.store(entity, logger)

    # Retrieve the entity
    retrieved = await repo.get(entity.id, logger)

    # Verify the entity was stored and retrieved correctly - normalize URLs for comparison
    assert retrieved.id == entity.id
    assert str(retrieved.metadata["stream_url"]).rstrip('/') == "https://media.example.com/stream".rstrip('/')
    assert str(retrieved.metadata["config"]["websocket"]).rstrip('/') == "ws://example.com:8765/".rstrip('/')
    assert str(retrieved.profile["services"]["notification_url"]).rstrip('/') == "https://notify.example.com/user1".rstrip('/')


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_update_with_url_types(repository_factory, logger):
    """Test updating an entity with URL types in metadata."""
    repo = repository_factory(Entity)

    # Create and store initial entity
    entity = Entity(name="Update URL Test")
    await repo.store(entity, logger)

    # Create update with URL fields
    update = Update().set("metadata", {
        "api_url": AnyHttpUrl("https://api.example.com/v1"),
        "connection": {
            "socket_url": AnyWebsocketUrl("ws://socket.example.com:9000/")
        }
    })

    # Update the entity
    query_options = QueryOptions(expression={"id": entity.id})
    await repo.update_one(query_options, update, logger)

    # Retrieve the updated entity
    updated = await repo.get(entity.id, logger)

    # Verify the update worked correctly - normalize URLs for comparison
    assert updated.id == entity.id
    assert updated.name == "Update URL Test"
    assert str(updated.metadata["api_url"]).rstrip('/') == "https://api.example.com/v1".rstrip('/')
    assert str(updated.metadata["connection"]["socket_url"]).rstrip('/') == "ws://socket.example.com:9000/".rstrip('/')


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_update_nested_url_field(repository_factory, logger):
    """Test updating a nested URL field directly."""
    repo = repository_factory(Entity)

    # Create and store initial entity with metadata
    entity = Entity(name="Nested URL Test")
    entity.metadata = {
        "api": {
            "url": "https://old-api.example.com"
        }
    }
    await repo.store(entity, logger)

    # Update just the nested URL field
    update = Update().set(
        "metadata.api.url",
        AnyHttpUrl("https://new-api.example.com")
    )

    # Perform the update
    query_options = QueryOptions(expression={"id": entity.id})
    await repo.update_one(query_options, update, logger)

    # Retrieve the updated entity
    updated = await repo.get(entity.id, logger)

    # Verify the update worked correctly - allow for trailing slash that Pydantic may add
    assert str(updated.metadata["api"]["url"]).rstrip('/') == "https://new-api.example.com".rstrip('/')


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_find_entities_with_url_query(repository_factory, logger):
    """Test finding entities based on URL string values."""
    repo = repository_factory(Entity)

    # Create two entities with different URLs in metadata
    entity1 = Entity(name="URL Query Test 1")
    entity1.metadata = {
        "service_url": "https://service-a.example.com"
    }

    entity2 = Entity(name="URL Query Test 2")
    entity2.metadata = {
        "service_url": "https://service-b.example.com"
    }

    # Store both entities
    await repo.store(entity1, logger)
    await repo.store(entity2, logger)

    # Create query options to find by URL
    query_options = QueryOptions(
        expression={"metadata.service_url": "https://service-a.example.com"}
    )

    # Find entities matching the query
    found_entities = []
    async for entity in repo.list(logger, query_options):
        found_entities.append(entity)

    # Verify we found the right entity
    assert len(found_entities) == 1
    assert found_entities[0].id == entity1.id
    assert found_entities[0].name == "URL Query Test 1"


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_upsert_with_url_types(repository_factory, logger):
    """Test upserting an entity with URL types."""
    repo = repository_factory(Entity)

    # Create entity with URL types in metadata
    entity = Entity(name="Upsert URL Test")
    entity.metadata = {
        "endpoint": AnyHttpUrl("https://original.example.com/api"),
        "websocket": AnyWebsocketUrl("ws://original.example.com:8080")
    }

    # Upsert the entity
    await repo.upsert(entity, logger)

    # Modify the entity with new URL values
    entity.metadata = {
        "endpoint": AnyHttpUrl("https://updated.example.com/api"),
        "websocket": AnyWebsocketUrl("ws://updated.example.com:8080")
    }

    # Upsert again
    await repo.upsert(entity, logger)

    # Retrieve the entity
    retrieved = await repo.get(entity.id, logger)

    # Verify the entity was updated correctly - normalize URLs for comparison
    assert retrieved.id == entity.id
    assert retrieved.name == "Upsert URL Test"
    assert str(retrieved.metadata["endpoint"]).rstrip('/') == "https://updated.example.com/api".rstrip('/')
    assert str(retrieved.metadata["websocket"]).rstrip('/') == "ws://updated.example.com:8080".rstrip('/')


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_update_many_with_url_types(repository_factory, logger):
    """Test updating multiple entities with URL types."""
    repo = repository_factory(Entity)

    # Create and store multiple entities
    for i in range(3):
        entity = Entity(name=f"Multi URL Test {i}")
        entity.metadata = {
            "api": {
                "url": f"https://old-{i}.example.com/api"
            }
        }
        await repo.store(entity, logger)

    # Create an update with URL fields to apply to all entities
    update = Update().set(
        "metadata.api.url",
        AnyHttpUrl("https://common-updated.example.com/api")
    )

    # Update all entities with the matching name pattern
    query_options = QueryOptions(
        expression={"name": {"operator": "startswith", "value": "Multi URL Test"}}
    )
    updated_count = await repo.update_many(query_options, update, logger)

    # Verify update count
    assert updated_count == 3

    # Check each entity was updated correctly by listing all matching entities
    query_options = QueryOptions(
        expression={"name": {"operator": "startswith", "value": "Multi URL Test"}}
    )

    updated_entities = []
    async for entity in repo.list(logger, query_options):
        updated_entities.append(entity)

    # All entities should have the same updated URL - normalize URLs for comparison
    for entity in updated_entities:
        assert str(entity.metadata["api"]["url"]).rstrip('/') == "https://common-updated.example.com/api".rstrip('/')


@pytest.mark.parametrize("repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True)
async def test_error_scenario_reproduction(repository_factory, logger):
    """Reproduce and test the specific error scenario from the logs."""
    repo = repository_factory(Entity)

    # Create a test entity
    entity = Entity(name="Error Scenario Test")
    await repo.store(entity, logger)

    # Create an update with the structure similar to the error case
    update = Update().set("metadata.on_outgoing_call", {
        "next_action": {
            "type": "play_url",
            "url": AnyHttpUrl("https://http-live.sr.se/p1-mp3-192"),
            "mode": "stream"
        },
        "type": "external_media_websocket",
        "host": AnyWebsocketUrl("ws://alge.se:8765/"),
        "codec": "pcm"
    })

    # Update the entity
    query_options = QueryOptions(
        expression={
            "and": [
                {"id": entity.id},
                {"name": "Error Scenario Test"}
            ]
        }
    )

    # This should not raise an exception after our fix
    await repo.update_one(query_options, update, logger)

    # Retrieve the updated entity
    updated = await repo.get(entity.id, logger)

    # Verify the update worked correctly - normalize URLs for comparison
    assert updated.metadata["on_outgoing_call"]["type"] == "external_media_websocket"
    assert str(updated.metadata["on_outgoing_call"]["host"]).rstrip('/') == "ws://alge.se:8765/".rstrip('/')
    assert updated.metadata["on_outgoing_call"]["codec"] == "pcm"
    assert updated.metadata["on_outgoing_call"]["next_action"]["type"] == "play_url"
    assert str(updated.metadata["on_outgoing_call"]["next_action"]["url"]).rstrip('/') == "https://http-live.sr.se/p1-mp3-192".rstrip('/')
    assert updated.metadata["on_outgoing_call"]["next_action"]["mode"] == "stream"
