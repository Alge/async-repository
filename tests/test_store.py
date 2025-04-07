import pytest
from async_repository.base.exceptions import KeyAlreadyExistsException
from async_repository.base.query import QueryOptions
from async_repository.postgresql.base import PostgreSQLRepository
from tests.conftest import Entity

from tests.conftest import REPOSITORY_IMPLEMENTATIONS

from tests.create_postgres_tables import create_entity_table


@pytest.mark.skip(
    "Does not work, the Entity class generates it's own IDs. Need another test class for this."
)
@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_with_none_app_id_when_fields_same(repository_factory, logger):
    """
    Test that the repository correctly handles storing entities with None app_id
    when app_id_field and db_id_field are the same.

    This test simulates the behavior observed with SIP subscribers where both
    fields reference the same property ('id') and the app_id is initially None.
    """
    # Create a repository where app_id_field and db_id_field are the same
    repo = repository_factory(Entity, app_id_field="id", db_id_field="id")

    if isinstance(repo, PostgreSQLRepository):
        await create_entity_table(pool=repo._pool, app_id_field="id", db_id_field="id")

    # Create an entity with id set to None
    entity1 = Entity(name="Test Entity 1")
    entity1.id = None

    # First store should succeed
    await repo.store(entity1, logger)

    # Create another entity with id also set to None
    entity2 = Entity(id=None, name="Test Entity 2")

    # Second store should also succeed without raising KeyAlreadyExistsException
    await repo.store(entity2, logger)  # This would fail if the bug is present

    # Verify both entities were stored
    count = await repo.count(logger)
    assert count >= 2, "Repository should contain at least the two test entities"

    # Get the entities from the repository
    entities = []
    async for entity in repo.list(logger):
        if entity.name in ["Test Entity 1", "Test Entity 2"]:
            entities.append(entity)

    # Verify the entities have proper IDs
    assert len(entities) == 2
    logger.info(entities)
    assert False
    assert all(
        entity.id is not None for entity in entities
    ), "All entities should have generated IDs"
    assert entities[0].id != entities[1].id, "Generated IDs should be unique"
