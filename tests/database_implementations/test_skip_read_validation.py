# tests/database_implementations/test_basic_crud.py
import pytest
from dataclasses import asdict

from pprint import pprint

from pydantic_core._pydantic_core import ValidationError

from async_repository import Repository
from async_repository.base.exceptions import (
    KeyAlreadyExistsException,
    ObjectNotFoundException, ObjectValidationException,
)
# Make sure QueryBuilder is imported
from async_repository.base.query import QueryOptions, QueryBuilder
from tests.conftest import Entity, REPOSITORY_IMPLEMENTATIONS

# Use the initialized_repository fixture for tests needing a ready DB
pytestmark = pytest.mark.usefixtures("initialized_repository")


async def test_instantiate_invalid_model():
    """
    Just a sanity check to make sure we cannot instantiate the test
    model with wrong data
    """

    with pytest.raises(ValueError):
        e = Entity(bool_has_to_be_true=False)
        pprint(e)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_read_invalid_entity(initialized_repository, test_entity, logger):
    """
    Test that reading an invalid entity triggers an exception
    """
    repo: Repository[Entity] = initialized_repository

    test_entity.bool_has_to_be_true = False

    await repo.store(test_entity, logger=logger)

    with pytest.raises(ObjectValidationException):
        read_entity = await repo.get(test_entity.id, logger=logger)

        # We should not reach this
        pprint(read_entity)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_read_invalid_entity_with_skip_validation_true(initialized_repository,
                                                             test_entity, logger):
    """
    Test that we can load data that does not pass the custom validators if
    we set the skip_custom_validation flag in the repository
    """
    repo: Repository[Entity] = initialized_repository

    test_entity.bool_has_to_be_true = False

    await repo.store(test_entity, logger=logger)

    repo._skip_custom_validation = True

    # This should not trigger an exception now!
    read_entity = await repo.get(test_entity.id, logger=logger)


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_read_invalid_entity_with_wrong_type(initialized_repository,
                                                   test_entity, logger):
    """
    Test that the type hinting is still validated.
    For example a bool should allways be a bool
    """
    repo: Repository[Entity] = initialized_repository

    test_entity.bool_has_to_be_true = "this is not a bool"

    await repo.store(test_entity, logger=logger)

    repo._skip_custom_validation = True

    # This should not trigger an exception now!
    read_entity = await repo.get(test_entity.id, logger=logger)
