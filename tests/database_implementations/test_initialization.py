# tests/database_implementations/test_initialization.py

import pytest
from logging import LoggerAdapter

# Import the base repository interface and entity
from async_repository.base.interfaces import Repository
from tests.conftest import Entity

# Use the parametrized factory fixture to run tests against all backends
pytestmark = pytest.mark.parametrize(
    "repository_factory", ["mongodb", "sqlite"], indirect=True # Add other backends as they are implemented
)


async def test_repository_instantiation(repository_factory, logger: LoggerAdapter):
    """Test that repository instances can be created without errors."""
    try:
        # Use the factory to create an instance
        repo = repository_factory(Entity)
        assert isinstance(repo, Repository)
        logger.info(f"Successfully instantiated {type(repo).__name__}")
    except Exception as e:
        pytest.fail(f"Repository instantiation failed: {e}")


async def test_explicit_create_schema(repository_factory, logger: LoggerAdapter):
    """Test that create_schema runs without error and is idempotent."""
    repo = repository_factory(Entity)
    try:
        # First call should succeed (or do nothing if schema exists)
        await repo.create_schema(logger)
        logger.info("First create_schema call succeeded.")
        # Second call should also succeed (idempotency check)
        await repo.create_schema(logger)
        logger.info("Second create_schema call succeeded (idempotency check).")
    except Exception as e:
        pytest.fail(f"create_schema failed: {e}")


async def test_explicit_create_indexes(repository_factory, logger: LoggerAdapter):
    """Test that create_indexes runs without error and is idempotent."""
    repo = repository_factory(Entity)
    try:
        # Indexes often require the schema (table/collection) to exist first
        await repo.create_schema(logger)
        # First call
        await repo.create_indexes(logger)
        logger.info("First create_indexes call succeeded.")
        # Second call (idempotency)
        await repo.create_indexes(logger)
        logger.info("Second create_indexes call succeeded (idempotency check).")
    except Exception as e:
        pytest.fail(f"create_indexes failed: {e}")


async def test_check_schema_after_create(repository_factory, logger: LoggerAdapter):
    """Verify check_schema returns True after create_schema is called."""
    repo = repository_factory(Entity)
    await repo.create_schema(logger)  # Ensure schema exists
    try:
        exists = await repo.check_schema(logger)
        assert exists is True, "check_schema should return True after create_schema"
    except Exception as e:
        pytest.fail(f"check_schema after create failed: {e}")


async def test_check_indexes_after_create(repository_factory, logger: LoggerAdapter):
    """Verify check_indexes returns True after create_indexes is called."""
    repo = repository_factory(Entity)
    await repo.create_schema(logger)   # Prerequisite for indexes
    await repo.create_indexes(logger)  # Ensure indexes exist
    try:
        exists = await repo.check_indexes(logger)
        assert exists is True, "check_indexes should return True after create_indexes"
    except Exception as e:
        pytest.fail(f"check_indexes after create failed: {e}")


async def test_initialize_orchestration_create(repository_factory, logger: LoggerAdapter):
    """Test the initialize method with creation flags set to True."""
    repo = repository_factory(Entity)
    try:
        # Call initialize to create schema and indexes
        await repo.initialize(
            logger,
            create_schema_if_needed=True,
            create_indexes_if_needed=True
        )
        logger.info("initialize call with create=True succeeded.")

        # Verify checks pass afterwards
        schema_ok = await repo.check_schema(logger)
        indexes_ok = await repo.check_indexes(logger)
        assert schema_ok is True, "Schema should exist after initialize(create_schema=True)"
        assert indexes_ok is True, "Indexes should exist after initialize(create_indexes=True)"

    except Exception as e:
        pytest.fail(f"initialize with create=True failed: {e}")


async def test_initialize_orchestration_no_create(repository_factory, logger: LoggerAdapter):
    """
    Test the initialize method with creation flags set to False.
    It should run without error, but schema/indexes might not exist afterwards
    (depending on pre-test state, which is usually clean for these fixtures).
    """
    repo = repository_factory(Entity)
    try:
        # Call initialize without requesting creation
        await repo.initialize(
            logger,
            create_schema_if_needed=False,
            create_indexes_if_needed=False
        )
        logger.info("initialize call with create=False succeeded (did nothing).")

        # Optional: Check schema/indexes again. They likely return False here
        # unless the DB was somehow pre-populated. Just checking for errors
        # during the initialize call itself is the main point.
        # schema_ok = await repo.check_schema(logger)
        # indexes_ok = await repo.check_indexes(logger)
        # logger.info(f"State after initialize(create=False): Schema={schema_ok}, Indexes={indexes_ok}")

    except Exception as e:
        pytest.fail(f"initialize with create=False failed: {e}")