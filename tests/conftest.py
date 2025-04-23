# tests/conftest.py
import asyncio
import glob
import logging
import os
import platform
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional



# Import necessary drivers
import asyncpg
import pytest
import pytest_asyncio
import aiosqlite
import motor.motor_asyncio
from pymongo.errors import ConnectionFailure

from async_repository.db_implementations.postgresql_repository import PostgresRepository

# Only import winreg on Windows
if platform.system() == "Windows":
    try:
        import winreg
    except ImportError:
        winreg = None  # type: ignore

# --- Import Repository Types ---
from async_repository.base.interfaces import Repository
from async_repository.db_implementations.mongodb_repository import MongoDBRepository
from async_repository.db_implementations.sqlite_repository import SqliteRepository

# from async_repository.postgresql.postgres_repository import PostgreSQLRepository

# --- Constants ---
TEST_MONGO_DB_NAME = "pytest_async_repo_db"
MONGO_URI = os.getenv("TEST_MONGO_URI", "mongodb://localhost:27017")

# --- List of available implementation keys ---
REPOSITORY_IMPLEMENTATIONS = ["mongodb", "sqlite", "postgresql"]


# --- Availability Checks ---
def is_postgres_available():
    return shutil.which("psql") is not None


def is_mongodb_available():
    """Check if MongoDB is available (basic check)."""
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URI, serverSelectionTimeoutMS=1000  # Quick timeout
        )
        client.admin.command("ismaster")
        logging.info(f"MongoDB found and responsive at {MONGO_URI}")
        client.close()
        return True
    except ConnectionFailure:
        logging.warning(
            f"MongoDB not found or not responsive at {MONGO_URI}. "
            "Skipping MongoDB tests."
        )
        return False
    except Exception as e:
        logging.warning(
            f"Error checking MongoDB connection at {MONGO_URI}: {e}. "
            "Skipping MongoDB tests."
        )
        return False


AVAILABLE_IMPLEMENTATIONS = []
if is_mongodb_available():
    AVAILABLE_IMPLEMENTATIONS.append("mongodb")
if True:  # Assume SQLite (in-memory) is always available
    AVAILABLE_IMPLEMENTATIONS.append("sqlite")
if is_postgres_available():
    AVAILABLE_IMPLEMENTATIONS.append("postgresql")


# --- Fixtures ---

# Event Loop (Function Scoped - Default for pytest-asyncio tests)


@pytest_asyncio.fixture
async def mock_postgres_pool(postgresql_proc):
    """
    Creates a PostgreSQL connection pool with a unique temporary database for each test.
    """
    # Generate a unique database name for this test
    temp_db_name = f"test_db_{uuid.uuid4().hex}"

    # Connection details for admin connection to create/drop database
    admin_conn_str = f"postgresql://postgres:postgres@{postgresql_proc.host}:{postgresql_proc.port}/postgres"

    # Create admin connection to postgres db
    admin_conn = await asyncpg.connect(admin_conn_str)

    try:
        # Create a temporary database for this test
        await admin_conn.execute(f'CREATE DATABASE "{temp_db_name}"')

        # Connection string for the test database
        test_conn_str = f"postgresql://postgres:postgres@{postgresql_proc.host}:{postgresql_proc.port}/{temp_db_name}"

        # Create connection pool to the test database
        pool = await asyncpg.create_pool(test_conn_str)

        yield pool

        # Close all connections in the pool
        await pool.close()

        # Drop the temporary database after test
        await admin_conn.execute(f'DROP DATABASE "{temp_db_name}"')

    finally:
        # Close admin connection
        await admin_conn.close()


# MongoDB Client Fixture (Function Scoped)
@pytest_asyncio.fixture(scope="function")
async def motor_client():
    """Provides a real Motor client connected for each test."""
    if "mongodb" not in AVAILABLE_IMPLEMENTATIONS:
        pytest.skip("MongoDB not available or connection failed.")

    client = None
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        await client.admin.command("ismaster")  # Quick check
        logging.debug("Motor client created for test function scope.")
        yield client
    except ConnectionFailure as e:
        pytest.skip(
            f"Skipping MongoDB test: Could not connect to MongoDB at {MONGO_URI}: {e}"
        )
    finally:
        if client:
            client.close()
            logging.debug("Motor client closed for test function scope.")


# MongoDB Database Cleanup Fixture (Function Scoped)
@pytest_asyncio.fixture(scope="function", autouse=True)
async def clean_mongo_db(motor_client):
    """Cleans the test MongoDB database before each test function runs."""
    if not motor_client:
        return  # Skip if client fixture was skipped

    db = motor_client[TEST_MONGO_DB_NAME]
    collections = await db.list_collection_names()
    logging.debug(
        f"Cleaning MongoDB database '{TEST_MONGO_DB_NAME}'. "
        f"Dropping collections: {collections}"
    )
    for name in collections:
        if not name.startswith("system."):
            try:
                await db.drop_collection(name)
                logging.debug(f"Dropped collection: {name}")
            except Exception as e:
                logging.error(f"Error dropping collection {name}: {e}")
    yield


# SQLite Fixture (Function Scoped)
@pytest_asyncio.fixture(scope="function")
async def sqlite_memory_db_conn():
    """Provides an in-memory aiosqlite database connection for testing."""
    conn = None
    try:
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        yield conn
    finally:
        if conn:
            await conn.close()


# --- Repository Factories (Function Scoped) ---


@pytest.fixture(scope="function")
def mongodb_repository_factory(motor_client, clean_mongo_db):
    """Factory for creating MongoDB repositories using a real client."""
    if not motor_client:
        pytest.skip("MongoDB client not available.")

    def _create(entity_cls, app_id_field="id", db_id_field="_id"):
        collection_name = f"{entity_cls.__name__.lower()}s_pytest"
        return MongoDBRepository(
            client=motor_client,
            database_name=TEST_MONGO_DB_NAME,
            collection_name=collection_name,
            entity_type=entity_cls,
            app_id_field=app_id_field,
            db_id_field=db_id_field,
        )

    return _create


@pytest.fixture(scope="function")
def sqlite_repository_factory(sqlite_memory_db_conn):
    """Factory for creating SQLite repositories using an in-memory DB."""
    if not sqlite_memory_db_conn:
        pytest.skip("SQLite connection not available.")

    def _create(entity_cls, app_id_field="id", db_id_field=None):
        table_name = f"{entity_cls.__name__.lower()}s_pytest"
        effective_db_id_field = (
            db_id_field if db_id_field is not None else app_id_field
        )
        return SqliteRepository(
            db_connection=sqlite_memory_db_conn,
            table_name=table_name,
            entity_type=entity_cls,
            db_id_field=effective_db_id_field,
            app_id_field=app_id_field,
        )

    return _create


@pytest.fixture
def postgresql_repository_factory(mock_postgres_pool):
    """Factory for creating PostgreSQL repositories."""

    def create_repo(entity_type, app_id_field="id", db_id_field="_id"):
        table_names = {
            "Entity": "entities",
        }

        table_name = (
            table_names.get(entity_type.__name__) or f"{entity_type.__name__.lower()}s"
        )

        return PostgresRepository(
            db_pool=mock_postgres_pool,
            table_name=table_name,
            entity_type=entity_type,
            db_id_field=db_id_field,
            app_id_field=app_id_field,
        )

    return create_repo

# --- Parametrized Factory and Initialized Repo ---


@pytest.fixture(params=AVAILABLE_IMPLEMENTATIONS)
def repository_factory(request):
    """Parametrized fixture to get the correct factory based on implementation key."""
    impl_key = request.param
    if impl_key == "mongodb":
        yield request.getfixturevalue("mongodb_repository_factory")
    elif impl_key == "sqlite":
        yield request.getfixturevalue("sqlite_repository_factory")
    elif impl_key == "postgresql":
        yield request.getfixturevalue("postgresql_repository_factory")
    else:
        raise ValueError(f"Unknown repository implementation key: {impl_key}")


@pytest_asyncio.fixture
async def initialized_repository(repository_factory, logger):
    """
    Provides a repository instance that has had its schema and indexes checked
    and potentially created. Uses the parametrized repository_factory.
    """
    repo = repository_factory(Entity)
    try:
        await repo.initialize(
            logger,
            create_schema_if_needed=True,
            create_indexes_if_needed=True,
        )
        yield repo
    except Exception as e:
        logger.error(
            f"Failed to initialize repository type '{type(repo).__name__}' "
            f"during test setup: {e}",
            exc_info=True,
        )
        pytest.fail(
            f"Repository initialization failed for {type(repo).__name__}: {e}"
        )


# --- Logger Fixture ---


@pytest.fixture(scope="session")
def logger():
    """Create a test logger."""
    _logger = logging.getLogger("test_repo_logger")
    if not _logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        _logger.addHandler(handler)
        _logger.setLevel(logging.DEBUG)
        _logger.propagate = False
    return logging.LoggerAdapter(_logger, {})


# --- Test Entity ---


@dataclass
class ProfileData:
    """Nested structure for Entity profile."""

    emails: List[str] = field(default_factory=list)
    phone: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    """A simple entity class for repository testing."""

    id: Optional[str] = field(default_factory=lambda: f"test-{uuid.uuid4()}")
    name: str = field(
        default_factory=lambda: f"Test Entity {uuid.uuid4().hex[:8]}"
    )
    value: int = 100
    float_value: float = 100.0
    tags: List[str] = field(default_factory=lambda: ["test", "sample"])
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = None
    owner: Optional[str] = None
    profile: ProfileData = field(default_factory=ProfileData)

    def copy(self):
        # Using asdict might require handling nested dataclasses if deep copy needed
        data = asdict(self)
        # Manually reconstruct ProfileData if asdict converted it
        if isinstance(data.get('profile'), dict):
             data['profile'] = ProfileData(**data['profile'])
        return Entity(**data)


    def model_dump(self, mode="json", by_alias=True) -> Dict[str, Any]:
        """Mimics Pydantic for compatibility tests."""
        # Simple asdict conversion - NOTE: This might not handle nested models
        # or aliases exactly like Pydantic v2 model_dump. Be aware in tests.
        return asdict(self)


@pytest.fixture
def test_entity() -> Entity:
    """Creates a test entity instance with default values."""
    return Entity()


# --- Optional: Helper to get specific repo type ---


@pytest.fixture
def get_repo_type(initialized_repository):
    """Returns a string identifying the type of the initialized repository."""
    repo = initialized_repository
    if isinstance(repo, SqliteRepository):
        return "sqlite"
    elif isinstance(repo, MongoDBRepository):
        return "mongodb"
    # elif isinstance(repo, PostgreSQLRepository): return "postgresql"
    else:
        return "unknown"