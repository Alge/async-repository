import asyncio
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any, Dict, List

from mongomock_motor import AsyncMongoMockClient

import pytest
import pytest_asyncio

import os
import platform
import glob

# from repositories.user_repositories.ram_userdatabase import RamUserRepository
from tests import create_postgres_tables

# Only import winreg on Windows
if platform.system() == "Windows":
    import winreg  # For Windows registry access

import asyncpg
from pytest_postgresql import factories

REPOSITORY_IMPLEMENTATIONS = ["mongodb", "postgresql", "memory"]


def pytest_addoption(parser):
    parser.addoption(
        "--run-perf-report",
        action="store_true",
        default=False,
        help="Run the full performance report (slow)",
    )


# Check if PostgreSQL binary is available
def is_postgres_available():
    """
    Check if PostgreSQL binaries are available on the system.
    Works on both Linux (Ubuntu) and macOS.


    Returns:
        bool: True if PostgreSQL is detected, False otherwise
    """
    logging.info("Checking for PostgreSQL installation...")

    # Detect operating system
    system = platform.system()
    logging.info(f"Operating system: {system}")

    # First try the standard PATH check - works on all platforms
    for binary in ["psql", "pg_ctl", "postgres"]:
        binary_path = shutil.which(binary)
        if binary_path:
            logging.info(f"Found PostgreSQL binary '{binary}' at: {binary_path}")
            return True

    logging.debug("Could not find PostgreSQL binaries in PATH")

    # System-specific checks
    if system == "Linux":
        return _check_postgres_linux()
    elif system == "Darwin":  # macOS
        return _check_postgres_macos()
    elif system == "Windows":
        return _check_postgres_windows()
    else:
        logging.warning(f"Unsupported operating system: {system}")
        return False


def _check_postgres_linux():
    """Linux-specific PostgreSQL detection"""
    # Check standard Ubuntu PostgreSQL locations
    pg_version_dirs = [
        f"/usr/lib/postgresql/{ver}/bin" for ver in range(10, 17)
    ]  # Check versions 10-16

    for pg_dir in pg_version_dirs:
        for binary in ["pg_ctl", "postgres"]:
            binary_path = os.path.join(pg_dir, binary)
            if os.path.exists(binary_path):
                logging.info(f"Found PostgreSQL binary at: {binary_path}")
                return True

    # Check if the PostgreSQL service is running via systemctl
    try:
        service_check = subprocess.run(
            ["systemctl", "is-active", "postgresql"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        if service_check.returncode == 0:
            logging.info("PostgreSQL service is active")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL service status with systemctl")

    # Check using package manager
    try:
        pkg_check = subprocess.run(
            ["dpkg", "-l", "postgresql*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        if "postgresql" in pkg_check.stdout.decode():
            logging.info("PostgreSQL packages are installed")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL packages with dpkg")

    return False


def _check_postgres_macos():
    """macOS-specific PostgreSQL detection"""
    # Check Homebrew installation locations
    brew_locations = [
        "/usr/local/opt/postgresql/bin",
        "/opt/homebrew/opt/postgresql/bin",
        "/opt/homebrew/opt/postgresql@14/bin",  # Version-specific Homebrew installs
        "/opt/homebrew/opt/postgresql@15/bin",
        "/opt/homebrew/opt/postgresql@16/bin",
    ]

    for brew_path in brew_locations:
        for binary in ["pg_ctl", "postgres"]:
            binary_path = os.path.join(brew_path, binary)
            if os.path.exists(binary_path):
                logging.info(f"Found PostgreSQL binary at: {binary_path}")
                return True

    # Check macOS PostgreSQL.app location
    postgres_app_paths = [
        "/Applications/Postgres.app/Contents/Versions",
        "~/Applications/Postgres.app/Contents/Versions",
    ]

    for base_path in postgres_app_paths:
        expanded_path = os.path.expanduser(base_path)
        if os.path.exists(expanded_path):
            # Check all versions in Postgres.app
            try:
                for version_dir in os.listdir(expanded_path):
                    bin_path = os.path.join(expanded_path, version_dir, "bin")
                    if os.path.exists(bin_path):
                        logging.info(
                            f"Found PostgreSQL.app installation at: {bin_path}"
                        )
                        return True
            except (FileNotFoundError, PermissionError):
                continue

    # Check if postgres is running via launchctl
    try:
        launchctl_check = subprocess.run(
            ["launchctl", "list", "org.postgresql.postgres"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        if launchctl_check.returncode == 0:
            logging.info("PostgreSQL service is active via launchctl")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL service status with launchctl")

    # Check using brew services
    try:
        brew_check = subprocess.run(
            ["brew", "services", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        if "postgresql" in brew_check.stdout.decode():
            logging.info("PostgreSQL service is managed by brew")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL service with brew")

    return False


def _check_postgres_windows():
    """Windows-specific PostgreSQL detection"""
    # Check Program Files locations for EnterpriseDB/PostgreSQL installations
    program_files_paths = [
        os.environ.get("ProgramFiles", "C:\\Program Files"),
        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
    ]

    # Check standard installation paths
    for base_path in program_files_paths:
        # Check EnterpriseDB standard installation
        postgres_paths = glob.glob(os.path.join(base_path, "PostgreSQL", "*", "bin"))
        for bin_path in postgres_paths:
            for binary in ["pg_ctl.exe", "postgres.exe"]:
                binary_path = os.path.join(bin_path, binary)
                if os.path.exists(binary_path):
                    logging.info(f"Found PostgreSQL binary at: {binary_path}")
                    return True

    # Check Windows registry for PostgreSQL installations
    try:
        reg_paths = [
            r"SOFTWARE\PostgreSQL\Installations",
            # 32-bit registry on 32-bit Windows or 64-bit registry on 64-bit Windows
            r"SOFTWARE\Wow6432Node\PostgreSQL\Installations",  # 32-bit registry on 64-bit Windows
        ]

        for reg_path in reg_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path) as key:
                    # Enumerate subkeys (PostgreSQL versions)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            try:
                                base_directory, _ = winreg.QueryValueEx(
                                    subkey, "Base Directory"
                                )
                                bin_path = os.path.join(base_directory, "bin")
                                logging.info(
                                    f"Found PostgreSQL installation in registry at: {bin_path}"
                                )
                                if os.path.exists(bin_path):
                                    return True
                            except WindowsError:
                                continue
            except WindowsError:
                continue
    except Exception as e:
        logging.debug(f"Error checking Windows registry: {e}")

    # Check services
    try:
        # Use SC to query PostgreSQL service status
        sc_query = subprocess.run(
            ["sc", "query", "postgresql"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )

        # Alternative service names
        if sc_query.returncode != 0:
            sc_query = subprocess.run(
                ["sc", "query", "postgresql-x64-14"],  # Example with specific version
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
            )

        if "RUNNING" in sc_query.stdout.decode():
            logging.info("PostgreSQL service is running")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL service status with sc")

    # Check for WSL PostgreSQL installation
    try:
        # If WSL is available, check there too
        wsl_check = subprocess.run(
            ["wsl", "which", "psql"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
        )
        if wsl_check.returncode == 0:
            logging.info("PostgreSQL is available in WSL")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.debug("Could not check PostgreSQL in WSL")

    return False


# PostgreSQL repository fixture using pytest-postgresql
postgresql_proc = factories.postgresql_proc(port=None)


@pytest_asyncio.fixture
async def mock_postgres_pool(postgresql_proc):
    """
    Creates a PostgreSQL connection pool using asyncpg.create_pool with the provided postgresql_proc fixture.
    """
    # Connection details
    conn_str = f"postgresql://postgres:postgres@{postgresql_proc.host}:{postgresql_proc.port}/postgres"

    # Create connection pool
    pool = await asyncpg.create_pool(conn_str)

    yield pool
    await pool.close()


@pytest_asyncio.fixture
async def mock_mongodb_conn():
    """
    Provides a mocked MongoDB connection using AsyncMongoMockClient.
    """
    client = AsyncMongoMockClient(tz_aware=False, mock_io_loop=asyncio.get_event_loop())
    yield client


@pytest.fixture
def memory_repository_factory():
    """Factory for creating in-memory repositories."""


    def create_repo(entity_cls, app_id_field="id", db_id_field="_id"):
        from async_repository.memory.base import MemoryRepository

        return MemoryRepository(
            entity_cls=entity_cls, app_id_field=app_id_field, db_id_field=db_id_field
        )

    return create_repo


@pytest.fixture
def mongodb_repository_factory(mock_mongodb_conn):
    """Factory for creating MongoDB repositories."""

    def create_repo(entity_cls, app_id_field="id", db_id_field="_id"):
        from async_repository.mongodb.base import MongoDBRepository

        collection_names = {
            Entity: "entities",
        }

        collection_name = (
            collection_names.get(entity_cls) or f"{entity_cls.__name__.lower()}s"
        )
        return MongoDBRepository(
            client=mock_mongodb_conn,
            database_name="test_db",
            collection_name=collection_name,
            entity_cls=entity_cls,
            app_id_field=app_id_field,
            db_id_field=db_id_field,
        )

    return create_repo


@pytest_asyncio.fixture
async def postgresql_repository_factory(mock_postgres_pool):
    """Factory for creating PostgreSQL repositories."""

    # Create a clean slate and set up tables
    await create_postgres_tables.create_tables(mock_postgres_pool)

    def create_repo(entity_cls, app_id_field="id", db_id_field="_id"):
        from async_repository.postgresql.base import PostgreSQLRepository

        # The table is assumed to exist or be created elsewhere

        table_names = {
            "Entity": "entities",
        }

        table_name = (
            table_names.get(entity_cls.__name__) or f"{entity_cls.__name__.lower()}s"
        )

        print(
            f"Creating a postgres repo for class: {entity_cls} (entity class: {Entity}). Table name: {table_name}"
        )

        return PostgreSQLRepository(
            pool=mock_postgres_pool,
            table_name=table_name,
            entity_cls=entity_cls,
            db_id_field=db_id_field,
            app_id_field=app_id_field,
        )

    return create_repo


@pytest.fixture
def repository_factory(request):
    try:
        repository_type = request.param
    except:
        repository_type = "memory"

    if repository_type == "memory":  # Changed from "memory_repository"
        return request.getfixturevalue("memory_repository_factory")
    elif repository_type == "mongodb":  # Changed from "mongodb_repository"
        return request.getfixturevalue("mongodb_repository_factory")
    elif repository_type == "postgresql":  # Changed from "postgresql_repository"
        # Await the async fixture to get the create_repo function
        return request.getfixturevalue("postgresql_repository_factory")
    else:
        raise ValueError(f"Unknown repository type: {repository_type}")


@pytest.fixture
def logger():
    """Create a test logger."""
    import logging
    from logging import LoggerAdapter

    logger = logging.getLogger("test_logger")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return LoggerAdapter(logger, {})


@dataclass
class Entity:
    """A simple entity class for repository testing."""

    id: Optional[str] = field(default_factory=lambda: f"test-{uuid.uuid4()}")
    name: str = field(default_factory=lambda: f"Test Entity {uuid.uuid4().hex[:8]}")
    value: int = 100
    tags: List[str] = field(default_factory=lambda: ["test", "sample"])
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    owner: Optional[str] = None
    profile: Dict[str, Any] = field(default_factory=lambda: {"emails": []})

    def model_dump(self, mode="json", by_alias=True) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary mimicking Pydantic's model_dump.
        The parameters are accepted for compatibility; alias transformation
        is not implemented in this simple dataclass.
        """
        return asdict(self)


@pytest.fixture
def test_entity():
    """
    Creates a test entity with unique values.
    """
    return Entity()
