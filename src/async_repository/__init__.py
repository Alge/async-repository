# src/async_repository/__init__.py

"""
Async Repository Library Initialization.

This package provides an asynchronous repository pattern implementation with
support for various database backends.

It initializes a logger with a NullHandler and makes core components like
Repository interfaces, Query/Update builders, exceptions, and backend
implementations available at the top level.
"""

import logging

# --------------------------------------------------------------------------
# Logging Setup
# --------------------------------------------------------------------------
# Initialize logger for the library. By default, it uses a NullHandler,
# which means library logs are discarded unless the consuming application
# configures logging. This prevents the library from interfering with the
# application's logging setup.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False  # Prevent log messages from propagating to the root logger

# --------------------------------------------------------------------------
# Core Interface and Exception Exports
# --------------------------------------------------------------------------
from .base.interfaces import Repository
from .base.exceptions import ObjectNotFoundException, KeyAlreadyExistsException

# --------------------------------------------------------------------------
# Query Building Exports
# --------------------------------------------------------------------------
# QueryBuilder is the primary way to construct queries.
# QueryOptions represents the built query configuration.
# QueryOperator provides the available filter operators.
from .base.query import QueryBuilder, QueryOptions, QueryOperator

# --------------------------------------------------------------------------
# Update Building Exports
# --------------------------------------------------------------------------
# Update is the primary way to construct update operations.
from .base.update import Update

# --------------------------------------------------------------------------
# Repository Implementation Exports
# --------------------------------------------------------------------------
# Make the specific database repository implementations directly available.
# Users can import them like: from async_repository import MongoDBRepository
from .db_implementations.mongodb_repository import MongoDBRepository
from .db_implementations.postgresql_repository import PostgresRepository
from .db_implementations.sqlite_repository import SqliteRepository
from .db_implementations.mysql_repository import MySQLRepository

# --------------------------------------------------------------------------
# __all__ Definition
# --------------------------------------------------------------------------
# Define the public API of the package. This controls what is imported
# when a user does `from async_repository import *`. It's also useful
# for documentation tools and linters.
__all__ = [
    # Core
    "Repository",
    # Exceptions
    "ObjectNotFoundException",
    "KeyAlreadyExistsException",
    # Query
    "QueryBuilder",
    "QueryOptions",
    "QueryOperator",
    # Update
    "Update",
    # Implementations
    "MongoDBRepository",
    "PostgresRepository",
    "SqliteRepository",
    "MySQLRepository",
    # Logging
    "logger",
]

# Optional: Define package version (often done here or in a separate _version.py)
# __version__ = "0.1.0"