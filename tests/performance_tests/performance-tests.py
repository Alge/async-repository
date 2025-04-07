import asyncio
import time
import statistics
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Type, Callable, Tuple, Optional
import matplotlib.pyplot as plt
import pytest
import pandas as pd

from repositories.base.exceptions import (
    ObjectNotFoundException,
    KeyAlreadyExistsException,
)
from repositories.base.query import QueryOptions
from repositories.base.update import Update
from tests.conftest import Entity
from tests.testlib import REPOSITORY_IMPLEMENTATIONS


# =============================================================================
# Performance Metrics Collection
# =============================================================================


class PerformanceMetrics:
    """Class to collect and analyze performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.latencies = []
        self.start_time = None
        self.end_time = None
        self.operation_count = 0

    def start(self):
        """Start timing a batch of operations."""
        self.start_time = time.time()

    def end(self, count: int):
        """End timing a batch of operations."""
        self.end_time = time.time()
        self.operation_count += count

    def record_latency(self, latency: float):
        """Record a single operation latency."""
        self.latencies.append(latency)

    def total_duration(self) -> float:
        """Get the total duration of all operations."""
        return (
            self.end_time - self.start_time if self.start_time and self.end_time else 0
        )

    def throughput(self) -> float:
        """Calculate operations per second."""
        duration = self.total_duration()
        if duration < 0.001:  # If duration is less than 1ms
            return self.operation_count * 1000  # Estimate based on average latency
        return self.operation_count / duration if duration > 0 else 0

    def avg_latency(self) -> float:
        """Calculate average latency of recorded operations."""
        return statistics.mean(self.latencies) if self.latencies else 0

    def min_latency(self) -> float:
        """Get minimum latency."""
        return min(self.latencies) if self.latencies else 0

    def max_latency(self) -> float:
        """Get maximum latency."""
        return max(self.latencies) if self.latencies else 0

    def p95_latency(self) -> float:
        """Get 95th percentile latency."""
        return (
            statistics.quantiles(self.latencies, n=20)[-1]
            if len(self.latencies) >= 20
            else self.max_latency()
        )

    def median_latency(self) -> float:
        """Get median latency."""
        return statistics.median(self.latencies) if self.latencies else 0

    def summary(self) -> Dict[str, Any]:
        """Return a summary of metrics as a dictionary."""
        return {
            "name": self.name,
            "operation_count": self.operation_count,
            "total_duration_sec": self.total_duration(),
            "throughput_ops_per_sec": self.throughput(),
            "avg_latency_ms": self.avg_latency() * 1000,
            "min_latency_ms": self.min_latency() * 1000,
            "median_latency_ms": self.median_latency() * 1000,
            "p95_latency_ms": self.p95_latency() * 1000,
            "max_latency_ms": self.max_latency() * 1000,
        }


# =============================================================================
# Test Data Generation and Validation
# =============================================================================


async def ensure_entities_exist(repo, entity_ids, logger, batch_size=100):
    """
    Verify that entities exist in the repository and recreate any missing ones.

    Args:
        repo: The repository to check
        entity_ids: List of entity IDs to verify
        logger: Logger adapter for recording operations
        batch_size: How many entities to check at once

    Returns:
        List of verified entity IDs that exist in the repository
    """
    verified_ids = []
    missing_ids = []

    # Check existence in batches
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i : i + batch_size]

        # Check each entity individually
        for entity_id in batch:
            try:
                # Try to get the entity to verify it exists
                await repo.get(entity_id, logger)
                verified_ids.append(entity_id)
            except ObjectNotFoundException:
                missing_ids.append(entity_id)

    # Recreate missing entities if needed
    if missing_ids:
        logger.info(
            f"Recreating {len(missing_ids)} missing entities for performance testing"
        )
        new_entities = []

        for i, missing_id in enumerate(missing_ids):
            # Create a new entity with the same ID
            entity = Entity(
                id=missing_id,  # Use the original ID
                name=f"Recreated-{i}",
                value=i % 100,
                active=i % 2 == 0,
                tags=[f"tag{i % 5}", f"tag{(i + 1) % 5}"],
                metadata={"recreated": True, "index": i},
                profile={"emails": [f"recreated{i}@example.com"]},
            )
            new_entities.append(entity)
            verified_ids.append(missing_id)

        # Store new entities in parallel
        await asyncio.gather(*[repo.store(entity, logger) for entity in new_entities])

    return verified_ids


async def generate_test_data(
    repo_factory, entity_class: Type, count: int, logger
) -> List[str]:
    """Generate test data and return the entity IDs."""
    repo = repo_factory(entity_class)
    entity_ids = []

    # Create entities in smaller batches to avoid overloading the system
    batch_size = min(100, count)
    for i in range(0, count, batch_size):
        batch_count = min(batch_size, count - i)
        entities = []
        for j in range(batch_count):
            entity = entity_class(
                name=f"PerfTest-{i + j}",
                value=j % 100,
                active=j % 2 == 0,
                tags=[f"tag{j % 5}", f"tag{(j + 1) % 5}"],
                metadata={"batch": i // batch_size, "index": j},
                profile={"emails": [f"test{j}@example.com", f"alt{j}@example.com"]},
            )
            entities.append(entity)
            entity_ids.append(entity.id)

        # Store entities in parallel for better performance
        try:
            await asyncio.gather(*[repo.store(entity, logger) for entity in entities])
        except Exception as e:
            logger.warning(f"Error storing batch {i // batch_size}: {str(e)}")
            # Continue with the next batch

    # Verify entities actually exist
    verified_ids = await ensure_entities_exist(repo, entity_ids, logger)
    return verified_ids


# =============================================================================
# Transaction Management for Test Cleanup
# =============================================================================


class TestTransaction:
    """Helper class to manage test data and cleanup for performance tests."""

    def __init__(self, repo_factory, entity_class, logger):
        self.repo_factory = repo_factory
        self.entity_class = entity_class
        self.logger = logger
        self.entity_ids = []
        self.repo = self.repo_factory(self.entity_class)

    async def setup(self, count: int) -> List[str]:
        """Set up test data and return verified entity IDs."""
        # Use the same repo factory that returns our existing repo instance
        self.entity_ids = await generate_test_data(
            self.repo_factory, self.entity_class, count, self.logger
        )
        return self.entity_ids

    async def cleanup(self) -> None:
        """Clean up test data created during the test."""
        if not self.repo or not self.entity_ids:
            return

        # Delete test entities in batches
        batch_size = 100
        for i in range(0, len(self.entity_ids), batch_size):
            batch = self.entity_ids[i : i + batch_size]
            options = QueryOptions(
                expression={"id": {"operator": "in", "value": batch}}
            )
            try:
                await self.repo.delete_many(options, self.logger)
            except Exception as e:
                self.logger.error(f"Error cleaning up test data: {e}")


# =============================================================================
# Performance Tests - CRUD Operations
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_get_performance(repository_factory, logger):
    """
    Test the performance of get operations.
    Measures latency of retrieving entities by ID.
    """
    # Configuration
    entity_count = 1000
    get_count = 200

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    entity_ids = await transaction.setup(entity_count)

    metrics = PerformanceMetrics("get")

    try:
        # Perform gets and measure latency
        for i in range(min(get_count, len(entity_ids))):
            entity_id = entity_ids[i % len(entity_ids)]

            start_time = time.time()
            await repo.get(entity_id, logger)
            end_time = time.time()

            metrics.record_latency(end_time - start_time)

        # Get repository name
        repo_name = repo.__class__.__name__

        # Print results
        summary = metrics.summary()
        print(f"\nGet Performance Results for {repo_name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_get_performance.txt", "w") as f:
            f.write(f"GET PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_store_performance(repository_factory, logger):
    """
    Test the performance of store operations.
    Measures latency of storing new entities.
    """
    # Configuration
    entity_count = 1000

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    metrics = PerformanceMetrics("store")
    stored_ids = []

    try:
        # Generate entities and measure latency
        for i in range(entity_count):
            entity = Entity(
                name=f"StorePerfTest-{i}",
                value=i % 100,
                active=i % 2 == 0,
                tags=[f"tag{i % 5}"],
                metadata={"test": "performance", "index": i},
            )

            start_time = time.time()
            result = await repo.store(entity, logger, return_value=True)
            end_time = time.time()

            if result:
                stored_ids.append(result.id)
            metrics.record_latency(end_time - start_time)

        # Get repository name
        repo_name = repo.__class__.__name__

        # Print results
        summary = metrics.summary()
        print(f"\nStore Performance Results for {repo_name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_store_performance.txt", "w") as f:
            f.write(f"STORE PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    finally:
        # Cleanup stored entities
        for i in range(0, len(stored_ids), 100):
            batch = stored_ids[i : i + 100]
            options = QueryOptions(
                expression={"id": {"operator": "in", "value": batch}}
            )
            try:
                await repo.delete_many(options, logger)
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_one_performance(repository_factory, logger):
    """
    Test the performance of single update operations.
    Measures latency of individual updates.
    """
    # Configuration
    entity_count = 1000
    update_count = 200

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    entity_ids = await transaction.setup(entity_count)

    metrics = PerformanceMetrics("update_one")

    try:
        # Perform updates and measure latency
        for i in range(min(update_count, len(entity_ids))):
            entity_id = entity_ids[i % len(entity_ids)]
            update = Update().set("value", i).set("updated_at", datetime.utcnow())
            options = QueryOptions(
                expression={"id": {"operator": "eq", "value": entity_id}}
            )

            start_time = time.time()
            await repo.update_one(options, update, logger)
            end_time = time.time()

            metrics.record_latency(end_time - start_time)

        # Get repository name
        repo_name = repo.__class__.__name__

        # Print results
        summary = metrics.summary()
        print(f"\nUpdate One Performance Results for {repo_name}:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_update_one_performance.txt", "w") as f:
            f.write(f"UPDATE ONE PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_update_many_performance(repository_factory, logger):
    """
    Test the performance of bulk update operations.
    Measures throughput of updates to multiple entities.
    """
    # Configuration
    entity_count = 5000
    batch_sizes = [10, 100, 500]

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    await transaction.setup(entity_count)

    try:
        results = []

        # Test different batch sizes
        for batch_size in batch_sizes:
            metrics = PerformanceMetrics(f"update_many_batch_{batch_size}")
            update = Update().set("value", 500).set("updated_at", datetime.utcnow())

            # Run multiple iterations to get stable results
            for i in range(3):
                options = QueryOptions(
                    expression={"name": {"operator": "like", "value": "PerfTest-%"}},
                    limit=batch_size,
                    offset=i * batch_size % (entity_count - batch_size),
                )

                metrics.start()
                count = await repo.update_many(options, update, logger)
                metrics.end(count)

            # Store results
            summary = metrics.summary()
            results.append(summary)
            print(f"\nUpdate Many (Batch Size: {batch_size}) Results:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

        # Get repository name for output files
        repo_name = repo.__class__.__name__

        # Create throughput comparison chart
        batch_sizes = [result["name"].split("_")[-1] for result in results]
        throughputs = [result["throughput_ops_per_sec"] for result in results]

        plt.figure(figsize=(10, 6))
        plt.bar(batch_sizes, throughputs)
        plt.title(f"{repo_name} - Update Throughput by Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Operations per Second")
        plt.savefig(f"{output_dir}/{repo_name}_update_throughput.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_update_many_performance.txt", "w") as f:
            f.write(f"UPDATE MANY PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for batch_size, result in zip(batch_sizes, results):
                f.write(f"Batch Size: {batch_size}\n")
                f.write("-" * 20 + "\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_delete_performance(repository_factory, logger):
    """
    Test the performance of delete operations.
    Measures latency of both delete_one and delete_many.
    """
    # Configuration
    entity_count = 1000
    delete_one_count = 100
    batch_sizes = [10, 50, 100]

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Test delete_one performance
    delete_one_metrics = PerformanceMetrics("delete_one")
    delete_one_ids = []

    # Create entities specifically for delete_one tests
    for i in range(delete_one_count):
        entity = Entity(name=f"DeleteOneTest-{i}", value=i)
        result = await repo.store(entity, logger, return_value=True)
        if result:
            delete_one_ids.append(result.id)

    # Perform delete_one operations
    for entity_id in delete_one_ids:
        start_time = time.time()
        try:
            await repo.delete_one(entity_id, logger)
        except Exception as e:
            logger.warning(f"Error deleting entity {entity_id}: {e}")
        end_time = time.time()

        delete_one_metrics.record_latency(end_time - start_time)

    # Print delete_one results
    one_summary = delete_one_metrics.summary()
    print(f"\nDelete One Performance Results:")
    for key, value in one_summary.items():
        print(f"  {key}: {value}")

    # Test delete_many performance
    delete_many_results = []

    for batch_size in batch_sizes:
        # Create entities for this batch
        batch_ids = []
        entities = []
        for i in range(batch_size):
            entity = Entity(name=f"DeleteManyTest-{batch_size}-{i}", value=i)
            entities.append(entity)

        # Store entities
        stored = await asyncio.gather(
            *[repo.store(entity, logger, return_value=True) for entity in entities]
        )
        batch_ids = [entity.id for entity in stored if entity]

        if not batch_ids:
            continue

        # Measure delete_many performance
        metrics = PerformanceMetrics(f"delete_many_batch_{batch_size}")
        options = QueryOptions(
            expression={"id": {"operator": "in", "value": batch_ids}}
        )

        metrics.start()
        count = await repo.delete_many(options, logger)
        metrics.end(count)

        # Store results
        summary = metrics.summary()
        delete_many_results.append(summary)
        print(f"\nDelete Many (Batch Size: {batch_size}) Results:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    # Get repository name
    repo_name = repo.__class__.__name__

    # Create delete performance comparison chart
    batch_sizes = ["one"] + [
        result["name"].split("_")[-1] for result in delete_many_results
    ]
    throughputs = [one_summary["throughput_ops_per_sec"]] + [
        result["throughput_ops_per_sec"] for result in delete_many_results
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(batch_sizes, throughputs)
    plt.title(f"{repo_name} - Delete Operation Throughput")
    plt.xlabel("Batch Size (one = delete_one)")
    plt.ylabel("Operations per Second")
    plt.savefig(f"{output_dir}/{repo_name}_delete_throughput.png")

    # Generate text output
    with open(f"{output_dir}/{repo_name}_delete_performance.txt", "w") as f:
        f.write(f"DELETE PERFORMANCE RESULTS FOR {repo_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write("DELETE ONE RESULTS:\n")
        f.write("-" * 20 + "\n")
        for key, value in one_summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("DELETE MANY RESULTS:\n")
        for i, result in enumerate(delete_many_results):
            batch_size = result["name"].split("_")[-1]
            f.write(f"Batch Size: {batch_size}\n")
            f.write("-" * 20 + "\n")
            for key, value in result.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_list_performance(repository_factory, logger):
    """
    Test the performance of list operations.
    Measures latency of listing entities with different query options.
    """
    # Configuration
    entity_count = 5000

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    await transaction.setup(entity_count)

    try:
        # Test scenarios for list operation
        test_scenarios = [
            ("List All", QueryOptions(limit=entity_count)),
            ("List with Pagination", QueryOptions(limit=100, offset=0)),
            (
                "List with Filter",
                QueryOptions(
                    expression={"active": {"operator": "eq", "value": True}},
                    limit=entity_count,
                ),
            ),
            (
                "List with Sort",
                QueryOptions(limit=entity_count, sort_by="value", sort_desc=True),
            ),
            (
                "List with Complex Filter",
                QueryOptions(
                    expression={
                        "and": [
                            {"value": {"operator": "gt", "value": 50}},
                            {"active": {"operator": "eq", "value": True}},
                        ]
                    },
                    limit=entity_count,
                ),
            ),
        ]

        results = []

        for scenario_name, options in test_scenarios:
            metrics = PerformanceMetrics(
                f"list_{scenario_name.replace(' ', '_').lower()}"
            )

            # Start timing
            metrics.start()

            # Get all entities
            fetched = []
            fetched_count = 0
            async for entity in repo.list(logger, options):
                fetched.append(entity)
                fetched_count += 1
                if fetched_count >= 1000:  # Limit to prevent excessive memory usage
                    break

            # End timing
            metrics.end(fetched_count)

            # Store results
            summary = metrics.summary()
            summary["entities_fetched"] = fetched_count
            results.append(summary)

            print(f"\nList ({scenario_name}) Results:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

        # Get repository name
        repo_name = repo.__class__.__name__

        # Create throughput comparison chart
        scenario_names = [result["name"].split("_", 1)[1] for result in results]
        throughputs = [result["throughput_ops_per_sec"] for result in results]

        plt.figure(figsize=(12, 6))
        plt.bar(scenario_names, throughputs)
        plt.title(f"{repo_name} - List Operation Throughput by Scenario")
        plt.xlabel("Scenario")
        plt.ylabel("Entities per Second")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{repo_name}_list_throughput.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_list_performance.txt", "w") as f:
            f.write(f"LIST PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for scenario, result in zip(test_scenarios, results):
                scenario_name = scenario[0]
                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 20 + "\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_find_one_performance(repository_factory, logger):
    """
    Test the performance of find_one operations.
    Measures latency of finding a single entity with different query options.
    """
    # Configuration
    entity_count = 1000

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    await transaction.setup(entity_count)

    try:
        # Test scenarios for find_one operation
        test_scenarios = [
            (
                "Simple ID",
                QueryOptions(
                    expression={
                        "id": {"operator": "eq", "value": transaction.entity_ids[0]}
                    }
                ),
            ),
            (
                "Name Match",
                QueryOptions(
                    expression={"name": {"operator": "eq", "value": "PerfTest-0"}}
                ),
            ),
            (
                "Complex Filter",
                QueryOptions(
                    expression={
                        "and": [
                            {"value": {"operator": "gt", "value": 50}},
                            {"active": {"operator": "eq", "value": True}},
                        ]
                    }
                ),
            ),
        ]

        results = []

        for scenario_name, options in test_scenarios:
            metrics = PerformanceMetrics(
                f"find_one_{scenario_name.replace(' ', '_').lower()}"
            )

            # Run multiple iterations to get stable results
            for i in range(20):
                start_time = time.time()
                try:
                    await repo.find_one(logger, options)
                    end_time = time.time()
                    metrics.record_latency(end_time - start_time)
                except ObjectNotFoundException:
                    logger.warning(f"No entity found for scenario: {scenario_name}")

            # Store results
            summary = metrics.summary()
            results.append(summary)

            print(f"\nFind One ({scenario_name}) Results:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

        # Get repository name
        repo_name = repo.__class__.__name__

        # Create latency comparison chart
        scenario_names = [result["name"].split("_", 2)[2] for result in results]
        latencies = [result["avg_latency_ms"] for result in results]

        plt.figure(figsize=(10, 6))
        plt.bar(scenario_names, latencies)
        plt.title(f"{repo_name} - Find One Average Latency by Scenario")
        plt.xlabel("Scenario")
        plt.ylabel("Average Latency (ms)")
        plt.savefig(f"{output_dir}/{repo_name}_find_one_latency.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_find_one_performance.txt", "w") as f:
            f.write(f"FIND ONE PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for scenario, result in zip(test_scenarios, results):
                scenario_name = scenario[0]
                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 20 + "\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_count_performance(repository_factory, logger):
    """
    Test the performance of count operations.
    Measures latency of counting entities with different query options.
    """
    # Configuration
    entity_count = 5000

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    await transaction.setup(entity_count)

    try:
        # Test scenarios for count operation
        test_scenarios = [
            ("Count All", None),
            (
                "Count Active",
                QueryOptions(expression={"active": {"operator": "eq", "value": True}}),
            ),
            (
                "Count by Name Pattern",
                QueryOptions(
                    expression={"name": {"operator": "like", "value": "PerfTest-%"}}
                ),
            ),
            (
                "Count with Complex Filter",
                QueryOptions(
                    expression={
                        "and": [
                            {"value": {"operator": "gt", "value": 50}},
                            {"active": {"operator": "eq", "value": True}},
                        ]
                    }
                ),
            ),
        ]

        results = []

        for scenario_name, options in test_scenarios:
            metrics = PerformanceMetrics(
                f"count_{scenario_name.replace(' ', '_').lower()}"
            )
            count_value = 0

            # Run multiple iterations to get stable results
            for i in range(10):
                start_time = time.time()
                count_value = await repo.count(logger, options)
                end_time = time.time()

                metrics.record_latency(end_time - start_time)

            # Store results
            summary = metrics.summary()
            summary["count_value"] = count_value
            results.append(summary)

            print(f"\nCount ({scenario_name}) Results:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

        # Get repository name
        repo_name = repo.__class__.__name__

        # Create latency comparison chart
        scenario_names = [result["name"].split("_", 1)[1] for result in results]
        latencies = [result["avg_latency_ms"] for result in results]

        plt.figure(figsize=(10, 6))
        plt.bar(scenario_names, latencies)
        plt.title(f"{repo_name} - Count Operation Average Latency by Scenario")
        plt.xlabel("Scenario")
        plt.ylabel("Average Latency (ms)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{repo_name}_count_latency.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_count_performance.txt", "w") as f:
            f.write(f"COUNT PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")
            for scenario, result in zip(test_scenarios, results):
                scenario_name = scenario[0]
                f.write(f"Scenario: {scenario_name}\n")
                f.write("-" * 20 + "\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_upsert_performance(repository_factory, logger):
    """
    Test the performance of upsert operations.
    Compares performance of insert (new entity) vs update (existing entity).
    """
    # Configuration
    entity_count = 500
    update_iterations = 5

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Create metrics
    insert_metrics = PerformanceMetrics("upsert_insert")
    update_metrics = PerformanceMetrics("upsert_update")

    # Generate test entities
    entities = []
    for i in range(entity_count):
        entity = Entity(
            name=f"UpsertTest-{i}",
            value=i % 100,
            active=i % 2 == 0,
            tags=[f"tag{i % 5}"],
            metadata={"test": "upsert", "index": i},
        )
        entities.append(entity)

    entity_ids = []

    try:
        # First pass: insert new entities
        for entity in entities:
            start_time = time.time()
            await repo.upsert(entity, logger)
            end_time = time.time()

            insert_metrics.record_latency(end_time - start_time)
            entity_ids.append(entity.id)

        # Multiple passes: update existing entities
        for iteration in range(update_iterations):
            for i, entity in enumerate(entities):
                # Modify the entity
                entity.value = (entity.value + 100) % 1000
                entity.updated_at = datetime.utcnow()

                start_time = time.time()
                await repo.upsert(entity, logger)
                end_time = time.time()

                update_metrics.record_latency(end_time - start_time)

        # Print results
        insert_summary = insert_metrics.summary()
        update_summary = update_metrics.summary()

        print(f"\nUpsert (Insert) Performance Results:")
        for key, value in insert_summary.items():
            print(f"  {key}: {value}")

        print(f"\nUpsert (Update) Performance Results:")
        for key, value in update_summary.items():
            print(f"  {key}: {value}")

        # Get repository name
        repo_name = repo.__class__.__name__

        # Create comparison chart
        operations = ["Insert", "Update"]
        latencies = [insert_summary["avg_latency_ms"], update_summary["avg_latency_ms"]]

        plt.figure(figsize=(8, 6))
        plt.bar(operations, latencies)
        plt.title(f"{repo_name} - Upsert Average Latency: Insert vs Update")
        plt.xlabel("Operation Type")
        plt.ylabel("Average Latency (ms)")
        plt.savefig(f"{output_dir}/{repo_name}_upsert_latency.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_upsert_performance.txt", "w") as f:
            f.write(f"UPSERT PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write("UPSERT (INSERT) RESULTS:\n")
            f.write("-" * 20 + "\n")
            for key, value in insert_summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

            f.write("UPSERT (UPDATE) RESULTS:\n")
            f.write("-" * 20 + "\n")
            for key, value in update_summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

    finally:
        # Cleanup
        for i in range(0, len(entity_ids), 100):
            batch = entity_ids[i : i + 100]
            options = QueryOptions(
                expression={"id": {"operator": "in", "value": batch}}
            )
            try:
                await repo.delete_many(options, logger)
            except Exception as e:
                logger.error(f"Error cleaning up upsert test: {e}")


# =============================================================================
# Performance Tests - Complex Operations
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_complex_update_performance(repository_factory, logger):
    """
    Test the performance of different types of update operations.
    Compares latency of set, push, pop, pull, and unset operations.
    """
    # Configuration
    entity_count = 500
    operations_per_type = 20

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    entity_ids = await transaction.setup(entity_count)

    try:
        operation_metrics = {
            "set": PerformanceMetrics("set_operation"),
            "push": PerformanceMetrics("push_operation"),
            "pop": PerformanceMetrics("pop_operation"),
            "pull": PerformanceMetrics("pull_operation"),
            "unset": PerformanceMetrics("unset_operation"),
            "nested_set": PerformanceMetrics("nested_set_operation"),
            "nested_push": PerformanceMetrics("nested_push_operation"),
        }

        # Test each operation type
        for i in range(operations_per_type):
            entity_id = entity_ids[i % len(entity_ids)]
            options = QueryOptions(
                expression={"id": {"operator": "eq", "value": entity_id}}
            )

            # Set operation
            update = Update().set("value", i).set("updated_at", datetime.utcnow())
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["set"].record_latency(time.time() - start_time)

            # Push operation
            update = Update().push("tags", f"perf_tag_{i}")
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["push"].record_latency(time.time() - start_time)

            # Pop operation
            update = Update().pop("tags", 1)  # Remove last tag
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["pop"].record_latency(time.time() - start_time)

            # Pull operation
            update = Update().pull("tags", "tag1")
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["pull"].record_latency(time.time() - start_time)

            # Unset operation (on metadata.index)
            update = Update().unset("metadata.index")
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["unset"].record_latency(time.time() - start_time)

            # Nested set operation
            update = Update().set("metadata.test_value", i)
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["nested_set"].record_latency(time.time() - start_time)

            # Nested push operation
            update = Update().push("profile.emails", f"perf{i}@example.com")
            start_time = time.time()
            await repo.update_one(options, update, logger)
            operation_metrics["nested_push"].record_latency(time.time() - start_time)

        # Collect and display results
        results = []
        for op_type, metrics in operation_metrics.items():
            metrics.operation_count = operations_per_type
            summary = metrics.summary()
            results.append(summary)
            print(f"\n{op_type.capitalize()} Operation Performance:")
            for key, value in summary.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # Create comparison chart
        op_types = [result["name"].split("_")[0] for result in results]
        avg_latencies = [result["avg_latency_ms"] for result in results]

        # Get repository name
        repo_name = repo.__class__.__name__

        plt.figure(figsize=(12, 6))
        plt.bar(op_types, avg_latencies)
        plt.title(f"{repo_name} - Average Latency by Update Operation Type")
        plt.xlabel("Operation Type")
        plt.ylabel("Average Latency (ms)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{repo_name}_update_operation_latency.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_complex_update_performance.txt", "w") as f:
            f.write(f"COMPLEX UPDATE PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")

            for op_type, metrics_obj in operation_metrics.items():
                summary = metrics_obj.summary()
                f.write(f"{op_type.upper()} OPERATION:\n")
                f.write("-" * 20 + "\n")
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_concurrent_performance(repository_factory, logger):
    """
    Test performance under concurrent operations.
    Measures how throughput changes with different levels of concurrency.
    """
    # Configuration
    entity_count = 1000
    operations_per_worker = 50
    concurrency_levels = [1, 5, 10, 20]

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    # Setup with the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    entity_ids = await transaction.setup(entity_count)

    try:
        # Define worker functions for different operation types
        async def get_worker(
            worker_id: int, num_ops: int, entity_ids: List[str]
        ) -> float:
            """Worker that performs get operations."""
            start_time = time.time()
            for i in range(num_ops):
                entity_id = entity_ids[(worker_id * 100 + i) % len(entity_ids)]
                await repo.get(entity_id, logger)
            return time.time() - start_time

        async def update_worker(
            worker_id: int, num_ops: int, entity_ids: List[str]
        ) -> float:
            """Worker that performs update operations."""
            start_time = time.time()
            for i in range(num_ops):
                entity_id = entity_ids[(worker_id * 100 + i) % len(entity_ids)]
                update = Update().set("value", i).set("updated_at", datetime.utcnow())
                options = QueryOptions(
                    expression={"id": {"operator": "eq", "value": entity_id}}
                )
                await repo.update_one(options, update, logger)
            return time.time() - start_time

        # Test operations
        operations = [("get", get_worker), ("update", update_worker)]

        all_results = {}

        for op_name, worker_func in operations:
            results = []

            # Test different concurrency levels
            for num_workers in concurrency_levels:
                metrics = PerformanceMetrics(f"{op_name}_concurrent_{num_workers}")
                metrics.start()

                # Create and run workers
                workers = [
                    worker_func(i, operations_per_worker, entity_ids)
                    for i in range(num_workers)
                ]
                worker_times = await asyncio.gather(*workers)

                # Calculate metrics
                total_ops = num_workers * operations_per_worker
                metrics.end(total_ops)

                # Also record individual operation latencies
                for worker_time in worker_times:
                    avg_op_time = worker_time / operations_per_worker
                    for _ in range(operations_per_worker):
                        metrics.record_latency(avg_op_time)

                # Store results
                summary = metrics.summary()
                results.append(summary)
                print(
                    f"\n{op_name.capitalize()} Concurrent Operations ({num_workers} workers) Results:"
                )
                for key, value in summary.items():
                    print(f"  {key}: {value}")

            all_results[op_name] = results

        # Create throughput vs. concurrency chart
        plt.figure(figsize=(12, 8))

        # Plot each operation type
        for idx, (op_name, results) in enumerate(all_results.items()):
            concurrency = [int(result["name"].split("_")[-1]) for result in results]
            throughputs = [result["throughput_ops_per_sec"] for result in results]

            plt.subplot(len(operations), 1, idx + 1)
            plt.plot(concurrency, throughputs, marker="o", label=op_name)
            plt.title(f"{op_name.capitalize()} Throughput vs. Concurrency")
            plt.xlabel("Number of Concurrent Workers")
            plt.ylabel("Operations per Second")
            plt.grid(True)
            plt.legend()

        # Get repository name
        repo_name = repo.__class__.__name__

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{repo_name}_concurrent_operations.png")

        # Generate text output
        with open(f"{output_dir}/{repo_name}_concurrent_performance.txt", "w") as f:
            f.write(f"CONCURRENT OPERATIONS PERFORMANCE RESULTS FOR {repo_name}\n")
            f.write("=" * 50 + "\n\n")

            for op_name, results_list in all_results.items():
                f.write(f"{op_name.upper()} OPERATIONS:\n")
                f.write("-" * 20 + "\n")

                for result, workers in zip(results_list, concurrency_levels):
                    f.write(f"Concurrency Level: {workers} workers\n")
                    for key, value in result.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

    finally:
        # Cleanup
        await transaction.cleanup()


# =============================================================================
# Performance Test for Scalability
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_scalability(repository_factory, logger):
    """
    Test how repository performance scales with database size.
    Measures operation latency as the number of entities increases.
    """
    # Configuration - test with progressively larger datasets
    dataset_sizes = [100, 500, 1000, 2000]
    operations_per_size = 50

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    repo_name = repo.__class__.__name__

    # Operations to test
    operations = [
        ("get", lambda entity_id: repo.get(entity_id, logger)),
        (
            "update",
            lambda entity_id: repo.update_one(
                QueryOptions(expression={"id": {"operator": "eq", "value": entity_id}}),
                Update().set("value", 999).set("updated_at", datetime.utcnow()),
                logger,
            ),
        ),
        (
            "find",
            lambda entity_id: repo.find_one(
                logger,
                QueryOptions(expression={"value": {"operator": "gt", "value": 50}}),
            ),
        ),
        ("count", lambda entity_id: repo.count(logger)),
    ]

    all_results = {}
    entity_ids_by_size = {}

    try:
        # Test with increasing dataset sizes
        for size in dataset_sizes:
            print(f"\nTesting with dataset size: {size}")

            # Generate test data for this size
            transaction = TestTransaction(lambda _: repo, Entity, logger)
            entity_ids = await transaction.setup(size)
            entity_ids_by_size[size] = entity_ids

            size_results = {}

            # Test each operation type
            for op_name, op_func in operations:
                metrics = PerformanceMetrics(f"{op_name}_scalability_{size}")

                # Perform operations
                for i in range(min(operations_per_size, len(entity_ids))):
                    # Use entity_id only for operations that need it
                    entity_id = entity_ids[i % len(entity_ids)]

                    start_time = time.time()
                    try:
                        await op_func(entity_id)
                    except Exception as e:
                        logger.warning(f"Error in {op_name} operation: {e}")
                    metrics.record_latency(time.time() - start_time)

                # Calculate metrics
                metrics.operation_count = operations_per_size
                summary = metrics.summary()

                if op_name not in all_results:
                    all_results[op_name] = []
                all_results[op_name].append(summary)

                size_results[op_name] = summary["avg_latency_ms"]

                print(f"  {op_name}: {summary['avg_latency_ms']:.2f} ms avg latency")

            # Clean up this size's data before moving to next size
            # Keep the largest dataset for final cleanup
            if size < dataset_sizes[-1]:
                await transaction.cleanup()
                del entity_ids_by_size[size]

        # Create scalability chart
        plt.figure(figsize=(12, 8))

        # Plot each operation type
        for idx, (op_name, results) in enumerate(all_results.items()):
            sizes = dataset_sizes[: len(results)]
            latencies = [result["avg_latency_ms"] for result in results]

            plt.subplot(2, 2, idx + 1)
            plt.plot(sizes, latencies, marker="o", label=op_name)
            plt.title(f"{op_name.capitalize()} Latency vs. Database Size")
            plt.xlabel("Number of Entities")
            plt.ylabel("Average Latency (ms)")
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{repo_name}_scalability.png")

    finally:
        # Cleanup remaining data
        for size, entity_ids in entity_ids_by_size.items():
            for i in range(0, len(entity_ids), 100):
                batch = entity_ids[i : i + 100]
                options = QueryOptions(
                    expression={"id": {"operator": "in", "value": batch}}
                )
                try:
                    await repo.delete_many(options, logger)
                except Exception as e:
                    logger.error(f"Error cleaning up scalability test: {e}")


# =============================================================================
# Comprehensive Performance Report
# =============================================================================


@pytest.mark.parametrize(
    "repository_factory", REPOSITORY_IMPLEMENTATIONS, indirect=True
)
async def test_generate_performance_report(repository_factory, logger):
    """
    Generate a comprehensive performance report for all repository implementations.
    Runs benchmarks for all major operations and generates visualizations.
    """
    # Only run if explicitly requested to save test time
    if not os.environ.get("RUN_PERF_REPORT", "false").lower() in ("true", "1", "yes"):
        pytest.skip(
            "Skipping full performance report. Set RUN_PERF_REPORT=true to generate."
        )

    # Configuration
    entity_count = 2000
    num_operations = 100

    # Create a single repository instance to use throughout
    repo = repository_factory(Entity)

    repo_name = repo.__class__.__name__

    print(f"\nGenerating performance report for {repo_name}...")

    # Create test data using the same repo instance
    transaction = TestTransaction(lambda _: repo, Entity, logger)
    entity_ids = await transaction.setup(entity_count)

    try:
        # Define test scenarios
        operations = [
            ("Get by ID", lambda: repo.get(entity_ids[0], logger)),
            (
                "Find one",
                lambda: repo.find_one(
                    logger,
                    QueryOptions(
                        expression={"name": {"operator": "like", "value": "PerfTest-%"}}
                    ),
                ),
            ),
            ("Count", lambda: repo.count(logger)),
            (
                "Update one",
                lambda: repo.update_one(
                    QueryOptions(
                        expression={"id": {"operator": "eq", "value": entity_ids[0]}}
                    ),
                    Update().set("value", 999),
                    logger,
                ),
            ),
            (
                "Update many (batch=100)",
                lambda: repo.update_many(
                    QueryOptions(
                        expression={
                            "name": {"operator": "like", "value": "PerfTest-%"}
                        },
                        limit=100,
                    ),
                    Update().set("updated_at", datetime.utcnow()),
                    logger,
                ),
            ),
            (
                "Simple list (limit=100)",
                lambda: fetch_entities(repo.list(logger, QueryOptions(limit=100)), 100),
            ),
            (
                "Filtered list",
                lambda: fetch_entities(
                    repo.list(
                        logger,
                        QueryOptions(
                            expression={"active": {"operator": "eq", "value": True}},
                            limit=100,
                        ),
                    ),
                    100,
                ),
            ),
        ]

        # Helper function to fetch entities from an async generator
        async def fetch_entities(gen, limit):
            result = []
            count = 0
            async for entity in gen:
                result.append(entity)
                count += 1
                if count >= limit:
                    break
            return result

        results = []

        # In test_generate_performance_report function, update the operation loop:
        for scenario_name, op_func in operations:
            metrics = PerformanceMetrics(f"{repo_name}_{scenario_name}")

            # Start timing the batch of operations
            metrics.start()

            # Perform multiple operations
            for i in range(num_operations):
                try:
                    start_time = time.time()
                    await op_func()
                    end_time = time.time()
                    metrics.record_latency(end_time - start_time)
                except Exception as e:
                    logger.warning(f"Error in {scenario_name}: {e}")

            # End timing and record the operation count
            metrics.end(num_operations)

        # Create DataFrame for analysis
        df = pd.DataFrame(results)

        # Generate report
        print(f"\nPerformance Summary for {repo_name}:")
        print(
            df[
                [
                    "scenario",
                    "avg_latency_ms",
                    "p95_latency_ms",
                    "throughput_ops_per_sec",
                ]
            ].to_string(index=False)
        )

        # Generate charts
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.bar(df["scenario"], df["avg_latency_ms"])
        plt.title(f"{repo_name} - Average Latency by Operation Type")
        plt.ylabel("Latency (ms)")
        plt.xticks(rotation=45, ha="right")

        plt.subplot(2, 1, 2)
        plt.bar(df["scenario"], df["throughput_ops_per_sec"])
        plt.title(f"{repo_name} - Throughput by Operation Type")
        plt.ylabel("Operations per Second")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{repo_name}_performance.png")

        # Generate text report
        with open(
            f"{output_dir}/{repo_name}_complete_performance_report.txt", "w"
        ) as f:
            f.write(f"COMPREHENSIVE PERFORMANCE REPORT FOR {repo_name}\n")
            f.write("=" * 80 + "\n\n")

            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(
                df[
                    [
                        "scenario",
                        "avg_latency_ms",
                        "p95_latency_ms",
                        "throughput_ops_per_sec",
                    ]
                ].to_string(index=False)
            )
            f.write("\n\n")

            f.write("DETAILED METRICS:\n")
            f.write("-" * 20 + "\n")
            for _, row in df.iterrows():
                f.write(f"Operation: {row['scenario']}\n")
                for key, value in row.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

        return df  # Return DataFrame for potential further analysis

    finally:
        # Cleanup
        await transaction.cleanup()


# Helper to make a dir for output files
output_dir = "performance-report-output"
os.makedirs(output_dir, exist_ok=True)
