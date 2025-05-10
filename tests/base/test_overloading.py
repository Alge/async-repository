# tests/base/update/test_field_conflicts.py

import pytest
from async_repository.base.update import (
    Update,
    SetOperation,
    PushOperation,
    PopOperation,
    PullOperation,
    UnsetOperation,
    IncrementOperation,
    MinOperation,
    MaxOperation,
    MultiplyOperation,
)
from tests.conftest import Entity
from tests.base.conftest import assert_operation_present


# Tests for basic field conflict detection (same field path)

def test_set_operation_conflict_detection():
    """Test that set operations detect field conflicts."""
    update = Update(Entity)
    update.set("name", "Test")

    with pytest.raises(ValueError) as excinfo:
        update.set("name", "Conflict")

    assert "name" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_push_operation_conflict_detection():
    """Test that push operations detect field conflicts."""
    update = Update(Entity)
    update.push("tags", "tag1")

    with pytest.raises(ValueError) as excinfo:
        update.push("tags", "tag2")

    assert "tags" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_pop_operation_conflict_detection():
    """Test that pop operations detect field conflicts."""
    update = Update(Entity)
    update.pop("tags")

    with pytest.raises(ValueError) as excinfo:
        update.pop("tags", -1)

    assert "tags" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_pull_operation_conflict_detection():
    """Test that pull operations detect field conflicts."""
    update = Update(Entity)
    update.pull("tags", "tag1")

    with pytest.raises(ValueError) as excinfo:
        update.pull("tags", "tag2")

    assert "tags" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_unset_operation_conflict_detection():
    """Test that unset operations detect field conflicts."""
    update = Update(Entity)
    update.unset("metadata.key1")

    with pytest.raises(ValueError) as excinfo:
        update.unset("metadata.key1")

    assert "metadata.key1" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_increment_operation_conflict_detection():
    """Test that increment operations detect field conflicts."""
    update = Update(Entity)
    update.increment("value", 5)

    with pytest.raises(ValueError) as excinfo:
        update.increment("value", 10)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_decrement_operation_conflict_detection():
    """Test that decrement operations detect field conflicts."""
    update = Update(Entity)
    update.decrement("value", 5)

    with pytest.raises(ValueError) as excinfo:
        update.decrement("value", 10)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_min_operation_conflict_detection():
    """Test that min operations detect field conflicts."""
    update = Update(Entity)
    update.min("value", 0)

    with pytest.raises(ValueError) as excinfo:
        update.min("value", 10)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_max_operation_conflict_detection():
    """Test that max operations detect field conflicts."""
    update = Update(Entity)
    update.max("value", 100)

    with pytest.raises(ValueError) as excinfo:
        update.max("value", 200)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_multiply_operation_conflict_detection():
    """Test that multiply operations detect field conflicts."""
    update = Update(Entity)
    update.mul("value", 2)

    with pytest.raises(ValueError) as excinfo:
        update.mul("value", 3)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


# Tests for different operation types on the same field

def test_push_pop_conflict():
    """Test that push conflicts with pop on the same field."""
    update = Update(Entity)
    update.push("tags", "tag1")

    with pytest.raises(ValueError) as excinfo:
        update.pop("tags")

    assert "tags" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_set_increment_conflict():
    """Test that set conflicts with increment on the same field."""
    update = Update(Entity)
    update.set("value", 100)

    with pytest.raises(ValueError) as excinfo:
        update.increment("value", 5)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_increment_multiply_conflict():
    """Test that increment conflicts with multiply on the same field."""
    update = Update(Entity)
    update.increment("float_value", 10.0)

    with pytest.raises(ValueError) as excinfo:
        update.mul("float_value", 2.0)

    assert "float_value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_unset_set_conflict():
    """Test that unset conflicts with set on the same field."""
    update = Update(Entity)
    update.unset("metadata.key1")

    with pytest.raises(ValueError) as excinfo:
        update.set("metadata.key1", "new value")

    assert "metadata.key1" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_min_max_conflict():
    """Test that min conflicts with max on the same field."""
    update = Update(Entity)
    update.min("value", 0)

    with pytest.raises(ValueError) as excinfo:
        update.max("value", 100)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_increment_decrement_conflict():
    """Test that increment conflicts with decrement on the same field."""
    update = Update(Entity)
    update.increment("value", 5)

    with pytest.raises(ValueError) as excinfo:
        update.decrement("value", 3)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_decrement_increment_conflict():
    """Test that decrement conflicts with increment on the same field."""
    update = Update(Entity)
    update.decrement("value", 3)

    with pytest.raises(ValueError) as excinfo:
        update.increment("value", 5)

    assert "value" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


# Tests for nested field conflicts

def test_nested_field_exact_conflict():
    """Test that operations on the same nested field raise ValueError."""
    update = Update(Entity)
    update.set("metadata.name", "Test Name")

    with pytest.raises(ValueError) as excinfo:
        update.set("metadata.name", "New Name")

    assert "metadata.name" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


def test_deeply_nested_field_conflict():
    """Test that operations on the same deeply nested field raise ValueError."""
    update = Update(Entity)
    update.set("metadata.stats.views", 100)

    with pytest.raises(ValueError) as excinfo:
        update.unset("metadata.stats.views")

    assert "metadata.stats.views" in str(excinfo.value)
    assert "already has an operation" in str(excinfo.value)


# Tests for parent-child field conflicts

def test_parent_to_child_field_conflict():
    """Test conflict when setting a child field after its parent is already set."""
    update = Update(Entity)
    update.set("metadata", {"key1": "value1", "key2": 42})

    # Operation on a child field after parent field has an operation
    with pytest.raises(ValueError) as excinfo:
        update.set("metadata.key1", "new value")

    assert "metadata" in str(excinfo.value)
    assert "Parent-child field conflicts are not allowed" in str(excinfo.value)


def test_child_to_parent_field_conflict():
    """Test conflict when setting a parent field after its child is already set."""
    update = Update(Entity)
    update.set("metadata.key1", "value1")

    # Operation on a parent field after child field has an operation
    with pytest.raises(ValueError) as excinfo:
        update.set("metadata", {"key1": "new value", "key2": 42})

    assert "metadata" in str(excinfo.value)
    assert "Parent-child field conflicts are not allowed" in str(excinfo.value)


def test_deeply_nested_parent_to_child_conflict():
    """Test that parent-to-child conflicts are detected with deeply nested objects."""
    update = Update(Entity)

    # Set a deeply nested parent field
    update.set("metadata.location.address",
               {"street": "123 Main St", "city": "New York"})

    # Attempt to set a child field
    with pytest.raises(ValueError) as excinfo:
        update.set("metadata.location.address.street", "456 Elm St")

    assert "metadata.location.address" in str(excinfo.value)
    assert "Parent-child field conflicts are not allowed" in str(excinfo.value)


def test_deeply_nested_child_to_parent_conflict():
    """Test that child-to-parent conflicts are detected with deeply nested objects."""
    update = Update(Entity)
    update.set("metadata.location.address.city", "Boston")

    # Attempt to set the parent field
    with pytest.raises(ValueError) as excinfo:
        update.set("metadata.location.address",
                   {"street": "789 Oak St", "city": "Chicago"})

    assert "metadata.location.address" in str(excinfo.value)
    assert "Parent-child field conflicts are not allowed" in str(excinfo.value)


# Tests for operations on different fields (positive cases)

def test_operations_on_different_fields():
    """Test that operations on different fields work correctly."""
    update = Update(Entity)
    update.set("name", "Entity Name")
    update.set("owner", "user@example.com")
    update.push("tags", "tag1")
    update.increment("value", 50)

    result = update.build()
    assert len(result) == 4
    assert_operation_present(result, SetOperation, "name", {"value": "Entity Name"})
    assert_operation_present(result, SetOperation, "owner",
                             {"value": "user@example.com"})
    assert_operation_present(result, PushOperation, "tags", {"items": ["tag1"]})
    assert_operation_present(result, IncrementOperation, "value", {"amount": 50})


def test_numeric_operations_on_different_fields():
    """Test that numeric operations on different fields work correctly."""
    update = Update(Entity)
    update.increment("value", 5)
    update.min("float_value", 10.0)

    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "value", {"amount": 5})
    assert_operation_present(result, MinOperation, "float_value", {"value": 10.0})


def test_sibling_field_updates_work():
    """Test that updating multiple sibling fields in a nested object works."""
    update = Update(Entity)

    # Set two different fields under the same parent
    update.set("metadata.city", "New York")
    update.set("metadata.zipcode", "10001")

    # This should build successfully with no conflicts
    result = update.build()

    # Verify both operations were included
    assert len(result) == 2
    assert_operation_present(result, SetOperation, "metadata.city",
                             {"value": "New York"})
    assert_operation_present(result, SetOperation, "metadata.zipcode",
                             {"value": "10001"})