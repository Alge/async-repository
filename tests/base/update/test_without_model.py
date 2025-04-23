# tests/base/update/test_without_model.py

import pytest
from typing import List, Type, Optional, TypeVar  # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,
    SetOperation,
    IncrementOperation,
    PushOperation,
    MinOperation,
    MaxOperation,
    MultiplyOperation,
    PopOperation,
    PullOperation,
    UnsetOperation,
    # No need for validator exceptions here as we test without model
)

from tests.base.conftest import assert_operation_present


def test_update_without_model():
    """Test basic operations without a model class build the correct list."""
    update = Update()  # No model provided

    # Basic operations with string fields
    update.set("name", "John Doe")
    update.increment("counter", 5)
    update.push("tags", "new_tag")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3

    # Check the operations in the list
    assert_operation_present(result, SetOperation, "name", {"value": "John Doe"})
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})


def test_fields_proxy_without_model():
    """Test using the fields proxy without a model class builds correct list."""
    update = Update()  # No model provided

    # Using dynamic fields proxy
    update.set(update.fields.name, "John Doe")
    update.increment(update.fields.counter, 5)
    update.push(update.fields.tags, "new_tag")

    # Nested fields access
    update.set(update.fields.user.profile.age, 30)
    update.set(update.fields.preferences.theme, "dark")

    # Multi-level nesting
    update.push(update.fields.posts.comments.replies, "New reply")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 6

    # Check operations using assert_operation_present
    assert_operation_present(result, SetOperation, "name", {"value": "John Doe"})
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})
    assert_operation_present(result, SetOperation, "user.profile.age", {"value": 30})
    assert_operation_present(
        result, SetOperation, "preferences.theme", {"value": "dark"}
    )
    assert_operation_present(
        result, PushOperation, "posts.comments.replies", {"items": ["New reply"]}
    )


def test_complex_operations_without_model():
    """Test more complex operations without a model class build correct list."""
    update = Update()  # No model provided

    # Min/max/mul operations
    update.min(update.fields.score, 0)
    update.max(update.fields.score, 100)  # Note: Multiple ops on score allowed
    update.mul(update.fields.multiplier, 1.5)

    # Array operations
    update.push(update.fields.items, {"id": 1, "name": "Item 1"})
    update.pop(update.fields.recent_items, -1)  # Remove first item, position = -1
    update.pull(update.fields.tags, "old_tag")

    # Nested operations
    update.unset(update.fields.user.temporary_data)
    update.increment(update.fields.stats.visits, 1)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 8  # All operations are added

    # Check operations using assert_operation_present
    assert_operation_present(result, MinOperation, "score", {"value": 0})
    assert_operation_present(result, MaxOperation, "score", {"value": 100})
    assert_operation_present(result, MultiplyOperation, "multiplier", {"factor": 1.5})
    assert_operation_present(
        result, PushOperation, "items", {"items": [{"id": 1, "name": "Item 1"}]}
    )
    assert_operation_present(result, PopOperation, "recent_items", {"position": -1})
    assert_operation_present(
        result, PullOperation, "tags", {"value_or_condition": "old_tag"}
    )
    assert_operation_present(result, UnsetOperation, "user.temporary_data")
    assert_operation_present(result, IncrementOperation, "stats.visits", {"amount": 1})


def test_increment_restrictions_without_model():
    """Test that increment restrictions still apply without a model."""
    update = Update()  # No model provided

    # First increment is fine
    update.increment(update.fields.counter, 5)

    # Second increment on same field should be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.counter, 3)

    # Decrement after increment should also be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.decrement(update.fields.counter, 2)

    # But increment on different field is okay
    update.increment(update.fields.another_counter, 10)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2  # Only the two successful increments

    # Check the operations that were successfully added
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})
    assert_operation_present(
        result, IncrementOperation, "another_counter", {"amount": 10}
    )
