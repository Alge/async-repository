# tests/base/update/test_increment.py

import pytest
from async_repository.base.update import (
    Update,
    IncrementOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import User, NumericModel, NestedTypes
from tests.base.conftest import assert_operation_present


def test_increment_with_type_validation():
    """Test that increment operations are type validated."""
    update = Update(User)

    # Valid increment - one per field
    update.increment(update.fields.points, 10)
    update.increment(update.fields.balance, 5.5)
    update.increment(update.fields.score, 3)  # Union type

    # Invalid field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.increment("non_existent", 5)

    # Invalid field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):
        update.increment(update.fields.name, 5)

    # Invalid amount type - Method raises TypeError
    with pytest.raises(TypeError):
        update.increment(update.fields.score, "5")

    with pytest.raises(TypeError):
        update.increment(update.fields.score, [5])


def test_multiple_increment_rejected():
    """Test that multiple increment/decrement operations on same field are rejected."""
    update = Update(User)

    # First increment is allowed
    update.increment(update.fields.points, 10)

    # Second increment on same field should be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.points, 5)

    # Default value increment is also rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.points)

    # But increment on a different field is fine
    update.increment(update.fields.balance, 5.5)

    # Verify only original increment operations are in the update
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2  # Only the first points and the balance increment

    assert_operation_present(result, IncrementOperation, "points", {"amount": 10})
    assert_operation_present(result, IncrementOperation, "balance", {"amount": 5.5})


def test_increment_after_decrement_rejected():
    """Test that increment after decrement on the same field is rejected."""
    update = Update(User)

    # First operation (decrement) is allowed
    update.decrement(update.fields.points, 5)

    # Increment on the same field should be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.points, 10)

    # Verify the original decrement remains
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    # Decrement is stored as IncrementOperation with negative amount
    assert_operation_present(result, IncrementOperation, "points", {"amount": -5})


def test_increment_with_optional_and_union_types():
    """Test increment with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Each field can only have one increment
    update.increment(update.fields.union_numeric, 5)
    # Incrementing Optional[int] should work
    update.increment(update.fields.optional_int, 10)

    # Second increment on the same field is rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.union_numeric, 3.14)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "union_numeric", {"amount": 5})
    assert_operation_present(result, IncrementOperation, "optional_int", {"amount": 10})


def test_increment_nested_fields():
    """Test increment operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested increment
    update.increment(update.fields.nested.inner.val, 5)

    # Multiple increments on same nested field are rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment(update.fields.nested.inner.val, 3)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.increment("nested.non_existent", 5)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):
        update.increment(update.fields.str_list, 5)  # str_list is not numeric


def test_increment_without_type_validation():
    """Test that increment works without a model type (no validation)."""
    update = Update()  # No model type

    # Each field gets only one increment
    update.increment("counter", 5)
    update.increment("value", 10.5)
    update.increment("score")  # Default increment by 1
    update.increment("nested.counter", 3)

    # Multiple increments on the same field are rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):
        update.increment("counter", 2)

    # Build should succeed with original increments
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4

    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})
    assert_operation_present(result, IncrementOperation, "value", {"amount": 10.5})
    assert_operation_present(result, IncrementOperation, "score", {"amount": 1})
    assert_operation_present(
        result, IncrementOperation, "nested.counter", {"amount": 3}
    )


def test_increment_build_result():
    """Test that increment operations build the correct agnostic operation list."""
    update = Update()
    # Use separate fields
    update.increment("views", 1)
    update.increment("score", 5.5)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "views", {"amount": 1})
    assert_operation_present(result, IncrementOperation, "score", {"amount": 5.5})


def test_increment_zero():
    """Test incrementing by zero."""
    update = Update().increment("counter", 0)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 0})


def test_increment_negative():
    """Test that increment can handle negative values (acts like decrement)."""
    update = Update().increment("counter", -5)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": -5})


def test_increment_float_precision():
    """Test that increment handles float precision."""
    # Using separate fields since multiple increments are not allowed
    update = Update().increment("value1", 0.1).increment("value2", 0.2)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "value1", {"amount": 0.1})
    assert_operation_present(result, IncrementOperation, "value2", {"amount": 0.2})
