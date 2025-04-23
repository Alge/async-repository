# tests/base/update/test_decrement.py

import pytest
from typing import List, Type, Optional, TypeVar  # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,  # Import base operation class
    IncrementOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import User, NumericModel, NestedTypes

from tests.base.conftest import assert_operation_present



def test_decrement_with_type_validation():
    """Test that decrement operations are type validated."""
    update = Update(User)

    # Valid decrements - one per field
    update.decrement(update.fields.points, 10)
    update.decrement(update.fields.balance, 5.5)
    update.decrement(update.fields.score, 3)  # Union type

    # Invalid field (non-existent) - Validator raises InvalidPathError via increment
    with pytest.raises(
        InvalidPathError
    ):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CORRECTED EXPECTED EXCEPTION
        update.decrement("non_existent", 5)

    # Invalid field (non-numeric) - Validator raises ValueTypeError via increment
    with pytest.raises(
        ValueTypeError
    ):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CORRECTED EXPECTED EXCEPTION
        update.decrement(update.fields.name, 5)

    # Invalid amount type - Method raises TypeError directly
    with pytest.raises(TypeError):
        update.decrement(update.fields.score, "5")

    with pytest.raises(TypeError):
        update.decrement(update.fields.score, [5])


def test_multiple_decrement_rejected():
    """Test that multiple decrement/increment operations on same field are rejected."""
    update = Update(User)

    # First decrement is allowed
    update.decrement(update.fields.points, 10)

    # Second decrement on same field should be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement(update.fields.points, 5)

    # Default value decrement is also rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement(update.fields.points)

    # But decrement on a different field is fine
    update.decrement(update.fields.balance, 5.5)

    # Verify only original decrement operations are in the update
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2  # Only the first points and the balance decrement

    assert_operation_present(result, IncrementOperation, "points", {"amount": -10})
    assert_operation_present(result, IncrementOperation, "balance", {"amount": -5.5})


def test_decrement_after_increment_rejected():
    """Test that decrement after increment on the same field is rejected."""
    update = Update(User)

    # First operation (increment) is allowed
    update.increment(update.fields.points, 5)

    # Decrement on the same field should be rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement(update.fields.points, 10)

    # Verify the original increment remains
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "points", {"amount": 5})


def test_decrement_with_optional_and_union_types():
    """Test decrement with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Each field can only have one decrement
    update.decrement(update.fields.union_numeric, 5)
    # Decrementing Optional[int] should work if validation allows numeric ops on it
    update.decrement(update.fields.optional_int, 10)

    # Second decrement on the same field is rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement(update.fields.union_numeric, 3.14)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(
        result, IncrementOperation, "union_numeric", {"amount": -5}
    )
    assert_operation_present(
        result, IncrementOperation, "optional_int", {"amount": -10}
    )


def test_decrement_nested_fields():
    """Test decrement operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested decrement
    update.decrement(update.fields.nested.inner.val, 5)

    # Multiple decrements on same nested field are rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement(update.fields.nested.inner.val, 3)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError via increment
    with pytest.raises(
        InvalidPathError
    ):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CORRECTED EXPECTED EXCEPTION
        update.decrement("nested.non_existent", 5)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError via increment
    with pytest.raises(
        ValueTypeError
    ):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CORRECTED EXPECTED EXCEPTION
        update.decrement(update.fields.str_list, 5)  # str_list is not numeric


def test_decrement_without_type_validation():
    """Test that decrement works without a model type (no validation)."""
    update = Update()  # No model type

    # Each field gets only one decrement
    update.decrement("counter", 5)
    update.decrement("value", 10.5)
    update.decrement("score")  # Default decrement by 1
    update.decrement("nested.counter", 3)

    # Multiple decrements on the same field are rejected
    with pytest.raises(
        ValueError, match="already has an increment/decrement operation"
    ):  # Updated match message
        update.decrement("counter", 2)

    # Build should succeed with original decrements
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4

    assert_operation_present(result, IncrementOperation, "counter", {"amount": -5})
    assert_operation_present(result, IncrementOperation, "value", {"amount": -10.5})
    assert_operation_present(result, IncrementOperation, "score", {"amount": -1})
    assert_operation_present(
        result, IncrementOperation, "nested.counter", {"amount": -3}
    )


def test_decrement_build_result():
    """Test that decrement operations build the correct agnostic operation list."""
    update = Update()
    # Use separate fields
    update.decrement("views", 1)
    update.decrement("score", 5.5)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "views", {"amount": -1})
    assert_operation_present(result, IncrementOperation, "score", {"amount": -5.5})


def test_decrement_zero():
    """Test decrementing by zero."""
    update = Update().decrement("counter", 0)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 0})


def test_decrement_negative():
    """Test that decrement with negative values (acts like increment)."""
    update = Update().decrement("counter", -5)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(
        result, IncrementOperation, "counter", {"amount": 5}
    )  # Negative decrement is positive increment


def test_decrement_reuses_increment():
    """Test that decrement uses the increment method logic."""
    update1 = Update(NumericModel)
    update2 = Update(NumericModel)

    # Use separate update objects
    update1.increment(update1.fields.int_field, -10)
    update2.decrement(update2.fields.int_field, 10)

    result1 = update1.build()
    result2 = update2.build()

    assert isinstance(result1, list)
    assert isinstance(result2, list)
    assert len(result1) == 1
    assert len(result2) == 1

    op1 = result1[0]
    op2 = result2[0]

    assert isinstance(op1, IncrementOperation)
    assert isinstance(op2, IncrementOperation)
    assert op1.field_path == "int_field"
    assert op2.field_path == "int_field"
    assert op1.amount == -10
    assert op2.amount == -10  # decrement(10) results in IncrementOperation(amount=-10)
