# tests/base/update/test_decrement.py

import pytest
from async_repository.base.update import (
    Update,
    IncrementOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.base.conftest import User, NumericModel, NestedTypes, \
    assert_operation_present
from tests.conftest import Entity


def test_decrement_valid_int_field():
    """Test decrement on a valid integer field creates a negative increment."""
    update = Update(User)
    update.decrement(update.fields.points, 10)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "points"
    assert operation.amount == -10


def test_decrement_valid_float_field():
    """Test decrement on a valid float field creates a negative increment."""
    update = Update(User)
    update.decrement(update.fields.balance, 5.5)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "balance"
    assert operation.amount == -5.5


def test_decrement_valid_union_numeric_field():
    """Test decrement on a valid union numeric field creates a negative increment."""
    update = Update(User)
    update.decrement(update.fields.score, 3)  # 'score' is Union[int, float]
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "score"
    assert operation.amount == -3


def test_decrement_non_existent_field_raises_invalid_path_error():
    """Test decrement on a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.decrement("non_existent_field", 5)
    assert len(update.build()) == 0  # Ensure operation was not added.


def test_decrement_non_numeric_field_raises_value_type_error():
    """Test decrement on a non-numeric field raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.decrement(update.fields.name, 5)  # 'name' is str.
    assert len(update.build()) == 0


def test_decrement_invalid_amount_type_str_raises_type_error():
    """Test decrement with a string amount raises TypeError."""
    update = Update(User)
    with pytest.raises(TypeError):
        update.decrement(update.fields.score, "5")
    assert len(update.build()) == 0


def test_decrement_invalid_amount_type_list_raises_type_error():
    """Test decrement with a list amount raises TypeError."""
    update = Update(User)
    with pytest.raises(TypeError):
        update.decrement(update.fields.score, [5])
    assert len(update.build()) == 0


def test_second_decrement_on_same_field_rejected():
    """Test that a second decrement operation on the same field is rejected."""
    update = Update(User)
    update.decrement(update.fields.points, 10)  # First decrement is allowed.
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"  # CORRECTED
    ):
        update.decrement(update.fields.points, 5)  # Second decrement is rejected.

    result = update.build()
    assert len(result) == 1  # Only the first operation should be present.
    assert_operation_present(result, IncrementOperation, "points", {"amount": -10})


def test_default_decrement_after_existing_decrement_on_same_field_rejected():
    """Test default decrement (by 1) after existing decrement on same field is rejected."""
    update = Update(User)
    update.decrement(update.fields.points, 10)
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"  # CORRECTED
    ):
        update.decrement(update.fields.points)  # Default decrement by 1.

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "points", {"amount": -10})


def test_decrement_on_different_fields_allowed():
    """Test that decrement operations on different fields are allowed."""
    update = Update(User)
    update.decrement(update.fields.points, 10)
    update.decrement(update.fields.balance, 5.5)

    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "points", {"amount": -10})
    assert_operation_present(result, IncrementOperation, "balance", {"amount": -5.5})


def test_decrement_after_increment_on_same_field_rejected():
    """Test that decrement after increment on the same field is rejected."""
    update = Update(User)
    update.increment(update.fields.points, 5)
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"  # CORRECTED
    ):
        update.decrement(update.fields.points, 10)

    result = update.build()
    assert len(result) == 1  # Only the original increment should remain.
    assert_operation_present(result, IncrementOperation, "points", {"amount": 5})


def test_decrement_on_union_numeric_field_in_numeric_model():
    """Test decrement on a Union[int, float] field in NumericModel."""
    update = Update(NumericModel)
    update.decrement(update.fields.union_numeric, 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "union_numeric",
                             {"amount": -5})


def test_decrement_on_optional_int_field_in_numeric_model():
    """Test decrement on an Optional[int] field in NumericModel."""
    update = Update(NumericModel)
    update.decrement(update.fields.optional_int, 10)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "optional_int",
                             {"amount": -10})


def test_second_decrement_on_same_union_numeric_field_rejected():
    """Test second decrement on the same Union[int, float] field is rejected."""
    update = Update(NumericModel)
    update.decrement(update.fields.union_numeric, 5)
    with pytest.raises(
            ValueError, match=r"Field 'union_numeric' already has an operation"
            # CORRECTED
    ):
        update.decrement(update.fields.union_numeric, 3.14)

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "union_numeric",
                             {"amount": -5})


def test_decrement_valid_nested_field():
    """Test decrement operation on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.decrement(update.fields.nested.inner.val, 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val",
                             {"amount": -5})


def test_multiple_decrements_on_same_nested_field_rejected():
    """Test multiple decrement operations on the same nested field are rejected."""
    update = Update(NestedTypes)
    update.decrement(update.fields.nested.inner.val, 5)
    with pytest.raises(
            ValueError, match=r"Field 'nested.inner.val' already has an operation"
            # CORRECTED
    ):
        update.decrement(update.fields.nested.inner.val, 3)

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val",
                             {"amount": -5})


def test_decrement_non_existent_nested_field_raises_invalid_path_error():
    """Test decrement on a non-existent nested field raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.decrement("nested.non_existent_path", 5)
    assert len(update.build()) == 0


def test_decrement_non_numeric_nested_field_raises_value_type_error():
    """Test decrement on a non-numeric nested field (e.g. List[str]) raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):  # 'str_list' is List[str], not numeric.
        update.decrement(update.fields.str_list, 5)
    assert len(update.build()) == 0


def test_decrement_simple_field_no_validation():
    """Test decrement on a simple field without type validation."""
    update = Update()  # No model type, so no path/type validation.
    update.decrement("counter", 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": -5})


def test_decrement_float_field_no_validation():
    """Test decrement with float on a field without type validation."""
    update = Update()
    update.decrement("value", 10.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "value", {"amount": -10.5})


def test_decrement_default_value_no_validation():
    """Test decrement with default value (by 1) without type validation."""
    update = Update()
    update.decrement("score")  # Default decrement by 1.
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "score", {"amount": -1})


def test_decrement_nested_field_no_validation():
    """Test decrement on a nested field string path without type validation."""
    update = Update()
    update.decrement("nested.counter", 3)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.counter",
                             {"amount": -3})


def test_multiple_decrements_on_same_field_no_validation_rejected():
    """Test multiple decrements on the same field (no model for validation) are rejected."""
    update = Update()
    update.decrement("counter", 5)
    with pytest.raises(
            ValueError, match=r"Field 'counter' already has an operation"  # CORRECTED
    ):
        update.decrement("counter", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": -5})


def test_decrement_build_result_multiple_fields_no_validation():
    """Test that decrement operations on separate fields build correctly without validation."""
    update = Update()
    update.decrement("views", 1)
    update.decrement("score", 5.5)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "views", {"amount": -1})
    assert_operation_present(result, IncrementOperation, "score", {"amount": -5.5})


def test_decrement_by_zero():
    """Test decrementing by zero results in an increment operation of amount zero."""
    update = Update().decrement("counter", 0)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 0})


def test_decrement_by_negative_value_acts_as_increment():
    """Test that decrement with a negative value effectively becomes a positive increment."""
    # Decrementing by -5 means adding 5.
    update = Update().decrement("counter", -5)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})


def test_decrement_internally_uses_increment_operation():
    """
    Test that decrement(field, X) is equivalent to increment(field, -X)
    in terms of the resulting operation.
    """
    update_decrement = Update(NumericModel)
    update_decrement.decrement(update_decrement.fields.int_field, 10)
    result_decrement = update_decrement.build()

    op_decrement = result_decrement[0]
    assert isinstance(op_decrement, IncrementOperation)
    assert op_decrement.field_path == "int_field"
    assert op_decrement.amount == -10


def test_update_dsl_for_nested_field_decrement():
    """Test decrement operation on nested numeric fields using string path (DSL-like)."""
    update = Update(Entity).decrement("metadata.stats.clicks", 10)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "metadata.stats.clicks"
    assert operation.amount == -10