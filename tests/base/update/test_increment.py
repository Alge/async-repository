# tests/base/update/test_increment.py

import pytest
from async_repository.base.update import (
    Update,
    IncrementOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.conftest import Entity
from tests.base.conftest import User, NumericModel, NestedTypes, \
    assert_operation_present


def test_increment_valid_int_field():
    """Test increment on a valid integer field is added."""
    update = Update(User)
    update.increment(update.fields.points, 10)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "points"
    assert operation.amount == 10


def test_increment_valid_float_field():
    """Test increment on a valid float field is added."""
    update = Update(User)
    update.increment(update.fields.balance, 5.5)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "balance"
    assert operation.amount == 5.5


def test_increment_valid_union_numeric_field():
    """Test increment on a valid union numeric field is added."""
    update = Update(User)
    update.increment(update.fields.score, 3)  # 'score' is Union[int, float]
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "score"
    assert operation.amount == 3


def test_increment_non_existent_field_raises_invalid_path_error():
    """Test increment on a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.increment("non_existent_field", 5)
    assert len(update.build()) == 0  # Ensure operation was not added.


def test_increment_non_numeric_field_raises_value_type_error():
    """Test increment on a non-numeric field raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.increment(update.fields.name, 5)  # 'name' is str.
    assert len(update.build()) == 0


def test_increment_invalid_amount_type_str_raises_type_error():
    """Test increment with a string amount raises TypeError."""
    update = Update(User)
    with pytest.raises(TypeError):
        update.increment(update.fields.score, "5")
    assert len(update.build()) == 0


def test_increment_invalid_amount_type_list_raises_type_error():
    """Test increment with a list amount raises TypeError."""
    update = Update(User)
    with pytest.raises(TypeError):
        update.increment(update.fields.score, [5])
    assert len(update.build()) == 0


def test_second_increment_on_same_field_rejected():
    """Test that a second increment operation on the same field is rejected."""
    update = Update(User)
    update.increment(update.fields.points, 10)  # First increment is allowed.
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"
    ):
        update.increment(update.fields.points, 5)  # Second increment is rejected.

    result = update.build()
    assert len(result) == 1  # Only the first operation should be present.
    assert_operation_present(result, IncrementOperation, "points", {"amount": 10})


def test_default_increment_after_existing_increment_on_same_field_rejected():
    """Test default increment (by 1) after existing increment on same field is rejected."""
    update = Update(User)
    update.increment(update.fields.points, 10)
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"
    ):
        update.increment(update.fields.points)  # Default increment by 1.

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "points", {"amount": 10})


def test_increment_on_different_fields_allowed():
    """Test that increment operations on different fields are allowed."""
    update = Update(User)
    update.increment(update.fields.points, 10)
    update.increment(update.fields.balance, 5.5)

    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "points", {"amount": 10})
    assert_operation_present(result, IncrementOperation, "balance", {"amount": 5.5})


def test_increment_after_decrement_on_same_field_rejected():
    """Test that increment after decrement on the same field is rejected."""
    update = Update(User)
    update.decrement(update.fields.points, 5)  # Stored as IncrementOperation(amount=-5)
    with pytest.raises(
            ValueError, match=r"Field 'points' already has an operation"
    ):
        update.increment(update.fields.points, 10)

    result = update.build()
    assert len(result) == 1  # Only the original decrement should remain.
    assert_operation_present(result, IncrementOperation, "points", {"amount": -5})


def test_increment_on_union_numeric_field_in_numeric_model():
    """Test increment on a Union[int, float] field in NumericModel."""
    update = Update(NumericModel)
    update.increment(update.fields.union_numeric, 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "union_numeric", {"amount": 5})


def test_increment_on_optional_int_field_in_numeric_model():
    """Test increment on an Optional[int] field in NumericModel."""
    update = Update(NumericModel)
    update.increment(update.fields.optional_int, 10)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "optional_int", {"amount": 10})


def test_second_increment_on_same_union_numeric_field_rejected():
    """Test second increment on the same Union[int, float] field is rejected."""
    update = Update(NumericModel)
    update.increment(update.fields.union_numeric, 5)
    with pytest.raises(
            ValueError, match=r"Field 'union_numeric' already has an operation"
    ):
        update.increment(update.fields.union_numeric, 3.14)

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "union_numeric", {"amount": 5})


def test_increment_valid_nested_field():
    """Test increment operation on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.increment(update.fields.nested.inner.val, 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val",
                             {"amount": 5})


def test_multiple_increments_on_same_nested_field_rejected():
    """Test multiple increment operations on the same nested field are rejected."""
    update = Update(NestedTypes)
    update.increment(update.fields.nested.inner.val, 5)
    with pytest.raises(
            ValueError, match=r"Field 'nested.inner.val' already has an operation"
    ):
        update.increment(update.fields.nested.inner.val, 3)

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val",
                             {"amount": 5})


def test_increment_non_existent_nested_field_raises_invalid_path_error():
    """Test increment on a non-existent nested field raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.increment("nested.non_existent_path", 5)
    assert len(update.build()) == 0


def test_increment_non_numeric_nested_field_raises_value_type_error():
    """Test increment on a non-numeric nested field (e.g. List[str]) raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):  # 'str_list' is List[str], not numeric.
        update.increment(update.fields.str_list, 5)
    assert len(update.build()) == 0


def test_increment_simple_field_no_validation():
    """Test increment on a simple field without type validation."""
    update = Update()  # No model type, so no path/type validation.
    update.increment("counter", 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})


def test_increment_float_field_no_validation():
    """Test increment with float on a field without type validation."""
    update = Update()
    update.increment("value", 10.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "value", {"amount": 10.5})


def test_increment_default_value_no_validation():
    """Test increment with default value (by 1) without type validation."""
    update = Update()
    update.increment("score")  # Default increment by 1.
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "score", {"amount": 1})


def test_increment_nested_field_no_validation():
    """Test increment on a nested field string path without type validation."""
    update = Update()
    update.increment("nested.counter", 3)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.counter",
                             {"amount": 3})


def test_multiple_increments_on_same_field_no_validation_rejected():
    """Test multiple increments on the same field (no model for validation) are rejected."""
    update = Update()
    update.increment("counter", 5)
    with pytest.raises(
            ValueError, match=r"Field 'counter' already has an operation"
    ):
        update.increment("counter", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})


def test_increment_build_result_multiple_fields_no_validation():
    """Test that increment operations on separate fields build correctly without validation."""
    update = Update()
    update.increment("views", 1)
    update.increment("score", 5.5)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "views", {"amount": 1})
    assert_operation_present(result, IncrementOperation, "score", {"amount": 5.5})


def test_increment_by_zero():
    """Test incrementing by zero results in an increment operation of amount zero."""
    update = Update().increment("counter", 0)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 0})


def test_increment_by_negative_value_acts_as_decrement():
    """Test that increment with a negative value effectively becomes a negative increment."""
    update = Update().increment("counter", -5)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": -5})


def test_increment_float_precision_on_separate_fields():
    """Test that increment handles float precision correctly for separate fields."""
    update = Update().increment("value1", 0.1).increment("value2", 0.2)
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    # Check value1
    op1 = next((op for op in result if op.field_path == "value1"), None)
    assert op1 is not None and op1.amount == 0.1
    # Check value2
    op2 = next((op for op in result if op.field_path == "value2"), None)
    assert op2 is not None and op2.amount == 0.2


def test_update_dsl_for_nested_field_increment():
    """Test increment operation on nested numeric fields using string path (DSL-like)."""
    update = Update(Entity).increment("metadata.stats.visits", 5)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, IncrementOperation)
    assert operation.field_path == "metadata.stats.visits"
    assert operation.amount == 5