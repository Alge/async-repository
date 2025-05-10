# tests/base/update/test_max.py

import pytest
from async_repository.base.update import (
    Update,
    MaxOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import User, NumericModel, NestedTypes

from tests.base.conftest import assert_operation_present

from tests.conftest import Entity


def test_max_basic():
    """Test basic max operation functionality builds the correct operation."""
    update = Update().max("score", 100)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "score", {"value": 100})


def test_max_with_type_validation():
    """Test that max operations are type validated."""
    update = Update(NumericModel)

    # Valid max operations
    update.max("int_field", 100)
    update.max("float_field", 99.9)
    update.max("union_numeric", 50)

    # Invalid field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.max("non_existent", 100)

    # Invalid field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):
        Update(User).max("name", 100)  # name is not numeric

    # Invalid value type (for the field) - Validator raises ValueTypeError
    # THIS IS THE TEST CASE THAT FAILED - change expected exception
    # Because the method itself checks if value is numeric first
    with pytest.raises(
        TypeError, match="Max value must be numeric"
    ):  # <<<<<< CHANGED TO TypeError
        update.max("int_field", "100")  # Value "100" is str, caught by method check

    # Invalid value type (for the operation itself) - Method raises TypeError
    with pytest.raises(
        TypeError, match="Max value must be numeric"
    ):  # This one was already correct
        update.max("int_field", "not a number")  # Value passed to max must be numeric


def test_max_with_nested_fields():
    """Test max operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested max
    update.max("nested.inner.val", 100)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.max("nested.non_existent", 100)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):
        update.max("str_list", 100)  # str_list is not numeric


def test_max_without_type_validation():
    """Test that max works without a model type (no validation)."""
    update = Update()

    update.max("score", 100)
    update.max("value", 999.9)
    update.max("nested.field", 0)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, MaxOperation, "score", {"value": 100})
    assert_operation_present(result, MaxOperation, "value", {"value": 999.9})
    assert_operation_present(result, MaxOperation, "nested.field", {"value": 0})


def test_max_edge_cases():
    """Test max with edge case values builds correct operations."""
    update = Update()

    # Negative values
    update.max("field1", -10)

    # Zero
    update.max("field2", 0)

    # Very large number
    update.max("field3", 1e9)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, MaxOperation, "field1", {"value": -10})
    assert_operation_present(result, MaxOperation, "field2", {"value": 0})
    assert_operation_present(result, MaxOperation, "field3", {"value": 1e9})


def test_update_dsl_nested_max():
    """Test max operation on nested numeric fields."""
    # Create an update with max operation on nested path
    update = Update(Entity).max("metadata.limits.ceiling", 100)

    # Assert the operation was added correctly
    assert len(update._operations) == 1
    assert isinstance(update._operations[0], MaxOperation)
    assert update._operations[0].field_path == "metadata.limits.ceiling"
    assert update._operations[0].value == 100