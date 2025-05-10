# tests/base/update/test_min.py

import pytest
from async_repository.base.update import (
    Update,
    MinOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.conftest import Entity
from tests.base.conftest import User, NumericModel, NestedTypes

from tests.base.conftest import assert_operation_present


def test_min_basic():
    """Test basic min operation functionality builds the correct operation."""
    update = Update().min("score", 50)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "score", {"value": 50})


def test_min_with_type_validation():
    """Test that min operations are type validated."""
    update = Update(NumericModel)

    # Valid min operations
    update.min("int_field", 10)
    update.min("float_field", 5.5)
    update.min("union_numeric", 3)  # Should pass with updated logic

    # Invalid field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
        update.min("non_existent", 5)

    # Invalid field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):  # Corrected expected exception
        Update(User).min("name", 5)  # name is not numeric

    # Invalid value type (for the field) - Validator raises ValueTypeError
    # This case is preempted by the method's own type check
    with pytest.raises(
        TypeError, match="Min value must be numeric"
    ):  # <<<<<< CHANGED TO TypeError
        update.min("int_field", "5")  # Value "5" is str, caught by method check

    # Invalid value type (for the operation itself) - Method raises TypeError
    with pytest.raises(
        TypeError, match="Min value must be numeric"
    ):  # This one was correct
        update.min("int_field", "not a number")


def test_min_with_nested_fields():
    """Test min operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested min
    update.min("nested.inner.val", 5)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
        update.min("nested.non_existent", 5)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):  # Corrected expected exception
        update.min("str_list", 5)  # str_list is not numeric


def test_min_without_type_validation():
    """Test that min works without a model type (no validation)."""
    update = Update()

    update.min("score", 50)
    update.min("value", -10.5)
    update.min("nested.field", 0)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, MinOperation, "score", {"value": 50})
    assert_operation_present(result, MinOperation, "value", {"value": -10.5})
    assert_operation_present(result, MinOperation, "nested.field", {"value": 0})


def test_min_edge_cases():
    """Test min with edge case values builds correct operations."""
    update = Update()

    # Negative values
    update.min("field1", -100)

    # Zero
    update.min("field2", 0)

    # Very small number
    update.min("field3", 0.0001)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, MinOperation, "field1", {"value": -100})
    assert_operation_present(result, MinOperation, "field2", {"value": 0})
    assert_operation_present(result, MinOperation, "field3", {"value": 0.0001})


def test_update_dsl_nested_min():
    """Test min operation on nested numeric fields."""
    # Create an update with min operation on nested path
    update = Update(Entity).min("metadata.limits.floor", 30)

    # Assert the operation was added correctly
    assert len(update._operations) == 1
    assert isinstance(update._operations[0], MinOperation)
    assert update._operations[0].field_path == "metadata.limits.floor"
    assert update._operations[0].value == 30