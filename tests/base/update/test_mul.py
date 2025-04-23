# tests/base/update/test_mul.py

import pytest
from async_repository.base.update import (
    Update,
    MultiplyOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import User, NumericModel, NestedTypes

from tests.base.conftest import assert_operation_present


def test_mul_basic():
    """Test basic mul operation functionality builds the correct operation."""
    update = Update().mul("price", 1.1)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "price", {"factor": 1.1})


def test_mul_with_type_validation():
    """Test that mul operations are type validated."""
    update = Update(NumericModel)

    # Valid mul operations
    update.mul("int_field", 2)
    update.mul("float_field", 0.5)
    update.mul("union_numeric", 1.5)  # Should pass with updated logic

    # Invalid field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
        update.mul("non_existent", 2)

    # Invalid field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):  # Corrected expected exception
        Update(User).mul("name", 2)  # name is not numeric

    # Invalid factor type (for the operation itself) - Method raises TypeError
    with pytest.raises(
        TypeError, match="Multiply factor must be numeric"
    ):  # <<<<<< CHANGED TO TypeError
        update.mul("int_field", "2")  # Factor "2" is str, caught by method check


def test_mul_with_nested_fields():
    """Test mul operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested mul
    update.mul("nested.inner.val", 2)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
        update.mul("nested.non_existent", 2)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError):  # Corrected expected exception
        update.mul("str_list", 2)  # str_list is not numeric


def test_mul_without_type_validation():
    """Test that mul works without a model type (no validation)."""
    update = Update()

    update.mul("price", 1.1)
    update.mul("quantity", 2)
    update.mul("nested.field", 0.75)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, MultiplyOperation, "price", {"factor": 1.1})
    assert_operation_present(result, MultiplyOperation, "quantity", {"factor": 2})
    assert_operation_present(
        result, MultiplyOperation, "nested.field", {"factor": 0.75}
    )


def test_mul_edge_cases():
    """Test multiplication with special values builds correct operations."""
    update = Update()

    # Multiply by 0
    update.mul("field1", 0)

    # Multiply by negative
    update.mul("field2", -1)

    # Multiply by very small number
    update.mul("field3", 0.001)

    # Multiply by very large number
    update.mul("field4", 1000)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4
    assert_operation_present(result, MultiplyOperation, "field1", {"factor": 0})
    assert_operation_present(result, MultiplyOperation, "field2", {"factor": -1})
    assert_operation_present(result, MultiplyOperation, "field3", {"factor": 0.001})
    assert_operation_present(result, MultiplyOperation, "field4", {"factor": 1000})
