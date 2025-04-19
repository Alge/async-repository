# tests/base/update/test_mul.py

import pytest
from typing import List, Type, Optional, TypeVar # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,        # Import base operation class
    MultiplyOperation,      # Import specific operation class
    InvalidPathError,       # Import specific exception
    ValueTypeError,         # Import specific exception
)
from .conftest import User, NumericModel, NestedTypes


# --- Test Helper (can be defined here or imported from a common place) ---
OpT = TypeVar('OpT', bound=UpdateOperation)

def find_operation(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str
) -> Optional[OpT]:
    """Finds the first operation of a specific type and field path."""
    for op in operations:
        if isinstance(op, op_type) and op.field_path == field_path:
            return op
    return None

def assert_operation_present(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str,
    expected_attrs: Optional[dict] = None # Check specific attributes like value, amount
):
    """Asserts that a specific operation exists and optionally checks its attributes."""
    op = find_operation(operations, op_type, field_path)
    assert op is not None, f"{op_type.__name__} for field '{field_path}' not found in {operations}"
    if expected_attrs:
        for attr, expected_value in expected_attrs.items():
            assert hasattr(op, attr), f"Operation {op!r} missing attribute '{attr}'"
            actual_value = getattr(op, attr)
            # Use pytest.approx for floats if needed
            if isinstance(expected_value, float):
                 import pytest
                 assert actual_value == pytest.approx(expected_value), \
                     f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
            else:
                 assert actual_value == expected_value, \
                    f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
# --- End Test Helper ---


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
    update.mul("union_numeric", 1.5) # Should pass with updated logic

    # Invalid field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.mul("non_existent", 2)

    # Invalid field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError): # Corrected expected exception
        Update(User).mul("name", 2) # name is not numeric

    # Invalid factor type (for the operation itself) - Method raises TypeError
    with pytest.raises(TypeError, match="Multiply factor must be numeric"): # <<<<<< CHANGED TO TypeError
        update.mul("int_field", "2") # Factor "2" is str, caught by method check


def test_mul_with_nested_fields():
    """Test mul operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested mul
    update.mul("nested.inner.val", 2)

    # Invalid nested field (non-existent) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.mul("nested.non_existent", 2)

    # Invalid nested field (non-numeric) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError): # Corrected expected exception
        update.mul("str_list", 2) # str_list is not numeric


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
    assert_operation_present(result, MultiplyOperation, "nested.field", {"factor": 0.75})


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