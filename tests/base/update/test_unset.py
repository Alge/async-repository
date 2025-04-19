# tests/base/update/test_unset.py

import pytest
from typing import List, Type, Optional, TypeVar # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,        # Import base operation class
    UnsetOperation,         # Import specific operation class
    InvalidPathError,       # Import specific exception
    # ValueTypeError is not typically raised by unset
)
from .conftest import User, NestedTypes


# --- Test Helper (Copied from previous response for completeness) ---
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
                 # Special handling for comparing potentially serialized dicts/lists
                 if isinstance(expected_value, (dict, list)) and isinstance(actual_value, (dict, list)):
                     assert actual_value == expected_value, \
                        f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                 # Handle list comparison specifically for PushOperation 'items' attribute
                 elif attr == 'items' and isinstance(expected_value, list) and isinstance(actual_value, list):
                     assert actual_value == expected_value, \
                        f"Attribute 'items' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                 else:
                     assert actual_value == expected_value, \
                        f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
# --- End Test Helper ---


def test_unset_with_type_validation():
    """Test that unset operations are validated for field existence."""
    update = Update(User)

    # Valid unset operations
    update.unset("name")
    update.unset("email")
    update.unset("active")

    # Non-existent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.unset("non_existent")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, UnsetOperation, "name")
    assert_operation_present(result, UnsetOperation, "email")
    assert_operation_present(result, UnsetOperation, "active")


def test_unset_with_nested_fields():
    """Test unset operations with nested fields."""
    update = Update(User)

    # Valid nested unset
    update.unset("metadata.key1")

    # Non-existent nested field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.unset("metadata.non_existent")

    # Non-existent parent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.unset("non_existent.field")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "metadata.key1")


def test_unset_with_complex_types():
    """Test unset operations with complex nested structures."""
    update = Update(NestedTypes)

    # Valid unset operations
    update.unset("counter")
    update.unset("nested.inner.val")

    # Non-existent nested field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError): # Corrected expected exception
        update.unset("nested.inner.non_existent")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, UnsetOperation, "counter")
    assert_operation_present(result, UnsetOperation, "nested.inner.val")


def test_unset_without_model_type():
    """Test that unset works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.unset("any_field")
    update.unset("nested.field")

    # Build should succeed
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, UnsetOperation, "any_field")
    assert_operation_present(result, UnsetOperation, "nested.field")


def test_unset_build_result():
    """Test that unset operations build the correct agnostic operation list."""
    update = Update().unset("name").unset("metadata.note")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, UnsetOperation, "name")
    # Even if metadata.note doesn't exist in a model, unset is allowed without model
    assert_operation_present(result, UnsetOperation, "metadata.note")