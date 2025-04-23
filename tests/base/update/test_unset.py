# tests/base/update/test_unset.py

import pytest
from typing import List, Type, Optional, TypeVar  # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,  # Import base operation class
    UnsetOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    # ValueTypeError is not typically raised by unset
)
from tests.base.conftest import User, NestedTypes

from tests.base.conftest import assert_operation_present

def test_unset_with_type_validation():
    """Test that unset operations are validated for field existence."""
    update = Update(User)

    # Valid unset operations
    update.unset("name")
    update.unset("email")
    update.unset("active")

    # Non-existent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
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
    with pytest.raises(InvalidPathError):  # Corrected expected exception
        update.unset("metadata.non_existent")

    # Non-existent parent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):  # Corrected expected exception
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
    with pytest.raises(InvalidPathError):  # Corrected expected exception
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
