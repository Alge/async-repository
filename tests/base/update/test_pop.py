# tests/base/update/test_pop.py

import pytest
from typing import List, Type, Optional, TypeVar  # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,  # Import base operation class
    PopOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    # ValueTypeError is not typically raised by pop itself, but InvalidPathError is
)
from tests.base.conftest import User, NestedTypes, Organization, find_operations

from tests.base.conftest import assert_operation_present

def test_pop_with_type_validation():
    """Test that pop operations are type validated."""
    update = Update(User)

    # Add some valid operations first to check later
    update.pop("tags")  # Default direction (1 - last element)
    update.pop("tags", 1)  # Last element explicitly
    update.pop("tags", -1)  # First element

    # Invalid field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("name")  # name is str, not a list

    # Non-existent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("non_existent")

    # Invalid direction - Method raises ValueError
    with pytest.raises(ValueError):
        update.pop("tags", 2)  # direction must be 1 or -1

    with pytest.raises(ValueError):
        update.pop("tags", 0)

    with pytest.raises(ValueError):
        update.pop("tags", -2)

    # Check the valid operations were added
    result = update.build()
    assert isinstance(result, list)
    pop_ops = find_operations(result, PopOperation, "tags")
    assert len(pop_ops) == 3
    assert pop_ops[0].position == 1  # Default
    assert pop_ops[1].position == 1  # Explicit last
    assert pop_ops[2].position == -1  # Explicit first


def test_pop_with_nested_fields():
    """Test pop operations with nested list fields."""
    update = Update(User)

    # Valid nested pop
    update.pop("addresses")  # Pop from a list field (addresses) which contains objects

    # Invalid nested field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("metadata.key1")  # key1 is str, not a list

    # Non-existent nested field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("metadata.non_existent")

    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "addresses", {"position": 1})


def test_pop_with_complex_types():
    """Test pop operations with complex list types."""
    update = Update(NestedTypes)

    # Valid pop from different list types
    update.pop("simple_list")  # List[int]
    update.pop("str_list")  # List[str]
    update.pop("complex_list")  # List[ComplexItem]

    # Invalid pop from non-list field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("counter")  # counter is an int

    with pytest.raises(InvalidPathError):
        update.pop("dict_field")  # dict_field is a dict

    result = update.build()
    assert len(result) == 3
    assert_operation_present(result, PopOperation, "simple_list", {"position": 1})
    assert_operation_present(result, PopOperation, "str_list", {"position": 1})
    assert_operation_present(result, PopOperation, "complex_list", {"position": 1})


def test_pop_without_model_type():
    """Test that pop works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.pop("tags")  # pos=1
    update.pop("numbers", 1)  # pos=1
    update.pop("nested.list", -1)  # pos=-1

    # Build should succeed
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3
    assert_operation_present(result, PopOperation, "tags", {"position": 1})
    assert_operation_present(result, PopOperation, "numbers", {"position": 1})
    assert_operation_present(result, PopOperation, "nested.list", {"position": -1})


def test_pop_build_result():
    """Test that pop operations build the correct agnostic operation list."""
    update = Update().pop("tags").pop("items", -1)  # pos=1, pos=-1

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, PopOperation, "tags", {"position": 1})
    assert_operation_present(result, PopOperation, "items", {"position": -1})


def test_pop_from_nested_list():
    """Test popping from lists within nested objects."""

    update = Update(Organization)

    # Pop from a list within a nested object
    update.pop("departments.0.members")  # pos=1
    update.pop("departments.0.members", -1)  # pos=-1

    # Pop from a deeply nested list
    update.pop("departments.0.categories.0.items", 1)  # pos=1
    update.pop("departments.0.categories.0.tags")  # pos=1
    update.pop("departments.0.categories.0.counts", -1)  # pos=-1

    # Invalid field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("departments.0.name")  # name is a string

    # Invalid nested path - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pop("departments.0.categories.0.non_existent")

    # Invalid direction - Method raises ValueError
    with pytest.raises(ValueError):
        update.pop("departments.0.categories.0.items", 2)

    # Build and check the result
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 5  # 2 for members, 3 for categories

    # Check specific operations - order matters here
    assert (
        isinstance(result[0], PopOperation)
        and result[0].field_path == "departments.0.members"
        and result[0].position == 1
    )
    assert (
        isinstance(result[1], PopOperation)
        and result[1].field_path == "departments.0.members"
        and result[1].position == -1
    )
    assert (
        isinstance(result[2], PopOperation)
        and result[2].field_path == "departments.0.categories.0.items"
        and result[2].position == 1
    )
    assert (
        isinstance(result[3], PopOperation)
        and result[3].field_path == "departments.0.categories.0.tags"
        and result[3].position == 1
    )
    assert (
        isinstance(result[4], PopOperation)
        and result[4].field_path == "departments.0.categories.0.counts"
        and result[4].position == -1
    )

    # Use helper to find the *last* operation if needed, or all operations
    members_ops = find_operations(result, PopOperation, "departments.0.members")
    assert len(members_ops) == 2
    assert members_ops[0].position == 1
    assert members_ops[1].position == -1
    assert_operation_present(
        result, PopOperation, "departments.0.categories.0.items", {"position": 1}
    )
    assert_operation_present(
        result, PopOperation, "departments.0.categories.0.tags", {"position": 1}
    )
    assert_operation_present(
        result, PopOperation, "departments.0.categories.0.counts", {"position": -1}
    )
