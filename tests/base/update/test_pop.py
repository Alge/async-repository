# tests/base/update/test_pop.py

import pytest
from async_repository.base.update import (
    Update,
    PopOperation,
    InvalidPathError,
)
from tests.base.conftest import User, NestedTypes, Organization, assert_operation_present
from tests.conftest import Entity


# --- Tests for `pop` with type validation (User model) ---

def test_pop_default_direction_with_type_validation():
    """Test pop with default direction (1 - last element) is validated."""
    update = Update(User)
    update.pop("tags")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "tags", {"position": 1})

def test_pop_explicit_last_direction_with_type_validation():
    """Test pop with explicit last element direction (1) is validated."""
    update = Update(User)
    update.pop("tags", 1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "tags", {"position": 1})

def test_pop_explicit_first_direction_with_type_validation():
    """Test pop with explicit first element direction (-1) is validated."""
    update = Update(User)
    update.pop("tags", -1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "tags", {"position": -1})

def test_pop_invalid_field_type_non_list_raises_invalid_path_error():
    """Test pop on a non-list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pop("name") # name is str, not a list
    assert len(update.build()) == 0

def test_pop_non_existent_field_raises_invalid_path_error():
    """Test pop on a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pop("non_existent_field")
    assert len(update.build()) == 0

@pytest.mark.parametrize("invalid_direction", [2, 0, -2, "1"])
def test_pop_invalid_direction_raises_value_error(invalid_direction):
    """Test pop with an invalid direction raises ValueError."""
    update = Update(User)
    with pytest.raises(ValueError): # Direction must be 1 or -1
        update.pop("tags", invalid_direction)
    assert len(update.build()) == 0


# --- Tests for `pop` with nested fields (User and NestedTypes models) ---

def test_pop_from_nested_list_of_objects():
    """Test pop from a nested list field containing objects."""
    update = Update(User)
    update.pop("addresses") # addresses is List[Address]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "addresses", {"position": 1})

def test_pop_from_nested_non_list_field_raises_invalid_path_error():
    """Test pop from a nested non-list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pop("metadata.key1") # key1 is str, not a list
    assert len(update.build()) == 0

def test_pop_from_non_existent_nested_field_raises_invalid_path_error():
    """Test pop from a non-existent nested field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pop("metadata.non_existent_key")
    assert len(update.build()) == 0


# --- Tests for `pop` with complex list types (NestedTypes model) ---

def test_pop_from_list_of_int():
    """Test pop from a List[int]."""
    update = Update(NestedTypes)
    update.pop("simple_list")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "simple_list", {"position": 1})

def test_pop_from_list_of_str():
    """Test pop from a List[str]."""
    update = Update(NestedTypes)
    update.pop("str_list")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "str_list", {"position": 1})

def test_pop_from_list_of_complex_item():
    """Test pop from a List[ComplexItem]."""
    update = Update(NestedTypes)
    update.pop("complex_list")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "complex_list", {"position": 1})

def test_pop_from_non_list_field_in_complex_model_raises_invalid_path_error():
    """Test pop from non-list field (int) in NestedTypes raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.pop("counter") # counter is an int
    assert len(update.build()) == 0

def test_pop_from_dict_field_in_complex_model_raises_invalid_path_error():
    """Test pop from dict field in NestedTypes raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.pop("dict_field") # dict_field is a dict
    assert len(update.build()) == 0


# --- Tests for `pop` without a model type (no validation) ---

def test_pop_default_direction_no_model():
    """Test pop with default direction without model type validation."""
    update = Update()
    update.pop("tags")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "tags", {"position": 1})

def test_pop_explicit_last_direction_no_model():
    """Test pop with explicit last direction without model type validation."""
    update = Update()
    update.pop("numbers", 1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "numbers", {"position": 1})

def test_pop_explicit_first_direction_no_model():
    """Test pop with explicit first direction on nested path without model type."""
    update = Update()
    update.pop("nested.list", -1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "nested.list", {"position": -1})


# --- Test `pop` build result ---

def test_pop_build_result_with_distinct_fields():
    """Test that pop operations on distinct fields build correctly."""
    update = Update().pop("tags").pop("items", -1) # Different fields
    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, PopOperation, "tags", {"position": 1})
    assert_operation_present(result, PopOperation, "items", {"position": -1})


# --- Tests for `pop` from deeply nested lists (Organization model) ---

def test_pop_from_deeply_nested_list_default_pos():
    """Test pop (default pos) from a list within a nested object array."""
    update = Update(Organization)
    update.pop("departments.0.members")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.members", {"position": 1})

def test_pop_from_deeply_nested_list_first_pos():
    """Test pop (first pos) from a list within a nested object array."""
    update = Update(Organization)
    update.pop("departments.0.members", -1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.members", {"position": -1})


def test_pop_from_very_deeply_nested_list_explicit_last_pos():
    """Test pop (explicit last pos) from a very deeply nested list."""
    update = Update(Organization)
    update.pop("departments.0.categories.0.items", 1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.categories.0.items", {"position": 1})

def test_pop_from_very_deeply_nested_list_default_pos():
    """Test pop (default pos) from another very deeply nested list."""
    update = Update(Organization)
    update.pop("departments.0.categories.0.tags")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.categories.0.tags", {"position": 1})

def test_pop_from_very_deeply_nested_list_first_pos():
    """Test pop (first pos) from yet another very deeply nested list."""
    update = Update(Organization)
    update.pop("departments.0.categories.0.counts", -1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.categories.0.counts", {"position": -1})


def test_pop_from_deeply_nested_non_list_field_raises_invalid_path_error():
    """Test pop from a deeply nested non-list field raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pop("departments.0.name") # name is a string
    assert len(update.build()) == 0

def test_pop_from_non_existent_deeply_nested_path_raises_invalid_path_error():
    """Test pop from a non-existent deeply nested path raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pop("departments.0.categories.0.non_existent_list")
    assert len(update.build()) == 0

@pytest.mark.parametrize("invalid_direction", [2, 0, "1"])
def test_pop_from_deeply_nested_list_invalid_direction_raises_value_error(invalid_direction):
    """Test pop from a deeply nested list with invalid direction raises ValueError."""
    update = Update(Organization)
    with pytest.raises(ValueError):
        update.pop("departments.0.categories.0.items", invalid_direction)
    assert len(update.build()) == 0


# --- Test `pop` with DSL-like path ---

def test_update_dsl_nested_pop():
    """Test pop operation on nested array fields using string path."""
    update = Update(Entity).pop("metadata.collections.items", 1)  # Pop from end
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PopOperation)
    assert operation.field_path == "metadata.collections.items"
    assert operation.position == 1