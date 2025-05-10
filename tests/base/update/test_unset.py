# tests/base/update/test_unset.py

import pytest
from async_repository.base.update import (
    Update,
    UnsetOperation,
    InvalidPathError,
)
from tests.base.conftest import User, NestedTypes, assert_operation_present
from tests.conftest import Entity


# --- Tests for `unset` with type validation (User model) ---

def test_unset_valid_top_level_field_with_type_validation():
    """Test unset on a valid top-level field is validated for existence."""
    update = Update(User)
    update.unset("name")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "name")

def test_unset_another_valid_top_level_field_with_type_validation():
    """Test unset on another valid top-level field."""
    update = Update(User)
    update.unset("email") # email is Optional[str]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "email")

def test_unset_non_existent_field_raises_invalid_path_error():
    """Test unset on a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.unset("non_existent_field")
    assert len(update.build()) == 0


# --- Tests for `unset` with nested fields (User model) ---

def test_unset_valid_nested_field():
    """Test unset on a valid nested field."""
    update = Update(User)
    update.unset("metadata.key1") # metadata.key1 is str
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "metadata.key1")

def test_unset_non_existent_nested_sub_field_raises_invalid_path_error():
    """Test unset on a non-existent nested sub-field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError): # metadata.non_existent_key does not exist
        update.unset("metadata.non_existent_key")
    assert len(update.build()) == 0

def test_unset_non_existent_nested_parent_field_raises_invalid_path_error():
    """Test unset where parent in path does not exist raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError): # non_existent_parent does not exist
        update.unset("non_existent_parent.some_field")
    assert len(update.build()) == 0


# --- Tests for `unset` with complex types (NestedTypes model) ---

def test_unset_simple_field_in_complex_type_model():
    """Test unset on a simple field within a complex model."""
    update = Update(NestedTypes)
    update.unset("counter") # counter is int
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "counter")

def test_unset_deeply_nested_field_in_complex_type_model():
    """Test unset on a deeply nested field within a complex model."""
    update = Update(NestedTypes)
    update.unset("nested.inner.val") # nested.inner.val is int
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "nested.inner.val")

def test_unset_non_existent_deeply_nested_field_in_complex_model_raises_error():
    """Test unset on a non-existent deeply nested field in complex model."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError): # nested.inner.non_existent_sub_field does not exist
        update.unset("nested.inner.non_existent_sub_field")
    assert len(update.build()) == 0


# --- Tests for `unset` without a model type (no validation for path) ---

def test_unset_any_top_level_field_no_model():
    """Test unset on any top-level field string without model type."""
    update = Update()
    update.unset("any_field_name")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "any_field_name")

def test_unset_any_nested_field_path_no_model():
    """Test unset on any nested field string path without model type."""
    update = Update()
    update.unset("some_nested.field_path")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "some_nested.field_path")


# --- Test `unset` build result for multiple distinct fields ---

def test_unset_build_result_multiple_distinct_fields():
    """Test that unset operations on distinct fields build correctly."""
    update = Update().unset("user_name").unset("user_profile.settings.theme")
    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, UnsetOperation, "user_name")
    assert_operation_present(result, UnsetOperation, "user_profile.settings.theme")


# --- Test `unset` with DSL-like path (Entity model) ---

def test_update_dsl_nested_unset():
    """Test unset operation on nested fields using string path with Entity model."""
    update = Update(Entity).unset("metadata.settings.notifications_enabled")
    result = update.build() # Get operations list from build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, UnsetOperation)
    assert operation.field_path == "metadata.settings.notifications_enabled"