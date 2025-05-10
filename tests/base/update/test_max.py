# tests/base/update/test_max.py

import pytest
from async_repository.base.update import (
    Update,
    MaxOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.base.conftest import User, NumericModel, NestedTypes, assert_operation_present
from tests.conftest import Entity


def test_max_basic_operation():
    """Test a basic max operation builds correctly without type validation."""
    update = Update().max("score", 100)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MaxOperation)
    assert operation.field_path == "score"
    assert operation.value == 100


# --- Tests for `max` with type validation (`NumericModel`) ---

def test_max_valid_int_field_with_type_validation():
    """Test max on a valid integer field with type validation."""
    update = Update(NumericModel)
    update.max("int_field", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "int_field", {"value": 100})


def test_max_valid_float_field_with_type_validation():
    """Test max on a valid float field with type validation."""
    update = Update(NumericModel)
    update.max("float_field", 99.9)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "float_field", {"value": 99.9})


def test_max_valid_union_numeric_field_with_type_validation():
    """Test max on a valid union numeric field with type validation."""
    update = Update(NumericModel)
    update.max("union_numeric", 50)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "union_numeric", {"value": 50})


def test_max_non_existent_field_raises_invalid_path_error():
    """Test max on a non-existent field raises InvalidPathError."""
    update = Update(NumericModel)
    with pytest.raises(InvalidPathError):
        update.max("non_existent_field", 100)
    assert len(update.build()) == 0


def test_max_non_numeric_model_field_raises_value_type_error():
    """Test max on a non-numeric field of a model raises ValueTypeError."""
    update = Update(User) # User model has 'name' as string.
    with pytest.raises(ValueTypeError):
        update.max("name", 100)
    assert len(update.build()) == 0


def test_max_value_as_string_int_raises_type_error():
    """Test max with a stringified integer value raises TypeError due to method check."""
    update = Update(NumericModel)
    with pytest.raises(TypeError, match="Max value must be numeric"):
        update.max("int_field", "100") # "100" is a string.
    assert len(update.build()) == 0


def test_max_value_as_non_numeric_string_raises_type_error():
    """Test max with a non-numeric string value raises TypeError due to method check."""
    update = Update(NumericModel)
    with pytest.raises(TypeError, match="Max value must be numeric"):
        update.max("int_field", "not a number")
    assert len(update.build()) == 0


# --- Tests for `max` with nested fields (`NestedTypes` model) ---

def test_max_valid_nested_field():
    """Test max operation on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.max("nested.inner.val", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "nested.inner.val", {"value": 100})


def test_max_non_existent_nested_field_raises_invalid_path_error():
    """Test max on a non-existent nested field raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.max("nested.non_existent_path", 100)
    assert len(update.build()) == 0


def test_max_non_numeric_nested_field_raises_value_type_error():
    """Test max on a non-numeric nested field (e.g. List[str]) raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError): # 'str_list' is List[str].
        update.max("str_list", 100)
    assert len(update.build()) == 0


# --- Tests for `max` without type validation (No model) ---

def test_max_simple_field_no_validation():
    """Test max on a simple field without type validation."""
    update = Update()
    update.max("score", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "score", {"value": 100})


def test_max_float_field_no_validation():
    """Test max with float value on a field without type validation."""
    update = Update()
    update.max("value", 999.9)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "value", {"value": 999.9})


def test_max_nested_field_no_validation():
    """Test max on a nested field string path without type validation."""
    update = Update()
    update.max("nested.field", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "nested.field", {"value": 0})


# --- Tests for `max` with edge case values (No model) ---

def test_max_with_negative_value():
    """Test max with a negative value."""
    update = Update().max("field1", -10)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "field1", {"value": -10})


def test_max_with_zero_value():
    """Test max with zero as the value."""
    update = Update().max("field2", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "field2", {"value": 0})


def test_max_with_large_number_value():
    """Test max with a very large number as the value."""
    large_num = 1e9
    update = Update().max("field3", large_num)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "field3", {"value": large_num})


# --- Test for `max` with DSL-like nested path (`Entity` model) ---

def test_update_dsl_for_nested_field_max():
    """Test max operation on nested numeric fields using string path (DSL-like)."""
    update = Update(Entity).max("metadata.limits.ceiling", 100)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MaxOperation)
    assert operation.field_path == "metadata.limits.ceiling"
    assert operation.value == 100