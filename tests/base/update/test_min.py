# tests/base/update/test_min.py

import pytest
from async_repository.base.update import (
    Update,
    MinOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.conftest import Entity
from tests.base.conftest import User, NumericModel, NestedTypes, assert_operation_present


def test_min_basic_operation():
    """Test a basic min operation builds correctly without type validation."""
    update = Update().min("score", 50)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MinOperation)
    assert operation.field_path == "score"
    assert operation.value == 50


# --- Tests for `min` with type validation (`NumericModel`) ---

def test_min_valid_int_field_with_type_validation():
    """Test min on a valid integer field with type validation."""
    update = Update(NumericModel)
    update.min("int_field", 10)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "int_field", {"value": 10})


def test_min_valid_float_field_with_type_validation():
    """Test min on a valid float field with type validation."""
    update = Update(NumericModel)
    update.min("float_field", 5.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "float_field", {"value": 5.5})


def test_min_valid_union_numeric_field_with_type_validation():
    """Test min on a valid union numeric field with type validation."""
    update = Update(NumericModel)
    update.min("union_numeric", 3)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "union_numeric", {"value": 3})


def test_min_non_existent_field_raises_invalid_path_error():
    """Test min on a non-existent field raises InvalidPathError."""
    update = Update(NumericModel)
    with pytest.raises(InvalidPathError):
        update.min("non_existent_field", 5)
    assert len(update.build()) == 0


def test_min_non_numeric_model_field_raises_value_type_error():
    """Test min on a non-numeric field of a model raises ValueTypeError."""
    update = Update(User) # User model has 'name' as string.
    with pytest.raises(ValueTypeError):
        update.min("name", 5)
    assert len(update.build()) == 0


def test_min_value_as_string_int_raises_type_error():
    """Test min with a stringified integer value raises TypeError due to method check."""
    update = Update(NumericModel)
    with pytest.raises(TypeError, match="Min value must be numeric"):
        update.min("int_field", "5") # "5" is a string.
    assert len(update.build()) == 0


def test_min_value_as_non_numeric_string_raises_type_error():
    """Test min with a non-numeric string value raises TypeError due to method check."""
    update = Update(NumericModel)
    with pytest.raises(TypeError, match="Min value must be numeric"):
        update.min("int_field", "not a number")
    assert len(update.build()) == 0


# --- Tests for `min` with nested fields (`NestedTypes` model) ---

def test_min_valid_nested_field():
    """Test min operation on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.min("nested.inner.val", 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "nested.inner.val", {"value": 5})


def test_min_non_existent_nested_field_raises_invalid_path_error():
    """Test min on a non-existent nested field raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.min("nested.non_existent_path", 5)
    assert len(update.build()) == 0


def test_min_non_numeric_nested_field_raises_value_type_error():
    """Test min on a non-numeric nested field (e.g. List[str]) raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError): # 'str_list' is List[str].
        update.min("str_list", 5)
    assert len(update.build()) == 0


# --- Tests for `min` without type validation (No model) ---

def test_min_simple_field_no_validation():
    """Test min on a simple field without type validation."""
    update = Update()
    update.min("score", 50)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "score", {"value": 50})


def test_min_float_field_no_validation():
    """Test min with float value on a field without type validation."""
    update = Update()
    update.min("value", -10.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "value", {"value": -10.5})


def test_min_nested_field_no_validation():
    """Test min on a nested field string path without type validation."""
    update = Update()
    update.min("nested.field", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "nested.field", {"value": 0})


# --- Tests for `min` with edge case values (No model) ---

def test_min_with_negative_value():
    """Test min with a negative value."""
    update = Update().min("field1", -100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "field1", {"value": -100})


def test_min_with_zero_value():
    """Test min with zero as the value."""
    update = Update().min("field2", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "field2", {"value": 0})


def test_min_with_small_number_value():
    """Test min with a very small (close to zero) number as the value."""
    small_num = 0.0001
    update = Update().min("field3", small_num)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "field3", {"value": small_num})


# --- Test for `min` with DSL-like nested path (`Entity` model) ---

def test_update_dsl_for_nested_field_min():
    """Test min operation on nested numeric fields using string path (DSL-like)."""
    update = Update(Entity).min("metadata.limits.floor", 30)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MinOperation)
    assert operation.field_path == "metadata.limits.floor"
    assert operation.value == 30