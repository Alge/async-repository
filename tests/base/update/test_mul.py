# tests/base/update/test_mul.py

import pytest
from async_repository.base.update import (
    Update,
    MultiplyOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.conftest import Entity
from tests.base.conftest import User, NumericModel, NestedTypes, assert_operation_present


def test_mul_basic_operation():
    """Test a basic multiply operation builds correctly without type validation."""
    update = Update().mul("price", 1.1)
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MultiplyOperation)
    assert operation.field_path == "price"
    assert operation.factor == 1.1


# --- Tests for `mul` with type validation (`NumericModel`) ---

def test_mul_valid_int_field_with_type_validation():
    """Test multiply on a valid integer field with type validation."""
    update = Update(NumericModel)
    update.mul("int_field", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "int_field", {"factor": 2})


def test_mul_valid_float_field_with_type_validation():
    """Test multiply on a valid float field with type validation."""
    update = Update(NumericModel)
    update.mul("float_field", 0.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "float_field", {"factor": 0.5})


def test_mul_valid_union_numeric_field_with_type_validation():
    """Test multiply on a valid union numeric field with type validation."""
    update = Update(NumericModel)
    update.mul("union_numeric", 1.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "union_numeric", {"factor": 1.5})


def test_mul_non_existent_field_raises_invalid_path_error():
    """Test multiply on a non-existent field raises InvalidPathError."""
    update = Update(NumericModel)
    with pytest.raises(InvalidPathError):
        update.mul("non_existent_field", 2)
    assert len(update.build()) == 0


def test_mul_non_numeric_model_field_raises_value_type_error():
    """Test multiply on a non-numeric field of a model raises ValueTypeError."""
    update = Update(User) # User model has 'name' as string.
    with pytest.raises(ValueTypeError):
        update.mul("name", 2)
    assert len(update.build()) == 0


def test_mul_factor_as_string_int_raises_type_error():
    """Test multiply with a stringified integer factor raises TypeError due to method check."""
    update = Update(NumericModel)
    with pytest.raises(TypeError, match="Multiply factor must be numeric"):
        update.mul("int_field", "2") # "2" is a string.
    assert len(update.build()) == 0


# --- Tests for `mul` with nested fields (`NestedTypes` model) ---

def test_mul_valid_nested_field():
    """Test multiply operation on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.mul("nested.inner.val", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "nested.inner.val", {"factor": 2})


def test_mul_non_existent_nested_field_raises_invalid_path_error():
    """Test multiply on a non-existent nested field raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.mul("nested.non_existent_path", 2)
    assert len(update.build()) == 0


def test_mul_non_numeric_nested_field_raises_value_type_error():
    """Test multiply on a non-numeric nested field (e.g. List[str]) raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError): # 'str_list' is List[str].
        update.mul("str_list", 2)
    assert len(update.build()) == 0


# --- Tests for `mul` without type validation (No model) ---

def test_mul_simple_field_no_validation():
    """Test multiply on a simple field without type validation."""
    update = Update()
    update.mul("price", 1.1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "price", {"factor": 1.1})


def test_mul_int_factor_field_no_validation():
    """Test multiply with integer factor on a field without type validation."""
    update = Update()
    update.mul("quantity", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "quantity", {"factor": 2})


def test_mul_nested_field_no_validation():
    """Test multiply on a nested field string path without type validation."""
    update = Update()
    update.mul("nested.field", 0.75)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "nested.field", {"factor": 0.75})


# --- Tests for `mul` with edge case factor values (No model) ---

def test_mul_by_zero():
    """Test multiply by zero."""
    update = Update().mul("field1", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "field1", {"factor": 0})


def test_mul_by_negative_one():
    """Test multiply by negative one."""
    update = Update().mul("field2", -1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "field2", {"factor": -1})


def test_mul_by_small_number():
    """Test multiply by a very small (close to zero) number."""
    small_num = 0.001
    update = Update().mul("field3", small_num)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "field3", {"factor": small_num})


def test_mul_by_large_number():
    """Test multiply by a very large number."""
    large_num = 1000
    update = Update().mul("field4", large_num)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "field4", {"factor": large_num})


# --- Test for `mul` with DSL-like nested path (`Entity` model) ---

def test_update_dsl_for_nested_field_multiply():
    """Test multiply operation on nested numeric fields using string path (DSL-like)."""
    update = Update(Entity).mul("metadata.rates.base", 2)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, MultiplyOperation)
    assert operation.field_path == "metadata.rates.base"
    assert operation.factor == 2