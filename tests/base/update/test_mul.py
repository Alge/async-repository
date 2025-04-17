import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_mul_basic():
    """Test basic mul operation functionality."""
    update = Update().mul("price", 1.1)

    result = update.build()
    assert "$mul" in result
    assert result["$mul"]["price"] == 1.1


def test_mul_with_type_validation():
    """Test that mul operations are type validated."""
    update = Update(NumericModel)

    # Valid mul operations
    update.mul("int_field", 2)
    update.mul("float_field", 0.5)
    update.mul("union_numeric", 1.5)

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.mul("non_existent", 2)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        Update(User).mul("name", 2)

    # Invalid value type
    with pytest.raises(TypeError):
        update.mul("int_field", "2")


def test_mul_with_nested_fields():
    """Test mul operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested mul
    update.mul("nested.inner.val", 2)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.mul("nested.non_existent", 2)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.mul("str_list", 2)


def test_mul_without_type_validation():
    """Test that mul works without a model type (no validation)."""
    update = Update()

    update.mul("price", 1.1)
    update.mul("quantity", 2)
    update.mul("nested.field", 0.75)

    result = update.build()
    assert "$mul" in result
    assert result["$mul"]["price"] == 1.1
    assert result["$mul"]["quantity"] == 2
    assert result["$mul"]["nested.field"] == 0.75


def test_mul_edge_cases():
    """Test multiplication with special values."""
    update = Update()

    # Multiply by 0 (zeroes out the field)
    update.mul("field1", 0)

    # Multiply by negative (changes sign)
    update.mul("field2", -1)

    # Multiply by very small number
    update.mul("field3", 0.001)

    # Multiply by very large number
    update.mul("field4", 1000)

    result = update.build()
    assert result["$mul"]["field1"] == 0
    assert result["$mul"]["field2"] == -1
    assert result["$mul"]["field3"] == 0.001
    assert result["$mul"]["field4"] == 1000