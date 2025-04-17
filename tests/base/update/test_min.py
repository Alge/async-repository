import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_min_basic():
    """Test basic min operation functionality."""
    update = Update().min("score", 50)

    result = update.build()
    assert "$min" in result
    assert result["$min"]["score"] == 50


def test_min_with_type_validation():
    """Test that min operations are type validated."""
    update = Update(NumericModel)

    # Valid min operations
    update.min("int_field", 10)
    update.min("float_field", 5.5)
    update.min("union_numeric", 3)

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.min("non_existent", 5)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        Update(User).min("name", 5)

    # Invalid value type
    with pytest.raises(TypeError):
        update.min("int_field", "5")


def test_min_with_nested_fields():
    """Test min operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested min
    update.min("nested.inner.val", 5)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.min("nested.non_existent", 5)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.min("str_list", 5)


def test_min_without_type_validation():
    """Test that min works without a model type (no validation)."""
    update = Update()

    update.min("score", 50)
    update.min("value", -10.5)
    update.min("nested.field", 0)

    result = update.build()
    assert "$min" in result
    assert result["$min"]["score"] == 50
    assert result["$min"]["value"] == -10.5
    assert result["$min"]["nested.field"] == 0


def test_min_edge_cases():
    """Test min with edge case values."""
    update = Update()

    # Negative values
    update.min("field1", -100)

    # Zero
    update.min("field2", 0)

    # Very small number
    update.min("field3", 0.0001)

    result = update.build()
    assert result["$min"]["field1"] == -100
    assert result["$min"]["field2"] == 0
    assert result["$min"]["field3"] == 0.0001