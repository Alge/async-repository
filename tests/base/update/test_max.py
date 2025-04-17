import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_max_basic():
    """Test basic max operation functionality."""
    update = Update().max("score", 100)

    result = update.build()
    assert "$max" in result
    assert result["$max"]["score"] == 100


def test_max_with_type_validation():
    """Test that max operations are type validated."""
    update = Update(NumericModel)

    # Valid max operations
    update.max("int_field", 100)
    update.max("float_field", 99.9)
    update.max("union_numeric", 50)

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.max("non_existent", 100)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        Update(User).max("name", 100)

    # Invalid value type
    with pytest.raises(TypeError):
        update.max("int_field", "100")


def test_max_with_nested_fields():
    """Test max operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested max
    update.max("nested.inner.val", 100)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.max("nested.non_existent", 100)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.max("str_list", 100)


def test_max_without_type_validation():
    """Test that max works without a model type (no validation)."""
    update = Update()

    update.max("score", 100)
    update.max("value", 999.9)
    update.max("nested.field", 0)

    result = update.build()
    assert "$max" in result
    assert result["$max"]["score"] == 100
    assert result["$max"]["value"] == 999.9
    assert result["$max"]["nested.field"] == 0


def test_max_edge_cases():
    """Test max with edge case values."""
    update = Update()

    # Negative values
    update.max("field1", -10)

    # Zero
    update.max("field2", 0)

    # Very large number
    update.max("field3", 1e9)

    result = update.build()
    assert result["$max"]["field1"] == -10
    assert result["$max"]["field2"] == 0
    assert result["$max"]["field3"] == 1e9