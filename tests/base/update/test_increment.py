import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_increment_with_type_validation():
    """Test that increment operations are type validated."""
    update = Update(User)

    # Valid increments
    update.increment("points", 10)
    update.increment("balance", 5.5)
    update.increment("points")  # Default increment by 1
    update.increment("score", 3)  # Union type

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.increment("non_existent", 5)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        update.increment("name", 5)  # name is string, not numeric

    # Invalid amount type
    with pytest.raises(TypeError):
        update.increment("points", "5")  # amount must be numeric, not string

    with pytest.raises(TypeError):
        update.increment("points", [5])  # amount must be numeric, not list


def test_increment_with_optional_and_union_types():
    """Test increment with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Union of numeric types
    update.increment("union_numeric", 5)
    update.increment("union_numeric", 3.14)

    # Optional numeric type
    update.increment("optional_int", 10)

    result = update.build()
    assert result["$inc"]["union_numeric"] == 8.14  # Operations are combined
    assert result["$inc"]["optional_int"] == 10


def test_increment_nested_fields():
    """Test increment operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested increment
    update.increment("nested.inner.val", 5)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.increment("nested.non_existent", 5)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.increment("str_list", 5)  # str_list is a list, not numeric


def test_increment_without_type_validation():
    """Test that increment works without a model type (no validation)."""
    update = Update()  # No model type

    # All these operations should work without errors
    update.increment("counter", 5)
    update.increment("value", 10.5)
    update.increment("score")  # Default increment by 1
    update.increment("nested.counter", 3)

    # Build should succeed
    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == 5
    assert result["$inc"]["value"] == 10.5
    assert result["$inc"]["score"] == 1
    assert result["$inc"]["nested.counter"] == 3


def test_increment_build_result():
    """Test that increment operations build the correct MongoDB update document."""
    update = Update().increment("views", 1).increment("score", 5.5)

    result = update.build()
    assert "$inc" in result
    assert len(result["$inc"]) == 2
    assert result["$inc"]["views"] == 1
    assert result["$inc"]["score"] == 5.5


def test_increment_zero():
    """Test incrementing by zero."""
    update = Update().increment("counter", 0)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == 0


def test_increment_negative():
    """Test that increment can handle negative values (acts like decrement)."""
    update = Update().increment("counter", -5)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == -5


def test_increment_float_precision():
    """Test that increment handles float precision correctly."""
    update = Update().increment("value", 0.1).increment("value", 0.2)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["value"] == pytest.approx(
        0.3
    )  # Account for floating point precision
