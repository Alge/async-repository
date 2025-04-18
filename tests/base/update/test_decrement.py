import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_decrement_with_type_validation():
    """Test that decrement operations are type validated."""
    update = Update(User)

    # Valid decrements
    update.decrement("points", 10)
    update.decrement("balance", 5.5)
    update.decrement("points")  # Default decrement by 1
    update.decrement("score", 3)  # Union type

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.decrement("non_existent", 5)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        update.decrement("name", 5)  # name is string, not numeric

    # Invalid amount type
    with pytest.raises(TypeError):
        update.decrement("points", "5")  # amount must be numeric, not string

    with pytest.raises(TypeError):
        update.decrement("points", [5])  # amount must be numeric, not list


def test_decrement_with_optional_and_union_types():
    """Test decrement with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Union of numeric types
    update.decrement("union_numeric", 5)
    update.decrement("union_numeric", 3.14)

    # Optional numeric type
    update.decrement("optional_int", 10)

    result = update.build()
    assert result["$inc"]["union_numeric"] == -8.14  # Operations are combined
    assert result["$inc"]["optional_int"] == -10


def test_decrement_nested_fields():
    """Test decrement operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested decrement
    update.decrement("nested.inner.val", 5)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.decrement("nested.non_existent", 5)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.decrement("str_list", 5)  # str_list is a list, not numeric


def test_decrement_without_type_validation():
    """Test that decrement works without a model type (no validation)."""
    update = Update()  # No model type

    # All these operations should work without errors
    update.decrement("counter", 5)
    update.decrement("value", 10.5)
    update.decrement("score")  # Default decrement by 1
    update.decrement("nested.counter", 3)

    # Build should succeed
    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == -5
    assert result["$inc"]["value"] == -10.5
    assert result["$inc"]["score"] == -1
    assert result["$inc"]["nested.counter"] == -3


def test_decrement_build_result():
    """Test that decrement operations build the correct MongoDB update document."""
    update = Update().decrement("views", 1).decrement("score", 5.5)

    result = update.build()
    assert "$inc" in result
    assert len(result["$inc"]) == 2
    assert result["$inc"]["views"] == -1
    assert result["$inc"]["score"] == -5.5


def test_decrement_zero():
    """Test decrementing by zero."""
    update = Update().decrement("counter", 0)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == 0


def test_decrement_negative():
    """Test that decrement with negative values (acts like increment)."""
    update = Update().decrement("counter", -5)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == 5  # Negative decrement is a positive increment


def test_decrement_reuses_increment():
    """Test that decrement reuses the increment method with negative value."""
    update = Update(NumericModel)

    # These should be equivalent
    update1 = Update(NumericModel).increment("int_field", -10)
    update2 = Update(NumericModel).decrement("int_field", 10)

    result1 = update1.build()
    result2 = update2.build()

    assert result1["$inc"]["int_field"] == result2["$inc"]["int_field"] == -10
