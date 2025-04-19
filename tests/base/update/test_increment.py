import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_increment_with_type_validation():
    """Test that increment operations are type validated."""
    update = Update(User)

    # Valid increment - one per field
    update.increment(update.fields.points, 10)
    update.increment(update.fields.balance, 5.5)
    update.increment(update.fields.score, 3)  # Union type

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.increment("non_existent", 5)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        update.increment(update.fields.name, 5)  # name is string, not numeric

    # Invalid amount type
    with pytest.raises(TypeError):
        update.increment(update.fields.score, "5")  # amount must be numeric, not string

    with pytest.raises(TypeError):
        update.increment(update.fields.score, [5])  # amount must be numeric, not list


def test_multiple_increment_rejected():
    """Test that multiple increment operations on same field are rejected."""
    update = Update(User)

    # First increment is allowed
    update.increment(update.fields.points, 10)

    # Second increment on same field should be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.points, 5)

    # Default value increment is also rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.points)

    # But increment on a different field is fine
    update.increment(update.fields.balance, 5.5)

    # Verify only original increment operations are in the update
    result = update.build()
    assert result["$inc"]["points"] == 10
    assert result["$inc"]["balance"] == 5.5


def test_increment_after_decrement_rejected():
    """Test that increment after decrement on the same field is rejected."""
    update = Update(User)

    # First operation (decrement) is allowed
    update.decrement(update.fields.points, 5)

    # Increment on the same field should be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.points, 10)

    # Verify the original decrement remains
    result = update.build()
    assert result["$inc"]["points"] == -5


def test_increment_with_optional_and_union_types():
    """Test increment with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Each field can only have one increment
    update.increment(update.fields.union_numeric, 5)
    update.increment(update.fields.optional_int, 10)

    # Second increment on the same field is rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.union_numeric, 3.14)

    result = update.build()
    assert result["$inc"]["union_numeric"] == 5  # Only the first increment
    assert result["$inc"]["optional_int"] == 10


def test_increment_nested_fields():
    """Test increment operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested increment
    update.increment(update.fields.nested.inner.val, 5)

    # Multiple increments on same nested field are rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.nested.inner.val, 3)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.increment("nested.non_existent", 5)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.increment(update.fields.str_list, 5)  # str_list is a list, not numeric


def test_increment_without_type_validation():
    """Test that increment works without a model type (no validation)."""
    update = Update()  # No model type

    # Each field gets only one increment
    update.increment("counter", 5)
    update.increment("value", 10.5)
    update.increment("score")  # Default increment by 1
    update.increment("nested.counter", 3)

    # Multiple increments on the same field are rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment("counter", 2)

    # Build should succeed with original increments
    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == 5
    assert result["$inc"]["value"] == 10.5
    assert result["$inc"]["score"] == 1
    assert result["$inc"]["nested.counter"] == 3


def test_increment_build_result():
    """Test that increment operations build the correct MongoDB update document."""
    update = Update()
    # Use separate fields
    update.increment("views", 1)
    update.increment("score", 5.5)

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
    """Test that increment handles float precision."""
    # Using separate fields since multiple increments are not allowed
    update = Update().increment("value1", 0.1).increment("value2", 0.2)

    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["value1"] == pytest.approx(0.1)
    assert result["$inc"]["value2"] == pytest.approx(0.2)