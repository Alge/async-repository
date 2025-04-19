import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_decrement_with_type_validation():
    """Test that decrement operations are type validated."""
    update = Update(User)

    # Valid decrements - one per field
    update.decrement(update.fields.points, 10)
    update.decrement(update.fields.balance, 5.5)
    update.decrement(update.fields.score, 3)  # Union type

    # Invalid field (non-existent)
    with pytest.raises(TypeError):
        update.decrement("non_existent", 5)

    # Invalid field (non-numeric)
    with pytest.raises(TypeError):
        update.decrement(update.fields.name, 5)  # name is string, not numeric

    # Invalid amount type
    with pytest.raises(TypeError):
        update.decrement(update.fields.score, "5")  # amount must be numeric, not string

    with pytest.raises(TypeError):
        update.decrement(update.fields.score, [5])  # amount must be numeric, not list


def test_multiple_decrement_rejected():
    """Test that multiple decrement operations on same field are rejected."""
    update = Update(User)

    # First decrement is allowed
    update.decrement(update.fields.points, 10)

    # Second decrement on same field should be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.points, 5)

    # Default value decrement is also rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.points)

    # But decrement on a different field is fine
    update.decrement(update.fields.balance, 5.5)

    # Verify only original decrement operations are in the update
    result = update.build()
    assert result["$inc"]["points"] == -10
    assert result["$inc"]["balance"] == -5.5


def test_decrement_after_increment_rejected():
    """Test that decrement after increment on the same field is rejected."""
    update = Update(User)

    # First operation (increment) is allowed
    update.increment(update.fields.points, 5)

    # Decrement on the same field should be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.points, 10)

    # Verify the original increment remains
    result = update.build()
    assert result["$inc"]["points"] == 5


def test_decrement_with_optional_and_union_types():
    """Test decrement with Optional and Union numeric types."""
    update = Update(NumericModel)

    # Each field can only have one decrement
    update.decrement(update.fields.union_numeric, 5)
    update.decrement(update.fields.optional_int, 10)

    # Second decrement on the same field is rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.union_numeric, 3.14)

    result = update.build()
    assert result["$inc"]["union_numeric"] == -5  # Only the first decrement
    assert result["$inc"]["optional_int"] == -10


def test_decrement_nested_fields():
    """Test decrement operations with nested fields."""
    update = Update(NestedTypes)

    # Valid nested decrement
    update.decrement(update.fields.nested.inner.val, 5)

    # Multiple decrements on same nested field are rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.nested.inner.val, 3)

    # Invalid nested field (non-existent)
    with pytest.raises(TypeError):
        update.decrement("nested.non_existent", 5)

    # Invalid nested field (non-numeric)
    with pytest.raises(TypeError):
        update.decrement(update.fields.str_list, 5)  # str_list is a list, not numeric


def test_decrement_without_type_validation():
    """Test that decrement works without a model type (no validation)."""
    update = Update()  # No model type

    # Each field gets only one decrement
    update.decrement("counter", 5)
    update.decrement("value", 10.5)
    update.decrement("score")  # Default decrement by 1
    update.decrement("nested.counter", 3)

    # Multiple decrements on the same field are rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement("counter", 2)

    # Build should succeed with original decrements
    result = update.build()
    assert "$inc" in result
    assert result["$inc"]["counter"] == -5
    assert result["$inc"]["value"] == -10.5
    assert result["$inc"]["score"] == -1
    assert result["$inc"]["nested.counter"] == -3


def test_decrement_build_result():
    """Test that decrement operations build the correct MongoDB update document."""
    update = Update()
    # Use separate fields
    update.decrement("views", 1)
    update.decrement("score", 5.5)

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
    # Create separate update objects for each case
    update1 = Update(NumericModel)
    update2 = Update(NumericModel)

    # Use separate update objects
    update1.increment(update1.fields.int_field, -10)
    update2.decrement(update2.fields.int_field, 10)

    result1 = update1.build()
    result2 = update2.build()

    assert result1["$inc"]["int_field"] == result2["$inc"]["int_field"] == -10