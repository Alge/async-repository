import pytest
from async_repository.base.update import Update
from .conftest import User, NestedTypes


def test_unset_with_type_validation():
    """Test that unset operations are validated for field existence."""
    update = Update(User)

    # Valid unset operations
    update.unset("name")
    update.unset("email")
    update.unset("active")

    # Non-existent field
    with pytest.raises(TypeError):
        update.unset("non_existent")


def test_unset_with_nested_fields():
    """Test unset operations with nested fields."""
    update = Update(User)

    # Valid nested unset
    update.unset("metadata.key1")

    # Non-existent nested field
    with pytest.raises(TypeError):
        update.unset("metadata.non_existent")

    # Non-existent parent field
    with pytest.raises(TypeError):
        update.unset("non_existent.field")


def test_unset_with_complex_types():
    """Test unset operations with complex nested structures."""
    update = Update(NestedTypes)

    # Valid unset operations
    update.unset("counter")
    update.unset("nested.inner.val")

    # Non-existent nested field
    with pytest.raises(TypeError):
        update.unset("nested.inner.non_existent")


def test_unset_without_model_type():
    """Test that unset works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.unset("any_field")
    update.unset("nested.field")

    # Build should succeed
    result = update.build()
    assert "$unset" in result
    assert result["$unset"]["any_field"] == ""
    assert result["$unset"]["nested.field"] == ""


def test_unset_build_result():
    """Test that unset operations build the correct MongoDB update document."""
    update = Update().unset("name").unset("metadata.note")

    result = update.build()
    assert "$unset" in result
    assert len(result["$unset"]) == 2
    assert result["$unset"]["name"] == ""  # In MongoDB, value is ignored for $unset
    assert result["$unset"]["metadata.note"] == ""
