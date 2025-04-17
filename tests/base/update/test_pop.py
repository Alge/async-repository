import pytest
from async_repository.base.update import Update
from .conftest import User, NestedTypes, Organization


def test_pop_with_type_validation():
    """Test that pop operations are type validated."""
    update = Update(User)

    # Valid pop operations
    update.pop("tags")  # Default direction (1 - last element)
    update.pop("tags", 1)  # Last element
    update.pop("tags", -1)  # First element

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pop("name")  # name is str, not a list

    # Non-existent field
    with pytest.raises(TypeError):
        update.pop("non_existent")

    # Invalid direction
    with pytest.raises(ValueError):
        update.pop("tags", 2)  # direction must be 1 or -1

    with pytest.raises(ValueError):
        update.pop("tags", 0)  # direction must be 1 or -1

    with pytest.raises(ValueError):
        update.pop("tags", -2)  # direction must be 1 or -1


def test_pop_with_nested_fields():
    """Test pop operations with nested list fields."""
    update = Update(User)

    # Valid nested pop
    update.pop("addresses")  # Pop from a nested list field

    # Invalid nested field (not a list)
    with pytest.raises(TypeError):
        update.pop("metadata.key1")  # key1 is not a list

    # Non-existent nested field
    with pytest.raises(TypeError):
        update.pop("metadata.non_existent")


def test_pop_with_complex_types():
    """Test pop operations with complex list types."""
    update = Update(NestedTypes)

    # Valid pop from different list types
    update.pop("simple_list")
    update.pop("str_list")
    update.pop("complex_list")

    # Invalid pop from non-list field
    with pytest.raises(TypeError):
        update.pop("counter")  # counter is an int, not a list

    with pytest.raises(TypeError):
        update.pop("dict_field")  # dict_field is a dict, not a list


def test_pop_without_model_type():
    """Test that pop works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.pop("tags")
    update.pop("numbers", 1)
    update.pop("nested.list", -1)

    # Build should succeed
    result = update.build()
    assert "$pop" in result
    assert result["$pop"]["tags"] == 1  # Default direction
    assert result["$pop"]["numbers"] == 1
    assert result["$pop"]["nested.list"] == -1


def test_pop_build_result():
    """Test that pop operations build the correct MongoDB update document."""
    update = Update().pop("tags").pop("items", -1)

    result = update.build()
    assert "$pop" in result
    assert len(result["$pop"]) == 2
    assert result["$pop"]["tags"] == 1  # Default is 1 (last element)
    assert result["$pop"]["items"] == -1  # First element


def test_pop_from_nested_list():
    """Test popping from lists within nested objects."""

    update = Update(Organization)

    # Pop from a list within a nested object
    update.pop("departments.0.members")  # Default direction (last)
    update.pop("departments.0.members", -1)  # First element

    # Pop from a deeply nested list
    update.pop("departments.0.categories.0.items", 1)  # Last element
    update.pop("departments.0.categories.0.tags")  # Default (last)
    update.pop("departments.0.categories.0.counts", -1)  # First element

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pop("departments.0.name")  # name is a string, not a list

    # Invalid nested path
    with pytest.raises(TypeError):
        update.pop("departments.0.categories.0.non_existent")

    # Invalid direction
    with pytest.raises(ValueError):
        update.pop("departments.0.categories.0.items", 2)  # direction must be 1 or -1

    # Build and check the result
    result = update.build()
    assert "$pop" in result
    assert result["$pop"]["departments.0.members"] == -1  # Last operation wins
    assert result["$pop"]["departments.0.categories.0.items"] == 1
    assert result["$pop"]["departments.0.categories.0.tags"] == 1
    assert result["$pop"]["departments.0.categories.0.counts"] == -1