import pytest
from async_repository.base.update import Update
from .conftest import User, NestedTypes, Organization


def test_push_with_type_validation():
    """Test that push operations are type validated."""
    update = Update(User)

    # Valid push
    update.push("tags", "new_tag")

    # Invalid push (wrong type)
    with pytest.raises(TypeError):
        update.push("tags", 123)  # tags should contain strings

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.push("name", "value")  # name is str, not a list

    # Non-existent field
    with pytest.raises(TypeError):
        update.push("non_existent", "value")


def test_push_with_nested_fields():
    """Test push operations with nested list fields."""
    update = Update(User)

    # Valid nested push (assuming addresses is a list of Address objects)
    address_dict = {"street": "123 Main St", "city": "Anytown", "zipcode": "12345"}
    update.push("addresses", address_dict)

    # Invalid nested push (wrong type for list item)
    with pytest.raises(TypeError):
        # Assuming addresses expects Address objects, not strings
        update.push("addresses", "not an address")

    # Non-existent nested field
    with pytest.raises(TypeError):
        update.push("metadata.list_field", "value")  # Metadata has no list_field


def test_push_with_complex_types():
    """Test push operations with complex types in lists."""
    update = Update(NestedTypes)

    # Valid push to simple list
    update.push("simple_list", 42)

    # Valid push to string list
    update.push("str_list", "new_string")

    # Valid push to complex list
    complex_item = {"name": "test", "value": 42}
    update.push("complex_list", complex_item)

    # Invalid push to simple list (wrong type)
    with pytest.raises(TypeError):
        update.push("simple_list", "not an int")

    # Invalid push to string list (wrong type)
    with pytest.raises(TypeError):
        update.push("str_list", 42)

    # Invalid push to complex list (wrong structure)
    with pytest.raises(TypeError):
        update.push("complex_list", {"name": 123, "value": "not an int"})


def test_push_without_model_type():
    """Test that push works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.push("tags", "new_tag")
    update.push("numbers", 42)
    update.push("complex_items", {"key": "value"})
    update.push("nested.list", "nested item")

    # Build should succeed
    result = update.build()
    assert "$push" in result
    assert result["$push"]["tags"] == "new_tag"
    assert result["$push"]["numbers"] == 42
    assert result["$push"]["complex_items"] == {"key": "value"}
    assert result["$push"]["nested.list"] == "nested item"


def test_push_build_result():
    """Test that push operations build the correct MongoDB update document."""
    update = Update().push("tags", "tag1").push("items", {"id": 1, "name": "Item 1"})

    result = update.build()
    assert "$push" in result
    assert len(result["$push"]) == 2
    assert result["$push"]["tags"] == "tag1"
    assert result["$push"]["items"] == {"id": 1, "name": "Item 1"}


def test_push_to_nested_list():
    """Test pushing to lists within nested objects."""

    update = Update(Organization)

    # Push to a list within a nested object
    update.push("departments.0.members", "new_member")

    # Push to a deeply nested list
    update.push("departments.0.categories.0.items", "new_item")
    update.push("departments.0.categories.0.tags", "new_tag")
    update.push("departments.0.categories.0.counts", 42)

    # Invalid push (wrong type)
    with pytest.raises(TypeError):
        update.push("departments.0.categories.0.items", 123)  # items contains strings

    with pytest.raises(TypeError):
        update.push(
            "departments.0.categories.0.counts", "not-a-number"
        )  # counts contains integers

    # Build and check the result
    result = update.build()
    assert "$push" in result
    assert result["$push"]["departments.0.members"] == "new_member"
    assert result["$push"]["departments.0.categories.0.items"] == "new_item"
    assert result["$push"]["departments.0.categories.0.tags"] == "new_tag"
    assert result["$push"]["departments.0.categories.0.counts"] == 42
