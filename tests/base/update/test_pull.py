import pytest
from async_repository.base.update import Update
from .conftest import User, NestedTypes, Organization


def test_pull_with_type_validation():
    """Test that pull operations are type validated."""
    update = Update(User)

    # Valid pull
    update.pull("tags", "tag_to_remove")

    # Invalid pull (wrong type)
    with pytest.raises(TypeError):
        update.pull("tags", 123)  # tags should contain strings

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pull("name", "value")  # name is str, not a list

    # Non-existent field
    with pytest.raises(TypeError):
        update.pull("non_existent", "value")


def test_pull_with_nested_fields():
    """Test pull operations with nested list fields."""
    update = Update(User)

    # Valid nested pull (assuming addresses is a list of Address objects)
    address_dict = {"street": "123 Main St", "city": "Anytown", "zipcode": "12345"}
    update.pull("addresses", address_dict)

    # Invalid nested pull (wrong type for list item)
    with pytest.raises(TypeError):
        # Assuming addresses expects Address objects, not ints
        update.pull("addresses", 123)

    # Non-existent nested field
    with pytest.raises(TypeError):
        update.pull("metadata.list_field", "value")  # Metadata has no list_field


def test_pull_with_complex_types():
    """Test pull operations with complex types in lists."""
    update = Update(NestedTypes)

    # Valid pull from simple list
    update.pull("simple_list", 42)

    # Valid pull from string list
    update.pull("str_list", "string_to_remove")

    # Valid pull from complex list
    complex_item = {"name": "test", "value": 42}
    update.pull("complex_list", complex_item)

    # Invalid pull from simple list (wrong type)
    with pytest.raises(TypeError):
        update.pull("simple_list", "not an int")

    # Invalid pull from string list (wrong type)
    with pytest.raises(TypeError):
        update.pull("str_list", 42)

    # Invalid pull from complex list (wrong structure)
    with pytest.raises(TypeError):
        update.pull("complex_list", {"name": 123, "value": "not an int"})


def test_pull_without_model_type():
    """Test that pull works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.pull("tags", "tag_to_remove")
    update.pull("numbers", 42)
    update.pull("complex_items", {"key": "value"})
    update.pull("nested.list", "nested item")

    # Build should succeed
    result = update.build()
    assert "$pull" in result
    assert result["$pull"]["tags"] == "tag_to_remove"
    assert result["$pull"]["numbers"] == 42
    assert result["$pull"]["complex_items"] == {"key": "value"}
    assert result["$pull"]["nested.list"] == "nested item"


def test_pull_build_result():
    """Test that pull operations build the correct MongoDB update document."""
    update = Update().pull("tags", "old_tag").pull("items", {"category": "removed"})

    result = update.build()
    assert "$pull" in result
    assert len(result["$pull"]) == 2
    assert result["$pull"]["tags"] == "old_tag"
    assert result["$pull"]["items"] == {"category": "removed"}


def test_pull_from_nested_list():
    """Test pulling specific values from lists within nested objects."""

    update = Update(Organization)

    # Pull from a list within a nested object
    update.pull("departments.0.members", "member1")

    # Pull from a deeply nested list
    update.pull("departments.0.categories.0.items", "item1")
    update.pull("departments.0.categories.0.tags", "tag1")
    update.pull("departments.0.categories.0.counts", 2)

    # Invalid pull (wrong type)
    with pytest.raises(TypeError):
        update.pull("departments.0.categories.0.items", 123)  # items contains strings

    with pytest.raises(TypeError):
        update.pull("departments.0.categories.0.counts",
                    "not-a-number")  # counts contains integers

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pull("departments.0.name", "value")  # name is a string, not a list

    # Invalid nested path
    with pytest.raises(TypeError):
        update.pull("departments.0.categories.0.non_existent", "value")

    # Build and check the result
    result = update.build()
    assert "$pull" in result
    assert result["$pull"]["departments.0.members"] == "member1"
    assert result["$pull"]["departments.0.categories.0.items"] == "item1"
    assert result["$pull"]["departments.0.categories.0.tags"] == "tag1"
    assert result["$pull"]["departments.0.categories.0.counts"] == 2


def test_pull_with_nested_object_criteria():
    """Test pulling with complex criteria from nested lists."""

    # Create a custom Organization with more complex data
    category = {
        "name": "Products",
        "items": ["product1", "product2", "product3"],
        "tags": ["featured", "new", "sale"],
        "counts": [10, 20, 30]
    }

    update = Update()

    # Pull using object criteria (simulating MongoDB query operators)
    update.pull("departments.0.categories", {"name": "Products"})

    # Pull using complex criteria
    update.pull("items", {"$in": ["item1", "item2"]})

    result = update.build()
    assert "$pull" in result
    assert result["$pull"]["departments.0.categories"] == {"name": "Products"}
    assert result["$pull"]["items"] == {"$in": ["item1", "item2"]}