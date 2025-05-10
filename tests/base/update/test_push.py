# tests/base/update/test_push.py

import pytest
from async_repository.base.update import (
    Update,
    PushOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.conftest import Entity
from tests.base.conftest import (
    User,
    NestedTypes,
    Organization,
    Address,
    ComplexItem,
)  # Import necessary models

# Import the prepare_for_storage function to test serialized values
from async_repository.base.utils import prepare_for_storage


from tests.base.conftest import assert_operation_present


def test_push_with_type_validation():
    """Test that push operations are type validated."""
    update = Update(User)

    # Valid push
    update.push("tags", "new_tag")

    # Invalid push (wrong type) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'tags'"):
        update.push("tags", 123)  # tags should contain strings

    # Invalid field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.push("name", "value")  # name is str, not a list

    # Non-existent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.push("non_existent", "value")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1  # Only the valid operation
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})


def test_push_with_nested_fields():
    """Test push operations with nested list fields."""
    update = Update(User)

    # Valid nested push - addresses is List[Address]
    # We push an Address object, which gets serialized by prepare_for_storage
    address_obj = Address("123 Main St", "Anytown", "12345")
    serialized_address = prepare_for_storage(address_obj)
    update.push("addresses", address_obj)  # Pass the object directly

    # Invalid nested push (wrong type for list item) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'addresses'"
    ):
        # Pushing a string to a list expecting Address-like objects
        update.push("addresses", "not an address object")

    # Non-existent nested field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.push("metadata.list_field", "value")  # Metadata has no list_field

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1  # Only the valid push
    # Check the 'items' attribute contains the serialized address dict
    assert_operation_present(
        result, PushOperation, "addresses", {"items": [serialized_address]}
    )


def test_push_with_complex_types():
    """Test push operations with complex types in lists."""
    update = Update(NestedTypes)

    # Valid push to simple list (List[int])
    update.push("simple_list", 42)

    # Valid push to string list (List[str])
    update.push("str_list", "new_string")

    # Valid push to complex list (List[ComplexItem])
    complex_item_obj = ComplexItem(name="test", value=42)
    serialized_complex_item = prepare_for_storage(complex_item_obj)
    update.push("complex_list", complex_item_obj)  # Pass the object

    # Invalid push to simple list (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'simple_list'"
    ):
        update.push("simple_list", "not an int")

    # Invalid push to string list (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'str_list'"
    ):
        update.push("str_list", 42)

    # Invalid push to complex list (wrong structure/type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'complex_list'"
    ):
        # Provide a dict that doesn't match ComplexItem structure
        update.push("complex_list", {"name": 123, "value": "not an int"})

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3  # Only valid pushes
    assert_operation_present(result, PushOperation, "simple_list", {"items": [42]})
    assert_operation_present(
        result, PushOperation, "str_list", {"items": ["new_string"]}
    )
    assert_operation_present(
        result, PushOperation, "complex_list", {"items": [serialized_complex_item]}
    )


def test_push_without_model_type():
    """Test that push works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.push("tags", "new_tag")
    update.push("numbers", 42)
    update.push("complex_items", {"key": "value"})  # Item is a dict
    update.push("nested.list", "nested item")

    # Build should succeed
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})
    assert_operation_present(result, PushOperation, "numbers", {"items": [42]})
    assert_operation_present(
        result, PushOperation, "complex_items", {"items": [{"key": "value"}]}
    )
    assert_operation_present(
        result, PushOperation, "nested.list", {"items": ["nested item"]}
    )


def test_push_build_result():
    """Test that push operations build the correct agnostic operation list."""
    update = Update().push("tags", "tag1").push("items", {"id": 1, "name": "Item 1"})

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(result, PushOperation, "tags", {"items": ["tag1"]})
    assert_operation_present(
        result, PushOperation, "items", {"items": [{"id": 1, "name": "Item 1"}]}
    )


def test_push_to_nested_list():
    """Test pushing to lists within nested objects."""
    update = Update(Organization)

    # Push to a list within a nested object (List[str])
    update.push("departments.0.members", "new_member")

    # Push to a deeply nested list (List[str], List[str], List[int])
    update.push("departments.0.categories.0.items", "new_item")
    update.push("departments.0.categories.0.tags", "new_tag")
    update.push("departments.0.categories.0.counts", 42)

    # Invalid push (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for push to 'departments.0.categories.0.items'",
    ):
        update.push("departments.0.categories.0.items", 123)  # items expects strings

    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for push to 'departments.0.categories.0.counts'",
    ):
        update.push(
            "departments.0.categories.0.counts", "not-a-number"
        )  # counts expects integers

    # Build and check the result
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4  # Only valid pushes
    assert_operation_present(
        result, PushOperation, "departments.0.members", {"items": ["new_member"]}
    )
    assert_operation_present(
        result,
        PushOperation,
        "departments.0.categories.0.items",
        {"items": ["new_item"]},
    )
    assert_operation_present(
        result, PushOperation, "departments.0.categories.0.tags", {"items": ["new_tag"]}
    )
    assert_operation_present(
        result, PushOperation, "departments.0.categories.0.counts", {"items": [42]}
    )


def test_update_dsl_nested_push():
    """Test push operation on nested array fields."""
    # Create an update with push operation on nested path
    update = Update(Entity).push("metadata.collections.tags", "new")

    # Assert the operation was added correctly
    assert len(update._operations) == 1
    assert isinstance(update._operations[0], PushOperation)
    assert update._operations[0].field_path == "metadata.collections.tags"
    assert "new" in update._operations[0].items