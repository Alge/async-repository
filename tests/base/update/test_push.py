# tests/base/update/test_push.py

import pytest
from async_repository.base.update import (
    Update,
    PushOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.conftest import Entity
from tests.base.conftest import (
    User,
    NestedTypes,
    Organization,
    Address,
    ComplexItem,
    assert_operation_present
)
from async_repository.base.utils import prepare_for_storage


# --- Tests for `push` with type validation (User model) ---

def test_push_valid_literal_with_type_validation():
    """Test push with a valid literal string to List[str] is validated."""
    update = Update(User)
    update.push("tags", "new_tag")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})

def test_push_invalid_literal_type_raises_value_type_error():
    """Test push with wrong literal type to list item raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'tags'"):
        update.push("tags", 123)  # tags list expects strings
    assert len(update.build()) == 0

def test_push_to_non_list_field_raises_invalid_path_error():
    """Test push to a non-list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.push("name", "value")  # name is str, not a list
    assert len(update.build()) == 0

def test_push_to_non_existent_field_raises_invalid_path_error():
    """Test push to a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.push("non_existent_field", "value")
    assert len(update.build()) == 0


# --- Tests for `push` with nested fields (User model) ---

def test_push_nested_object_valid():
    """Test push a valid object to a nested List[Address]."""
    update = Update(User)
    # Address model has zipcode: int
    address_obj = Address(street="123 Main St", city="Anytown", zipcode=12345)
    serialized_address = prepare_for_storage(address_obj)
    update.push("addresses", address_obj)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "addresses", {"items": [serialized_address]})

def test_push_nested_object_invalid_type_raises_value_type_error():
    """Test push an invalid type to List[Address] raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'addresses'"):
        update.push("addresses", "not_an_address_object")
    assert len(update.build()) == 0

def test_push_to_non_existent_nested_list_field_raises_invalid_path_error():
    """Test push to a non-existent nested list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError): # Metadata has no 'list_field'
        update.push("metadata.list_field", "value")
    assert len(update.build()) == 0


# --- Tests for `push` with complex types (NestedTypes model) ---

def test_push_to_list_of_int_valid():
    """Test push a valid int to List[int]."""
    update = Update(NestedTypes)
    update.push("simple_list", 42)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "simple_list", {"items": [42]})

def test_push_to_list_of_str_valid():
    """Test push a valid str to List[str]."""
    update = Update(NestedTypes)
    update.push("str_list", "new_string")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "str_list", {"items": ["new_string"]})

def test_push_to_list_of_complex_item_valid_object():
    """Test push a valid ComplexItem object to List[ComplexItem]."""
    update = Update(NestedTypes)
    complex_item_obj = ComplexItem(name="test_item", value=100)
    serialized_complex_item = prepare_for_storage(complex_item_obj)
    update.push("complex_list", complex_item_obj)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "complex_list", {"items": [serialized_complex_item]})

def test_push_to_list_of_int_invalid_type_raises_value_type_error():
    """Test push a string to List[int] raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'simple_list'"):
        update.push("simple_list", "not_an_int")
    assert len(update.build()) == 0

def test_push_to_list_of_str_invalid_type_raises_value_type_error():
    """Test push an int to List[str] raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'str_list'"):
        update.push("str_list", 12345)
    assert len(update.build()) == 0

def test_push_to_list_of_complex_item_invalid_type_raises_value_type_error():
    """Test push a non-matching dict/object to List[ComplexItem] raises ValueTypeError."""
    update = Update(NestedTypes)
    # This dict does not match ComplexItem structure (e.g., name is int, expecting str)
    invalid_item_data = {"name": 123, "value": "not_an_int_either"}
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'complex_list'"):
        update.push("complex_list", invalid_item_data)
    assert len(update.build()) == 0


# --- Tests for `push` without a model type (no validation) ---

def test_push_literal_string_no_model():
    """Test push literal string without model type."""
    update = Update()
    update.push("tags", "new_tag")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})

def test_push_literal_int_no_model():
    """Test push literal int without model type."""
    update = Update()
    update.push("numbers", 42)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "numbers", {"items": [42]})

def test_push_dict_item_no_model():
    """Test push a dictionary as an item without model type."""
    update = Update()
    item_dict = {"key": "value"}
    update.push("complex_items", item_dict)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "complex_items", {"items": [item_dict]})

def test_push_nested_path_literal_no_model():
    """Test push literal to a nested path string without model type."""
    update = Update()
    update.push("nested.list_field", "new_nested_item")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "nested.list_field", {"items": ["new_nested_item"]})


# --- Test `push` build result ---

def test_push_build_result_with_distinct_fields():
    """Test that push operations on distinct fields build correctly."""
    update = Update().push("tags", "tag1").push("items", {"id": 1, "name": "Item 1"})
    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, PushOperation, "tags", {"items": ["tag1"]})
    assert_operation_present(result, PushOperation, "items", {"items": [{"id": 1, "name": "Item 1"}]})


# --- Tests for `push` to deeply nested lists (Organization model) ---

def test_push_to_deeply_nested_list_str_valid():
    """Test push a string to a list in a nested object array."""
    update = Update(Organization)
    update.push("departments.0.members", "new_member_alpha")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "departments.0.members", {"items": ["new_member_alpha"]})

def test_push_to_very_deeply_nested_list_str_valid():
    """Test push a string to a very deeply nested list."""
    update = Update(Organization)
    update.push("departments.0.categories.0.items", "new_item_beta")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "departments.0.categories.0.items", {"items": ["new_item_beta"]})

def test_push_to_very_deeply_nested_list_int_valid():
    """Test push an int to another very deeply nested list."""
    update = Update(Organization)
    update.push("departments.0.categories.0.counts", 77)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "departments.0.categories.0.counts", {"items": [77]})

def test_push_to_deeply_nested_list_invalid_type_raises_value_type_error():
    """Test push with wrong type to a deeply nested List[str] raises ValueTypeError."""
    update = Update(Organization)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'departments.0.categories.0.items'"):
        update.push("departments.0.categories.0.items", 999)  # items expects strings
    assert len(update.build()) == 0

def test_push_to_deeply_nested_list_int_invalid_type_raises_value_type_error():
    """Test push with wrong type to a deeply nested List[int] raises ValueTypeError."""
    update = Update(Organization)
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'departments.0.categories.0.counts'"):
        update.push("departments.0.categories.0.counts", "not_an_int_value")  # counts expects integers
    assert len(update.build()) == 0


# --- Test `push` with DSL-like path ---

def test_update_dsl_nested_push():
    """Test push operation on nested array fields using string path."""
    update = Update(Entity).push("metadata.collections.tags", "new_dsl_tag")
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PushOperation)
    assert operation.field_path == "metadata.collections.tags"
    assert operation.items == ["new_dsl_tag"]