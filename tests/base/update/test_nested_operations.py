# tests/base/update/test_nested_operations.py

import pytest
from async_repository.base.update import (
    Update,
    SetOperation,
    PushOperation,
    PopOperation,
    PullOperation,
    UnsetOperation,
    IncrementOperation,
    MinOperation,
    MaxOperation,
    MultiplyOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.base.conftest import ( # Assuming your conftest models are here
    User,
    Organization,
    NestedTypes,
    Inner,
    Outer,
    Metadata,
    Address, # This Address should match the one provided
    assert_operation_present
)
from async_repository.base.utils import prepare_for_storage


# --- Test Set Operations ---

def test_set_on_nested_field_valid():
    """Test set on a valid nested field."""
    update = Update(User)
    update.set("metadata.key1", "new_value")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "metadata.key1", {"value": "new_value"})

def test_set_on_nested_field_different_type_valid():
    """Test set on a valid nested field with a different primitive type."""
    update = Update(User)
    update.set("metadata.key2", 42) # key2 is int
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "metadata.key2", {"value": 42})

def test_set_on_deeply_nested_field_valid():
    """Test set on a deeply nested valid field."""
    update = Update(NestedTypes)
    update.set("nested.inner.val", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "nested.inner.val", {"value": 100})

def test_set_on_nested_field_invalid_type_raises_value_type_error():
    """Test set on a nested field with an incorrect type raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):
        update.set("nested.inner.val", "not_an_int") # val expects int
    assert len(update.build()) == 0

def test_set_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test set on a non-existent nested path raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.set("metadata.non_existent_key", "value")
    assert len(update.build()) == 0

def test_set_complete_nested_object_valid():
    """Test setting a complete valid nested object."""
    update = Update(User)
    new_metadata = Metadata(key1="new_key", key2=99, flag=True)
    update.set("metadata", new_metadata)
    result = update.build()
    assert len(result) == 1
    expected_value = prepare_for_storage(new_metadata)
    assert_operation_present(result, SetOperation, "metadata", {"value": expected_value})

def test_set_complete_nested_object_invalid_type_raises_value_type_error():
    """Test setting a complete nested object with an invalid type raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.set("metadata", "not_a_metadata_object") # metadata expects Metadata instance
    assert len(update.build()) == 0


# --- Test Push Operations ---

def test_push_to_nested_list_valid_object():
    """Test push a valid object to a nested list."""
    update = Update(User)
    # CORRECTED: Changed zip_code to zipcode to match Address.__init__
    # Also, zipcode expects an int, so "12345" should be 12345.
    new_address = Address(street="123 Main St", city="Anytown", zipcode=12345)
    update.push("addresses", new_address)
    result = update.build()
    assert len(result) == 1
    expected_item = prepare_for_storage(new_address)
    assert_operation_present(result, PushOperation, "addresses", {"items": [expected_item]})

def test_push_to_deeply_nested_list_valid_primitive():
    """Test push a valid primitive to a deeply nested list."""
    update = Update(Organization)
    update.push("departments.0.members", "new_member") # members is List[str]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "departments.0.members", {"items": ["new_member"]})

def test_push_to_nested_list_invalid_type_raises_value_type_error():
    """Test push an item of invalid type to a nested list raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.push("addresses", "not_an_address_object") # addresses expects Address instance
    assert len(update.build()) == 0

def test_push_to_nested_list_non_existent_path_raises_invalid_path_error():
    """Test push to a non-existent nested list path raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.push("departments.0.non_existent_list", "value")
    assert len(update.build()) == 0

def test_push_to_nested_non_list_field_raises_invalid_path_error():
    """Test push to a nested field that is not a list raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError): # departments.0.name is str, not list
        update.push("departments.0.name", "value")
    assert len(update.build()) == 0


# --- Test Pop Operations ---

def test_pop_from_nested_list_default_position():
    """Test pop from a nested list with default position (last item)."""
    update = Update(User)
    update.pop("addresses") # Default position is 1 (last)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "addresses", {"position": 1})

def test_pop_from_nested_list_first_position():
    """Test pop from a nested list with first position."""
    update = Update(User)
    update.pop("addresses", -1) # Position -1 is first
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "addresses", {"position": -1})

def test_pop_from_deeply_nested_list_valid():
    """Test pop from a deeply nested list."""
    update = Update(Organization)
    update.pop("departments.0.members")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "departments.0.members", {"position": 1})

def test_pop_from_nested_list_invalid_position_raises_value_error():
    """Test pop from a nested list with an invalid position raises ValueError."""
    update = Update(User)
    with pytest.raises(ValueError): # Position 2 is not valid for pop (only 1 or -1)
        update.pop("addresses", 2)
    assert len(update.build()) == 0

def test_pop_from_nested_list_non_existent_path_raises_invalid_path_error():
    """Test pop from a non-existent nested list path raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pop("departments.0.non_existent_list")
    assert len(update.build()) == 0

def test_pop_from_nested_non_list_field_raises_invalid_path_error():
    """Test pop from a nested field that is not a list raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError): # departments.0.name is str, not list
        update.pop("departments.0.name")
    assert len(update.build()) == 0


# --- Test Pull Operations ---

def test_pull_from_nested_list_valid_primitive():
    """Test pull a valid primitive value from a nested list."""
    update = Update(User)
    update.pull("tags", "tag_to_remove") # tags is List[str]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "tags", {"value_or_condition": "tag_to_remove"})

def test_pull_from_deeply_nested_list_valid_primitive():
    """Test pull a valid primitive value from a deeply nested list."""
    update = Update(Organization)
    update.pull("departments.0.members", "member_to_remove")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "departments.0.members", {"value_or_condition": "member_to_remove"})

def test_pull_from_nested_list_invalid_type_raises_value_type_error():
    """Test pull an item of invalid type from a nested list raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError): # tags is List[str], 42 is int
        update.pull("tags", 42)
    assert len(update.build()) == 0

def test_pull_from_nested_list_non_existent_path_raises_invalid_path_error():
    """Test pull from a non-existent nested list path raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pull("departments.0.non_existent_list", "value")
    assert len(update.build()) == 0

def test_pull_from_nested_non_list_field_raises_invalid_path_error():
    """Test pull from a nested field that is not a list raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError): # departments.0.name is str, not list
        update.pull("departments.0.name", "value")
    assert len(update.build()) == 0


# --- Test Unset Operations ---

def test_unset_on_nested_field_valid():
    """Test unset on a valid nested field."""
    update = Update(User)
    update.unset("metadata.key1")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "metadata.key1")

def test_unset_on_deeply_nested_field_valid():
    """Test unset on a deeply nested valid field."""
    update = Update(NestedTypes)
    update.unset("nested.inner.val")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "nested.inner.val")

def test_unset_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test unset on a non-existent nested path raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.unset("metadata.non_existent_key")
    assert len(update.build()) == 0


# --- Test Increment Operations ---

def test_increment_on_nested_numeric_field_valid():
    """Test increment on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.increment("nested.inner.val", 5) # val is int
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val", {"amount": 5})

def test_increment_on_nested_non_numeric_field_raises_value_type_error():
    """Test increment on a nested non-numeric field raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError): # 'nested.inner' is an object, not numeric
        update.increment("nested.inner", 5)
    assert len(update.build()) == 0

def test_increment_on_nested_list_field_raises_value_type_error():
    """Test increment on a nested list field (not directly numeric) raises ValueTypeError."""
    update = Update(Organization)
    # 'counts' is List[int], not a single numeric field for direct increment.
    with pytest.raises(ValueTypeError):
        update.increment("departments.0.categories.0.counts", 5)
    assert len(update.build()) == 0

def test_increment_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test increment on a non-existent nested path raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.increment("nested.inner.non_existent_val", 5)
    assert len(update.build()) == 0

def test_increment_on_nested_field_invalid_amount_type_raises_type_error():
    """Test increment on a nested field with invalid amount type raises TypeError."""
    update = Update(NestedTypes)
    with pytest.raises(TypeError):
        update.increment("nested.inner.val", "not_a_number")
    assert len(update.build()) == 0


# --- Test Decrement Operations (uses IncrementOperation internally) ---

def test_decrement_on_nested_numeric_field_valid():
    """Test decrement on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.decrement("nested.inner.val", 3)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "nested.inner.val", {"amount": -3})

def test_decrement_on_nested_non_numeric_field_raises_value_type_error():
    """Test decrement on a nested non-numeric field raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):
        update.decrement("nested.inner", 3) # 'nested.inner' is an object
    assert len(update.build()) == 0

def test_decrement_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test decrement on a non-existent nested path raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.decrement("nested.inner.non_existent_val", 3)
    assert len(update.build()) == 0

def test_decrement_on_nested_field_invalid_amount_type_raises_type_error():
    """Test decrement on a nested field with invalid amount type raises TypeError."""
    update = Update(NestedTypes)
    with pytest.raises(TypeError):
        update.decrement("nested.inner.val", "not_a_number")
    assert len(update.build()) == 0


# --- Test Min Operations ---

def test_min_on_nested_numeric_field_valid():
    """Test min on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.min("nested.inner.val", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "nested.inner.val", {"value": 0})

def test_min_on_nested_non_numeric_field_raises_value_type_error():
    """Test min on a nested non-numeric field raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):
        update.min("nested.inner", 0) # 'nested.inner' is an object
    assert len(update.build()) == 0

def test_min_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test min on a non-existent nested path raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.min("nested.inner.non_existent_val", 0)
    assert len(update.build()) == 0

def test_min_on_nested_field_invalid_value_type_raises_type_error():
    """Test min on a nested field with invalid value type raises TypeError."""
    update = Update(NestedTypes)
    with pytest.raises(TypeError):
        update.min("nested.inner.val", "not_a_number")
    assert len(update.build()) == 0


# --- Test Max Operations ---

def test_max_on_nested_numeric_field_valid():
    """Test max on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.max("nested.inner.val", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "nested.inner.val", {"value": 100})

def test_max_on_nested_non_numeric_field_raises_value_type_error():
    """Test max on a nested non-numeric field raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):
        update.max("nested.inner", 100) # 'nested.inner' is an object
    assert len(update.build()) == 0

def test_max_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test max on a non-existent nested path raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.max("nested.inner.non_existent_val", 100)
    assert len(update.build()) == 0

def test_max_on_nested_field_invalid_value_type_raises_type_error():
    """Test max on a nested field with invalid value type raises TypeError."""
    update = Update(NestedTypes)
    with pytest.raises(TypeError):
        update.max("nested.inner.val", "not_a_number")
    assert len(update.build()) == 0


# --- Test Multiply Operations ---

def test_mul_on_nested_numeric_field_valid():
    """Test multiply on a valid nested numeric field."""
    update = Update(NestedTypes)
    update.mul("nested.inner.val", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "nested.inner.val", {"factor": 2})

def test_mul_on_nested_non_numeric_field_raises_value_type_error():
    """Test multiply on a nested non-numeric field raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError):
        update.mul("nested.inner", 2) # 'nested.inner' is an object
    assert len(update.build()) == 0

def test_mul_on_nested_field_non_existent_path_raises_invalid_path_error():
    """Test multiply on a non-existent nested path raises InvalidPathError."""
    update = Update(NestedTypes)
    with pytest.raises(InvalidPathError):
        update.mul("nested.inner.non_existent_val", 2)
    assert len(update.build()) == 0

def test_mul_on_nested_field_invalid_factor_type_raises_type_error():
    """Test multiply on a nested field with invalid factor type raises TypeError."""
    update = Update(NestedTypes)
    with pytest.raises(TypeError):
        update.mul("nested.inner.val", "not_a_number")
    assert len(update.build()) == 0


# --- Test Combined Operations on Distinct Nested Paths ---

def test_combined_nested_operations_on_distinct_paths_builds_correctly():
    """
    Test that a sequence of different nested operations on distinct paths
    builds the correct list.
    """
    update = Update(Organization)

    # Each operation targets a unique path within the Organization structure
    update.set("name", "New Org Name")
    update.set("departments.0.name", "R&D Updated")
    update.push("departments.0.members", "Alice")
    update.pull("departments.1.members", "Bob_to_remove_from_dept1") # Assuming departments[1] path is valid for this operation
    update.push("departments.0.categories.0.items", "Shiny New Item")
    update.pop("departments.0.categories.1.counts") # Assuming departments[0].categories[1] path is valid
    update.unset("departments.0.categories.0.name")

    result = update.build()
    assert len(result) == 7

    # Verify each operation by its properties and order
    assert isinstance(result[0], SetOperation) and result[0].field_path == "name" and result[0].value == "New Org Name"
    assert isinstance(result[1], SetOperation) and result[1].field_path == "departments.0.name" and result[1].value == "R&D Updated"
    assert isinstance(result[2], PushOperation) and result[2].field_path == "departments.0.members" and result[2].items == ["Alice"]
    assert isinstance(result[3], PullOperation) and result[3].field_path == "departments.1.members" and result[3].value_or_condition == "Bob_to_remove_from_dept1"
    assert isinstance(result[4], PushOperation) and result[4].field_path == "departments.0.categories.0.items" and result[4].items == ["Shiny New Item"]
    assert isinstance(result[5], PopOperation) and result[5].field_path == "departments.0.categories.1.counts" and result[5].position == 1 # Default pop position
    assert isinstance(result[6], UnsetOperation) and result[6].field_path == "departments.0.categories.0.name"