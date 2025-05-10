# tests/base/update/test_pull.py

import pytest
from async_repository.base.update import (
    Update,
    PullOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.base.conftest import (
    User,
    NestedTypes,
    Organization,
    Address,
    ComplexItem, # Assuming ComplexItem is defined in conftest
    assert_operation_present
)
from async_repository.base.utils import prepare_for_storage
from tests.conftest import Entity


# --- Tests for `pull` with type validation (User model) ---

def test_pull_valid_literal_with_type_validation():
    """Test pull with a valid literal string from List[str] is validated."""
    update = Update(User)
    update.pull("tags", "tag_to_remove")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "tags", {"value_or_condition": "tag_to_remove"})

def test_pull_invalid_literal_type_raises_value_type_error():
    """Test pull with wrong literal type for list item raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'tags'"):
        update.pull("tags", 123)  # tags list expects strings
    assert len(update.build()) == 0

def test_pull_from_non_list_field_raises_invalid_path_error():
    """Test pull from a non-list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pull("name", "value")  # name is str, not a list
    assert len(update.build()) == 0

def test_pull_from_non_existent_field_raises_invalid_path_error():
    """Test pull from a non-existent field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.pull("non_existent_field", "value")
    assert len(update.build()) == 0


# --- Tests for `pull` with nested fields (User model) ---

def test_pull_nested_object_with_matching_dict():
    """Test pull a nested object using its dict representation for matching."""
    update = Update(User)
    address_to_pull = Address(street="123 Main St", city="Anytown", zipcode=12345)
    serialized_address_dict = prepare_for_storage(address_to_pull)
    update.pull("addresses", serialized_address_dict)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(
        result, PullOperation, "addresses", {"value_or_condition": serialized_address_dict}
    )

def test_pull_nested_object_invalid_literal_type_raises_value_type_error():
    """Test pull from List[Address] with an int literal raises ValueTypeError."""
    update = Update(User)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'addresses'"):
        update.pull("addresses", 123) # Pulling an int from a list expecting Address-like objects
    assert len(update.build()) == 0

def test_pull_from_non_existent_nested_list_field_raises_invalid_path_error():
    """Test pull from a non-existent nested list field raises InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError): # Metadata has no 'list_field'
        update.pull("metadata.list_field", "value")
    assert len(update.build()) == 0


# --- Tests for `pull` with complex types (NestedTypes model) ---

def test_pull_from_list_of_int_valid():
    """Test pull a valid int from List[int]."""
    update = Update(NestedTypes)
    update.pull("simple_list", 42)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "simple_list", {"value_or_condition": 42})

def test_pull_from_list_of_str_valid():
    """Test pull a valid str from List[str]."""
    update = Update(NestedTypes)
    update.pull("str_list", "string_to_remove")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "str_list", {"value_or_condition": "string_to_remove"})

def test_pull_from_list_of_complex_item_with_matching_dict():
    """Test pull from List[ComplexItem] using a structurally matching dictionary."""
    update = Update(NestedTypes)
    # Assuming ComplexItem(name: str, value: int)
    complex_item_dict = {"name": "test_item", "value": 100}
    update.pull("complex_list", complex_item_dict)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "complex_list", {"value_or_condition": complex_item_dict})

def test_pull_from_list_of_int_invalid_type_raises_value_type_error():
    """Test pull from List[int] with a string literal raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'simple_list'"):
        update.pull("simple_list", "not_an_int")
    assert len(update.build()) == 0

def test_pull_from_list_of_str_invalid_type_raises_value_type_error():
    """Test pull from List[str] with an int literal raises ValueTypeError."""
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'str_list'"):
        update.pull("str_list", 12345)
    assert len(update.build()) == 0

def test_pull_from_list_of_complex_item_invalid_dict_type_raises_value_type_error():
    """Test pull from List[ComplexItem] with a dict of incorrect type raises ValueTypeError."""
    update = Update(NestedTypes)
    # ComplexItem expects name:str, value:int. Here name is int.
    invalid_complex_item_dict = {"name": 123, "value": "not_an_int"}
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'complex_list'"):
        update.pull("complex_list", invalid_complex_item_dict)
    assert len(update.build()) == 0


# --- Tests for `pull` without a model type (no validation) ---

def test_pull_literal_string_no_model():
    """Test pull literal string without model type."""
    update = Update()
    update.pull("tags", "tag_to_remove")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "tags", {"value_or_condition": "tag_to_remove"})

def test_pull_literal_int_no_model():
    """Test pull literal int without model type."""
    update = Update()
    update.pull("numbers", 42)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "numbers", {"value_or_condition": 42})

def test_pull_dict_criteria_no_model():
    """Test pull with a dictionary (as criteria or item match) without model type."""
    update = Update()
    criteria_dict = {"key": "value"}
    update.pull("complex_items", criteria_dict)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "complex_items", {"value_or_condition": criteria_dict})

def test_pull_nested_path_literal_no_model():
    """Test pull literal from a nested path string without model type."""
    update = Update()
    update.pull("nested.list_field", "nested_item_to_remove")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "nested.list_field", {"value_or_condition": "nested_item_to_remove"})


# --- Test `pull` build result ---

def test_pull_build_result_with_distinct_fields():
    """Test that pull operations on distinct fields build correctly."""
    update = Update().pull("tags", "old_tag").pull("items", {"category": "removed"})
    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, PullOperation, "tags", {"value_or_condition": "old_tag"})
    assert_operation_present(result, PullOperation, "items", {"value_or_condition": {"category": "removed"}})


# --- Tests for `pull` from deeply nested lists (Organization model) ---

def test_pull_from_deeply_nested_list_str_valid():
    """Test pull a string from a list in a nested object array."""
    update = Update(Organization)
    update.pull("departments.0.members", "member1")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "departments.0.members", {"value_or_condition": "member1"})

def test_pull_from_very_deeply_nested_list_str_valid():
    """Test pull a string from a very deeply nested list."""
    update = Update(Organization)
    update.pull("departments.0.categories.0.items", "item1")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "departments.0.categories.0.items", {"value_or_condition": "item1"})

def test_pull_from_very_deeply_nested_list_int_valid():
    """Test pull an int from another very deeply nested list."""
    update = Update(Organization)
    update.pull("departments.0.categories.0.counts", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "departments.0.categories.0.counts", {"value_or_condition": 2})

def test_pull_from_deeply_nested_list_invalid_type_raises_value_type_error():
    """Test pull with wrong type from a deeply nested List[str] raises ValueTypeError."""
    update = Update(Organization)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'departments.0.categories.0.items'"):
        update.pull("departments.0.categories.0.items", 123)  # items expects strings
    assert len(update.build()) == 0

def test_pull_from_deeply_nested_list_int_invalid_type_raises_value_type_error():
    """Test pull with wrong type from a deeply nested List[int] raises ValueTypeError."""
    update = Update(Organization)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'departments.0.categories.0.counts'"):
        update.pull("departments.0.categories.0.counts", "not_a_number")  # counts expects integers
    assert len(update.build()) == 0

def test_pull_from_deeply_nested_non_list_field_raises_invalid_path_error():
    """Test pull from a deeply nested non-list field raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pull("departments.0.name", "value")  # name is a string
    assert len(update.build()) == 0

def test_pull_from_non_existent_deeply_nested_path_raises_invalid_path_error():
    """Test pull from a non-existent deeply nested path raises InvalidPathError."""
    update = Update(Organization)
    with pytest.raises(InvalidPathError):
        update.pull("departments.0.categories.0.non_existent_list", "value")
    assert len(update.build()) == 0


# --- Tests for `pull` with dictionary criteria (operator vs. item match) ---

def test_pull_with_nested_object_criteria_incomplete_dict_raises_value_type_error():
    """
    Test pulling from List[Category] with an incomplete dict (not an operator dict)
    raises ValueTypeError due to failing item structure validation.
    """
    update = Update(Organization)
    # This dict is incomplete for a Category object (e.g., missing 'items', 'tags', 'counts').
    # Because it's not an operator dict (like {"$in": ...}), it's treated as a potential item match.
    incomplete_category_match = {"name": "Products"}
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'departments.0.categories'"):
        update.pull("departments.0.categories", incomplete_category_match)
    assert len(update.build()) == 0

def test_pull_with_nested_object_operator_criteria_bypasses_item_validation():
    """
    Test pulling using complex criteria (MongoDB-like operators) bypasses item structure validation
    for the criteria dict itself, but not for literal values if mixed.
    """
    update = Update(Organization)
    # This is an operator dict, so it's passed through as the condition.
    # The validation for the *elements* of 'tags' (which are str) is still active
    # if a literal were passed instead of an operator dict.
    complex_criteria = {"$in": ["featured_tag", "sale_tag"]}
    update.pull("departments.0.categories.0.tags", complex_criteria)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(
        result, PullOperation, "departments.0.categories.0.tags", {"value_or_condition": complex_criteria}
    )

def test_pull_with_operator_criteria_then_invalid_literal_type_still_validates_literal():
    """
    Test that after a pull with operator criteria, a subsequent pull with an invalid literal
    on the same field (which would be rejected due to field conflict if not for this setup)
    still correctly validates the literal against the item type.
    This test focuses on the validation of the literal if it were allowed.
    """
    # First, a valid pull with operator criteria.
    # This is more about setting up a scenario to then test a specific validation logic.
    # In practice, the second pull would raise a field conflict error.
    # We are testing the ValueTypeError path here if that conflict wasn't there.
    update_setup = Update(Organization)
    complex_criteria = {"$in": ["tagA", "tagB"]}
    update_setup.pull("departments.0.categories.0.tags", complex_criteria)
    # update_setup.build() # This would normally be built.

    # Now, on a new Update (or if we could bypass field conflict for this specific check),
    # test that pulling an invalid literal type for 'tags' still raises ValueTypeError.
    update_for_literal_check = Update(Organization)
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'departments.0.categories.0.tags'"):
        # This part tests the ValueTypeError if the field 'departments.0.categories.0.tags'
        # was available for a new operation.
        update_for_literal_check.pull("departments.0.categories.0.tags", 123) # tags expects strings
    assert len(update_for_literal_check.build()) == 0


# --- Test `pull` with DSL-like path ---

def test_update_dsl_nested_pull():
    """Test pull operation on nested array fields using string path."""
    update = Update(Entity).pull("metadata.collections.categories", "CategoryA")
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PullOperation)
    assert operation.field_path == "metadata.collections.categories"
    assert operation.value_or_condition == "CategoryA"