# tests/base/update/test_pull.py

import pytest
from async_repository.base.update import (
    Update,
    PullOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import (
    User,
    NestedTypes,
    Organization,
    Address,
)  # Added Category

# Import the prepare_for_storage function to test serialized values
from async_repository.base.utils import prepare_for_storage

from tests.base.conftest import assert_operation_present
from tests.conftest import Entity

def test_pull_with_type_validation():
    """Test that pull operations are type validated for literals."""
    update = Update(User)

    # Valid pull (literal string for List[str])
    update.pull("tags", "tag_to_remove")

    # Invalid pull (wrong type for list item) - Validator raises ValueTypeError
    with pytest.raises(ValueTypeError, match="Invalid value type for pull from 'tags'"):
        update.pull("tags", 123)  # tags list expects strings

    # Invalid field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pull("name", "value")  # name is str, not a list

    # Non-existent field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pull("non_existent", "value")

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1  # Only the valid operation
    assert_operation_present(
        result, PullOperation, "tags", {"value_or_condition": "tag_to_remove"}
    )


def test_pull_with_nested_fields():
    """Test pull operations with nested list fields containing objects."""
    update = Update(User)

    # Valid nested pull - pulling an Address object using its dict representation
    address_to_pull = Address("123 Main St", "Anytown", "12345")
    # prepare_for_storage converts the Address object to a dict for matching/validation
    serialized_address_dict = prepare_for_storage(address_to_pull)
    update.pull("addresses", serialized_address_dict)

    # Invalid nested pull (wrong type for list item) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for pull from 'addresses'"
    ):
        # Pulling an int from a list expecting Address-like objects
        update.pull("addresses", 123)

    # Non-existent nested field - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pull("metadata.list_field", "value")  # Metadata has no list_field

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 1  # Only the valid pull operation
    assert_operation_present(
        result,
        PullOperation,
        "addresses",
        {"value_or_condition": serialized_address_dict},
    )


def test_pull_with_complex_types():
    """Test pull operations with various complex types in lists."""
    update = Update(NestedTypes)

    # Valid pull from simple list (List[int])
    update.pull("simple_list", 42)

    # Valid pull from string list (List[str])
    update.pull("str_list", "string_to_remove")

    # Valid pull from complex list (List[ComplexItem]) using a matching dictionary
    complex_item_dict = {"name": "test", "value": 42}
    # Validation ensures the dict matches the ComplexItem structure
    update.pull("complex_list", complex_item_dict)

    # Invalid pull from simple list (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for pull from 'simple_list'"
    ):
        update.pull("simple_list", "not an int")

    # Invalid pull from string list (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError, match="Invalid value type for pull from 'str_list'"
    ):
        update.pull("str_list", 42)

    # Invalid pull from complex list (wrong structure/type) - Validator raises ValueTypeError
    # This should now raise ValueTypeError because the dict is validated against ComplexItem
    with pytest.raises(
        ValueTypeError, match="Invalid value type for pull from 'complex_list'"
    ):
        update.pull(
            "complex_list", {"name": 123, "value": "not an int"}
        )  # name should be str

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 3  # Only valid pulls
    assert_operation_present(
        result, PullOperation, "simple_list", {"value_or_condition": 42}
    )
    assert_operation_present(
        result, PullOperation, "str_list", {"value_or_condition": "string_to_remove"}
    )
    # The dict itself is stored as the condition
    assert_operation_present(
        result, PullOperation, "complex_list", {"value_or_condition": complex_item_dict}
    )


def test_pull_without_model_type():
    """Test that pull works without a model type (no validation)."""
    update = Update()  # No model type

    # These operations should work without errors
    update.pull("tags", "tag_to_remove")
    update.pull("numbers", 42)
    update.pull("complex_items", {"key": "value"})  # Criteria dict
    update.pull("nested.list", "nested item")

    # Build should succeed
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4
    assert_operation_present(
        result, PullOperation, "tags", {"value_or_condition": "tag_to_remove"}
    )
    assert_operation_present(
        result, PullOperation, "numbers", {"value_or_condition": 42}
    )
    assert_operation_present(
        result, PullOperation, "complex_items", {"value_or_condition": {"key": "value"}}
    )
    assert_operation_present(
        result, PullOperation, "nested.list", {"value_or_condition": "nested item"}
    )


def test_pull_build_result():
    """Test that pull operations build the correct agnostic operation list."""
    update = Update().pull("tags", "old_tag").pull("items", {"category": "removed"})

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 2
    assert_operation_present(
        result, PullOperation, "tags", {"value_or_condition": "old_tag"}
    )
    # The criteria dict is passed through
    assert_operation_present(
        result, PullOperation, "items", {"value_or_condition": {"category": "removed"}}
    )


def test_pull_from_nested_list():
    """Test pulling specific literal values from lists within nested objects."""
    update = Update(Organization)

    # Pull from a list within a nested object (List[str])
    update.pull("departments.0.members", "member1")

    # Pull from a deeply nested list (List[str], List[str], List[int])
    update.pull("departments.0.categories.0.items", "item1")
    update.pull("departments.0.categories.0.tags", "tag1")
    update.pull("departments.0.categories.0.counts", 2)

    # Invalid pull (wrong type) - Validator raises ValueTypeError
    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for pull from 'departments.0.categories.0.items'",
    ):
        update.pull("departments.0.categories.0.items", 123)  # items expects strings

    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for pull from 'departments.0.categories.0.counts'",
    ):
        update.pull(
            "departments.0.categories.0.counts", "not-a-number"
        )  # counts expects integers

    # Invalid field (not a list) - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pull("departments.0.name", "value")  # name is a string

    # Invalid nested path - Validator raises InvalidPathError
    with pytest.raises(InvalidPathError):
        update.pull("departments.0.categories.0.non_existent", "value")

    # Build and check the result
    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4  # Only valid pulls
    assert_operation_present(
        result,
        PullOperation,
        "departments.0.members",
        {"value_or_condition": "member1"},
    )
    assert_operation_present(
        result,
        PullOperation,
        "departments.0.categories.0.items",
        {"value_or_condition": "item1"},
    )
    assert_operation_present(
        result,
        PullOperation,
        "departments.0.categories.0.tags",
        {"value_or_condition": "tag1"},
    )
    assert_operation_present(
        result,
        PullOperation,
        "departments.0.categories.0.counts",
        {"value_or_condition": 2},
    )


def test_pull_with_nested_object_criteria():
    """Test pulling with dictionary criteria from nested lists."""
    update = Update(Organization)  # Using model for path validation

    # Pulling a category object based on a matching dict.
    # This dict WILL be validated against Category structure because it's not an operator dict.
    category_match = {"name": "Products"}
    # Expect ValueTypeError because the dict is incomplete for a Category object.
    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for pull from 'departments.0.categories'",
    ):
        update.pull("departments.0.categories", category_match)

    # Reset operations for the next check
    update._operations = []  # Clear previous (failed) operation

    # Pull using complex criteria (MongoDB-like operators) - this bypasses item validation.
    complex_criteria = {"$in": ["featured", "sale"]}
    update.pull("departments.0.categories.0.tags", complex_criteria)

    result = update.build()
    assert isinstance(result, list)
    # Only the operation with the operator dict ($in) should be present
    assert len(result) == 1

    # Check that the operator dictionary was passed through
    assert_operation_present(
        result,
        PullOperation,
        "departments.0.categories.0.tags",
        {"value_or_condition": complex_criteria},
    )

    # Test invalid literal value type still fails validation against item type
    with pytest.raises(
        ValueTypeError,
        match="Invalid value type for pull from 'departments.0.categories.0.tags'",
    ):
        update.pull(
            "departments.0.categories.0.tags", 123
        )  # tags expects strings, 123 is literal


def test_update_dsl_nested_pull():
    """Test pull operation on nested array fields."""
    # Create an update with pull operation on nested path
    update = Update(Entity).pull("metadata.collections.categories", "A")

    # Assert the operation was added correctly
    assert len(update._operations) == 1
    assert isinstance(update._operations[0], PullOperation)
    assert update._operations[0].field_path == "metadata.collections.categories"
    assert update._operations[0].value_or_condition == "A"