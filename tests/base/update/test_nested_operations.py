# tests/base/update/test_nested_operations.py

import pytest
from typing import List, Type, Optional, TypeVar # Added for helper
from async_repository.base.update import (
    Update,
    UpdateOperation,
    SetOperation,
    PushOperation,
    PopOperation,
    PullOperation,
    UnsetOperation,
    IncrementOperation,
    # Decrement is handled via IncrementOperation
    MinOperation,
    MaxOperation,
    MultiplyOperation,
    InvalidPathError,
    ValueTypeError,
)
from .conftest import (
    User,
    Organization,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
    Metadata,
    Address,
)
# Import the prepare_for_storage function to test serialized values
from async_repository.base.utils import prepare_for_storage


# --- Test Helper (can be defined here or imported from a common place) ---
OpT = TypeVar('OpT', bound=UpdateOperation)

def find_operation(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str
) -> Optional[OpT]:
    """Finds the first operation of a specific type and field path."""
    for op in operations:
        if isinstance(op, op_type) and op.field_path == field_path:
            return op
    return None

def assert_operation_present(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str,
    expected_attrs: Optional[dict] = None # Check specific attributes like value, amount
):
    """Asserts that a specific operation exists and optionally checks its attributes."""
    op = find_operation(operations, op_type, field_path)
    assert op is not None, f"{op_type.__name__} for field '{field_path}' not found in {operations}"
    if expected_attrs:
        for attr, expected_value in expected_attrs.items():
            assert hasattr(op, attr), f"Operation {op!r} missing attribute '{attr}'"
            actual_value = getattr(op, attr)
            # Use pytest.approx for floats if needed
            if isinstance(expected_value, float):
                 import pytest
                 assert actual_value == pytest.approx(expected_value), \
                     f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
            else:
                 assert actual_value == expected_value, \
                    f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"

def find_operations(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: Optional[str] = None # Optional field path filtering
) -> List[OpT]:
    """Finds all operations of a specific type, optionally filtered by field path."""
    found = []
    for op in operations:
        if isinstance(op, op_type):
             if field_path is None or op.field_path == field_path:
                found.append(op)
    return found
# --- End Test Helper ---


class TestNestedOperations:
    """
    Comprehensive tests for all operations on nested objects with type checking.
    """

    def test_set_on_nested_objects(self):
        """Test set operations on various nested object structures."""
        update = Update(User)
        update.set("metadata.key1", "new_value")
        update.set("metadata.key2", 42)
        update.set("metadata.flag", True)

        result_user = update.build()
        assert len(result_user) == 3
        assert_operation_present(result_user, SetOperation, "metadata.key1", {"value": "new_value"})
        assert_operation_present(result_user, SetOperation, "metadata.key2", {"value": 42})
        assert_operation_present(result_user, SetOperation, "metadata.flag", {"value": True})

        update_nested = Update(NestedTypes)
        update_nested.set("nested.inner.val", 100)
        result_nested = update_nested.build()
        assert len(result_nested) == 1
        assert_operation_present(result_nested, SetOperation, "nested.inner.val", {"value": 100})

        # Test incorrect types
        with pytest.raises(ValueTypeError): # Validator raises ValueTypeError
            update_nested.set("nested.inner.val", "not_an_int")

        with pytest.raises(ValueTypeError): # Validator raises ValueTypeError
            Update(User).set("metadata.key2", "not_an_int")

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(User).set("metadata.non_existent", "value")

        with pytest.raises(InvalidPathError):
            Update(NestedTypes).set("nested.inner.non_existent", 42)

        # Test setting complete nested objects
        metadata = Metadata("new_key1", 99, False)
        update_user2 = Update(User)
        update_user2.set("metadata", metadata)
        result_user2 = update_user2.build()
        # Assert the serialized version is stored
        expected_metadata = prepare_for_storage(metadata)
        assert_operation_present(result_user2, SetOperation, "metadata", {"value": expected_metadata})

        inner = Inner(500)
        outer = Outer(inner)
        update_nested2 = Update(NestedTypes)
        update_nested2.set("nested", outer)
        result_nested2 = update_nested2.build()
        expected_outer = prepare_for_storage(outer)
        assert_operation_present(result_nested2, SetOperation, "nested", {"value": expected_outer})

        # Test setting complete nested objects with invalid types
        with pytest.raises(ValueTypeError): # Validator raises ValueTypeError
            Update(User).set("metadata", "not_a_metadata")

        with pytest.raises(ValueTypeError): # Validator raises ValueTypeError
            Update(NestedTypes).set("nested", "not_an_outer")

    def test_push_on_nested_lists(self):
        """Test push operations on nested lists."""
        update_user = Update(User)
        address = Address("123 Main St", "Anytown", "12345")
        update_user.push("addresses", address)
        result_user = update_user.build()
        expected_address = prepare_for_storage(address)
        # PushOperation stores a list of items
        assert_operation_present(result_user, PushOperation, "addresses", {"items": [expected_address]})

        update_org = Update(Organization)
        update_org.push("departments.0.members", "new_member")
        update_org.push("departments.0.categories.0.items", "new_item")
        update_org.push("departments.0.categories.0.counts", 99)
        result_org = update_org.build()
        assert len(result_org) == 3
        assert_operation_present(result_org, PushOperation, "departments.0.members", {"items": ["new_member"]})
        assert_operation_present(result_org, PushOperation, "departments.0.categories.0.items", {"items": ["new_item"]})
        assert_operation_present(result_org, PushOperation, "departments.0.categories.0.counts", {"items": [99]})

        # Test incorrect types - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).push("addresses", "not_an_address")

        with pytest.raises(ValueTypeError):
            Update(Organization).push("departments.0.categories.0.items", 42) # Expects str

        with pytest.raises(ValueTypeError):
            Update(Organization).push("departments.0.categories.0.counts", "not_an_int") # Expects int

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).push("departments.0.non_existent", "value")

        # Test pushing to non-list fields - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).push("departments.0.name", "value")

    def test_pop_on_nested_lists(self):
        """Test pop operations on nested lists."""
        update_user = Update(User)
        update_user.pop("addresses")      # Default (last) position=1
        update_user.pop("addresses", -1)  # First position=-1
        result_user = update_user.build()
        # Last operation for the same field wins if not checking intermediate states
        assert len(result_user) == 2 # Two distinct operations added
        pop_ops_user = find_operations(result_user, PopOperation, "addresses")
        assert len(pop_ops_user) == 2
        assert pop_ops_user[0].position == 1
        assert pop_ops_user[1].position == -1

        update_org = Update(Organization)
        update_org.pop("departments.0.members")                # pos=1
        update_org.pop("departments.0.categories.0.items", 1)  # pos=1
        update_org.pop("departments.0.categories.0.counts", -1)# pos=-1
        result_org = update_org.build()
        assert len(result_org) == 3
        assert_operation_present(result_org, PopOperation, "departments.0.members", {"position": 1})
        assert_operation_present(result_org, PopOperation, "departments.0.categories.0.items", {"position": 1})
        assert_operation_present(result_org, PopOperation, "departments.0.categories.0.counts", {"position": -1})

        # Test invalid direction - Method raises ValueError
        with pytest.raises(ValueError):
            Update(User).pop("addresses", 2)

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).pop("departments.0.non_existent")

        # Test popping from non-list fields - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).pop("departments.0.name")

    def test_pull_on_nested_lists(self):
        """Test pull operations on nested lists."""
        update_user = Update(User)
        update_user.pull("tags", "tag_to_remove")
        result_user = update_user.build()
        assert_operation_present(result_user, PullOperation, "tags", {"value_or_condition": "tag_to_remove"})

        update_org = Update(Organization)
        update_org.pull("departments.0.members", "member1")
        update_org.pull("departments.0.categories.0.items", "item1")
        update_org.pull("departments.0.categories.0.counts", 2)
        result_org = update_org.build()
        assert len(result_org) == 3
        assert_operation_present(result_org, PullOperation, "departments.0.members", {"value_or_condition": "member1"})
        assert_operation_present(result_org, PullOperation, "departments.0.categories.0.items", {"value_or_condition": "item1"})
        assert_operation_present(result_org, PullOperation, "departments.0.categories.0.counts", {"value_or_condition": 2})

        # Test incorrect types - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).pull("tags", 42) # Expects str

        with pytest.raises(ValueTypeError):
            Update(Organization).pull("departments.0.categories.0.items", 42) # Expects str

        with pytest.raises(ValueTypeError):
            Update(Organization).pull("departments.0.categories.0.counts", "not_an_int") # Expects int

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).pull("departments.0.non_existent", "value")

        # Test pulling from non-list fields - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(Organization).pull("departments.0.name", "value")

    def test_unset_on_nested_fields(self):
        """Test unset operations on nested fields."""
        update_user = Update(User)
        update_user.unset("metadata.key1")
        update_user.unset("metadata.key2")
        update_user.unset("metadata.flag")
        result_user = update_user.build()
        assert len(result_user) == 3
        assert_operation_present(result_user, UnsetOperation, "metadata.key1")
        assert_operation_present(result_user, UnsetOperation, "metadata.key2")
        assert_operation_present(result_user, UnsetOperation, "metadata.flag")

        update_nested = Update(NestedTypes)
        update_nested.unset("nested.inner.val")
        result_nested = update_nested.build()
        assert len(result_nested) == 1
        assert_operation_present(result_nested, UnsetOperation, "nested.inner.val")

        update_org = Update(Organization)
        update_org.unset("departments.0.name")
        update_org.unset("departments.0.categories.0.name")
        result_org = update_org.build()
        assert len(result_org) == 2
        assert_operation_present(result_org, UnsetOperation, "departments.0.name")
        assert_operation_present(result_org, UnsetOperation, "departments.0.categories.0.name")

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(User).unset("metadata.non_existent")

        with pytest.raises(InvalidPathError):
            Update(NestedTypes).unset("nested.inner.non_existent")

    def test_increment_on_nested_fields(self):
        """Test increment operations on nested numeric fields."""
        update_nested = Update(NestedTypes)
        update_nested.increment("nested.inner.val", 5)
        update_nested.increment("counter", 10)
        result_nested = update_nested.build()
        assert len(result_nested) == 2
        assert_operation_present(result_nested, IncrementOperation, "nested.inner.val", {"amount": 5})
        assert_operation_present(result_nested, IncrementOperation, "counter", {"amount": 10})

        # Test Organization.departments[0].categories[0].counts directly (not valid for increment)
        # Counts is a list field, not a numeric field for direct increment
        with pytest.raises(ValueTypeError): # Validator raises ValueTypeError
            Update(Organization).increment("departments.0.categories.0.counts", 5)

        # Test non-numeric fields - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).increment("name", 5)

        with pytest.raises(ValueTypeError):
            Update(NestedTypes).increment("nested.inner", 5) # inner is an object

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(NestedTypes).increment("nested.inner.non_existent", 5)

        # Test invalid increment amount type - Method raises TypeError
        with pytest.raises(TypeError):
            Update(NestedTypes).increment("nested.inner.val", "not_a_number")

    def test_decrement_on_nested_fields(self):
        """Test decrement operations on nested numeric fields."""
        update_nested = Update(NestedTypes)
        update_nested.decrement("nested.inner.val", 5)
        update_nested.decrement("counter", 10)
        result_nested = update_nested.build()
        assert len(result_nested) == 2
        # Decrement adds an IncrementOperation with negative amount
        assert_operation_present(result_nested, IncrementOperation, "nested.inner.val", {"amount": -5})
        assert_operation_present(result_nested, IncrementOperation, "counter", {"amount": -10})

        # Test non-numeric fields - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).decrement("name", 5)

        with pytest.raises(ValueTypeError):
            Update(NestedTypes).decrement("nested.inner", 5) # inner is an object

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(NestedTypes).decrement("nested.inner.non_existent", 5)

        # Test invalid decrement amount type - Method raises TypeError
        with pytest.raises(TypeError):
            Update(NestedTypes).decrement("nested.inner.val", "not_a_number")

    def test_min_on_nested_fields(self):
        """Test min operations on nested numeric fields."""
        update_nested = Update(NestedTypes)
        update_nested.min("nested.inner.val", 5)
        update_nested.min("counter", 0)
        result_nested = update_nested.build()
        assert len(result_nested) == 2
        assert_operation_present(result_nested, MinOperation, "nested.inner.val", {"value": 5})
        assert_operation_present(result_nested, MinOperation, "counter", {"value": 0})

        # Test non-numeric fields - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).min("name", 5)

        with pytest.raises(ValueTypeError):
            Update(NestedTypes).min("nested.inner", 5) # inner is an object

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(NestedTypes).min("nested.inner.non_existent", 5)

        # Test invalid min value type - Method raises TypeError
        with pytest.raises(TypeError):
            Update(NestedTypes).min("nested.inner.val", "not_a_number")

    def test_max_on_nested_fields(self):
        """Test max operations on nested numeric fields."""
        update_nested = Update(NestedTypes)
        update_nested.max("nested.inner.val", 100)
        update_nested.max("counter", 1000)
        result_nested = update_nested.build()
        assert len(result_nested) == 2
        assert_operation_present(result_nested, MaxOperation, "nested.inner.val", {"value": 100})
        assert_operation_present(result_nested, MaxOperation, "counter", {"value": 1000})

        # Test non-numeric fields - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).max("name", 100)

        with pytest.raises(ValueTypeError):
            Update(NestedTypes).max("nested.inner", 100) # inner is an object

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(NestedTypes).max("nested.inner.non_existent", 100)

        # Test invalid max value type - Method raises TypeError
        with pytest.raises(TypeError):
            Update(NestedTypes).max("nested.inner.val", "not_a_number")

    def test_mul_on_nested_fields(self):
        """Test mul operations on nested numeric fields."""
        update_nested = Update(NestedTypes)
        update_nested.mul("nested.inner.val", 2)
        update_nested.mul("counter", 1.5)
        result_nested = update_nested.build()
        assert len(result_nested) == 2
        assert_operation_present(result_nested, MultiplyOperation, "nested.inner.val", {"factor": 2})
        assert_operation_present(result_nested, MultiplyOperation, "counter", {"factor": 1.5})

        # Test non-numeric fields - Validator raises ValueTypeError
        with pytest.raises(ValueTypeError):
            Update(User).mul("name", 2)

        with pytest.raises(ValueTypeError):
            Update(NestedTypes).mul("nested.inner", 2) # inner is an object

        # Test non-existent nested paths - Validator raises InvalidPathError
        with pytest.raises(InvalidPathError):
            Update(NestedTypes).mul("nested.inner.non_existent", 2)

        # Test invalid mul factor type - Method raises TypeError
        with pytest.raises(TypeError):
            Update(NestedTypes).mul("nested.inner.val", "not_a_number")

    def test_combined_nested_operations(self):
        """Test combined operations on nested structures builds correct list."""
        update = Update(Organization)

        update.set("name", "New Organization")                       # op 0
        update.set("departments.0.name", "Updated Department")       # op 1
        update.push("departments.0.members", "new_member")           # op 2
        update.pull("departments.0.members", "member1")              # op 3
        update.push("departments.0.categories.0.items", "new_item")  # op 4
        update.pop("departments.0.categories.0.counts")              # op 5
        update.unset("departments.0.categories.0.name")              # op 6

        # Verify the resulting update operation list
        result = update.build()
        assert isinstance(result, list)
        assert len(result) == 7

        assert isinstance(result[0], SetOperation) and result[0].field_path == "name" and result[0].value == "New Organization"
        assert isinstance(result[1], SetOperation) and result[1].field_path == "departments.0.name" and result[1].value == "Updated Department"
        assert isinstance(result[2], PushOperation) and result[2].field_path == "departments.0.members" and result[2].items == ["new_member"]
        assert isinstance(result[3], PullOperation) and result[3].field_path == "departments.0.members" and result[3].value_or_condition == "member1"
        assert isinstance(result[4], PushOperation) and result[4].field_path == "departments.0.categories.0.items" and result[4].items == ["new_item"]
        assert isinstance(result[5], PopOperation) and result[5].field_path == "departments.0.categories.0.counts" and result[5].position == 1 # default pop position is 1
        assert isinstance(result[6], UnsetOperation) and result[6].field_path == "departments.0.categories.0.name"

        # Can also use the helper for specific checks
        assert_operation_present(result, SetOperation, "name", {"value": "New Organization"})
        assert_operation_present(result, PushOperation, "departments.0.members", {"items": ["new_member"]})
        assert_operation_present(result, PopOperation, "departments.0.categories.0.counts", {"position": 1})
        assert_operation_present(result, UnsetOperation, "departments.0.categories.0.name")