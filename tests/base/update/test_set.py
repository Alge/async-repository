# tests/base/update/test_set.py

import pytest
from async_repository.base.update import (
    Update,
    SetOperation,  # Import specific operation class
    InvalidPathError,  # Import specific exception
    ValueTypeError,  # Import specific exception
)
from tests.base.conftest import (
    User,
    ModelWithUnions,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
)

# Import the prepare_for_storage function to test serialized values
from async_repository.base.utils import prepare_for_storage

from tests.base.conftest import assert_operation_present, find_operations


def test_set_with_valid_types():
    """Test that valid types in set operations pass validation."""
    update = Update(User)

    update.set(update.fields.name, "John")
    update.set(update.fields.age, 30)
    update.set(update.fields.email, "john@example.com")
    update.set(update.fields.active, False)
    update.set(update.fields.tags, ["tag1", "tag2"])

    valid_metadata_dict = {"key1": "some_value", "key2": 42, "flag": True}
    update.set(update.fields.metadata, valid_metadata_dict)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 6

    assert_operation_present(result, SetOperation, "name", {"value": "John"})
    assert_operation_present(result, SetOperation, "age", {"value": 30})
    assert_operation_present(
        result, SetOperation, "email", {"value": "john@example.com"}
    )
    assert_operation_present(result, SetOperation, "active", {"value": False})
    assert_operation_present(result, SetOperation, "tags", {"value": ["tag1", "tag2"]})
    assert_operation_present(
        result, SetOperation, "metadata", {"value": valid_metadata_dict}
    )


def test_set_with_invalid_types():
    """Test that invalid types in set operations raise ValueTypeError."""
    update = Update(User)

    with pytest.raises(ValueTypeError):
        update.set(update.fields.name, 123)
    with pytest.raises(ValueTypeError):
        update.set(update.fields.age, "thirty")
    with pytest.raises(ValueTypeError):
        update.set(update.fields.active, "yes")
    with pytest.raises(ValueTypeError):
        update.set(update.fields.tags, "not-a-list")
    with pytest.raises(ValueTypeError):
        update.set(update.fields.tags, [1, 2, 3])  # Item type mismatch
    with pytest.raises(ValueTypeError):
        update.set(update.fields.metadata, ["not", "a", "dict"])
    with pytest.raises(ValueTypeError, match="missing required field 'key1'"):
        update.set(update.fields.metadata, {"key2": 123, "flag": False})
    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(
            update.fields.metadata, {"key1": "val", "key2": "123", "flag": False}
        )


def test_set_with_invalid_field():
    """Test that non-existent fields in set operations raise InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.set("non_existent_field", "value")


def test_set_with_optional_fields():
    """Test that None is accepted for Optional fields."""
    update = Update(User)

    update.set(update.fields.email, None)  # email is Optional[str]
    update.set(update.fields.email, "valid@example.com")

    with pytest.raises(ValueTypeError, match="received None but expected str"):
        update.set(update.fields.name, None)

    result = update.build()
    assert len(result) == 2
    email_ops = find_operations(result, SetOperation, "email")
    assert len(email_ops) == 2
    assert email_ops[0].value is None
    assert email_ops[-1].value == "valid@example.com"


def test_set_with_union_types():
    """Test that Union type validation works correctly."""
    update = Update(ModelWithUnions)

    update.set(update.fields.field, "string value")  # Union[str, int]
    update.set(update.fields.field, 42)  # Union[str, int]
    update.set(
        update.fields.container, ["string", 42, True]
    )  # List[Union[str, int, bool]]

    with pytest.raises(ValueTypeError):
        update.set(update.fields.field, [])
    with pytest.raises(ValueTypeError):
        update.set(update.fields.container, ["string", 42, {}])

    result = update.build()
    assert len(result) == 3
    field_ops = find_operations(result, SetOperation, "field")
    assert field_ops[-1].value == 42
    assert_operation_present(
        result, SetOperation, "container", {"value": ["string", 42, True]}
    )


def test_set_with_nested_fields():
    """Test set operations with nested field access."""
    update = Update(User)

    update.set(update.fields.metadata.key1, "value1")
    update.set(update.fields.metadata.key2, 99)
    update.set(update.fields.metadata.flag, False)

    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(update.fields.metadata.key2, "not-an-int")

    with pytest.raises(
        InvalidPathError, match="Field 'non_existent' does not exist in type Metadata"
    ):
        update.set(update.fields.metadata.non_existent, "value")

    # Test setting on non-existent parent - Expect InvalidPathError
    # Update the match pattern to reflect the actual error message
    with pytest.raises(
        InvalidPathError, match="Field 'non_existent' does not exist in type User"
    ):  # <<<<< CORRECTED MATCH
        update.set("non_existent.field", "value")

    result = update.build()
    assert len(result) == 3  # Only valid ops
    assert_operation_present(result, SetOperation, "metadata.key1", {"value": "value1"})
    assert_operation_present(result, SetOperation, "metadata.key2", {"value": 99})
    assert_operation_present(result, SetOperation, "metadata.flag", {"value": False})


def test_set_with_complex_nested_validations():
    """Test validation with complex nested structures."""
    update = Update(NestedTypes)

    # Valid operations
    update.set(update.fields.simple_list, [1, 2, 3])
    update.set(update.fields.str_list, ["a", "b", "c"])
    update.set(update.fields.dict_field, {"key": "value"})

    inner = Inner(42)
    outer = Outer(inner)
    serialized_outer = prepare_for_storage(outer)
    update.set(update.fields.nested, outer)
    update.set(update.fields.nested, {"inner": {"val": 50}})
    update.set(update.fields.nested.inner, {"val": 60})
    update.set(update.fields.nested.inner.val, 70)

    item_instance = ComplexItem("test", 42)
    serialized_item = prepare_for_storage(item_instance)
    update.set(update.fields.complex_list, [item_instance])
    list_of_dicts = [{"name": "item1", "value": 100}, {"name": "item2", "value": 200}]
    update.set(update.fields.complex_list, list_of_dicts)

    # Invalid operations - expect ValueTypeError from validator
    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(update.fields.simple_list, ["not", "integers"])
    with pytest.raises(ValueTypeError, match="expected type str"):
        update.set(update.fields.dict_field, {"key": 123})
    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(update.fields.nested.inner.val, "not_int")
    with pytest.raises(ValueTypeError, match="expected type str"):
        update.set(update.fields.complex_list, [{"name": 123, "value": 42}])
    with pytest.raises(ValueTypeError, match="missing required field 'value'"):
        update.set(update.fields.complex_list, [{"name": "incomplete"}])

    result = update.build()
    assert len(result) == 9

    assert find_operations(result, SetOperation, "simple_list")[-1].value == [1, 2, 3]
    assert find_operations(result, SetOperation, "nested.inner.val")[-1].value == 70
    assert (
        find_operations(result, SetOperation, "complex_list")[-1].value == list_of_dicts
    )


def test_set_without_model_type():
    """Test that set works without a model type (no validation)."""
    update = Update()

    update.set("any_field", "any_value")
    update.set("number", 123)
    update.set("nested.field", "nested value")
    update.set("a_list", [1, {"a": 2}])

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4
    assert_operation_present(result, SetOperation, "any_field", {"value": "any_value"})
    assert_operation_present(result, SetOperation, "number", {"value": 123})
    assert_operation_present(
        result, SetOperation, "nested.field", {"value": "nested value"}
    )
    assert_operation_present(result, SetOperation, "a_list", {"value": [1, {"a": 2}]})
