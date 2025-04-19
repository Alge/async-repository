import pytest
from async_repository.base.update import Update
from .conftest import (
    User,
    ModelWithUnions,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
    Metadata,
)
from async_repository.base.model_validator import (
    ValueTypeError,
)  # Import custom exception


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
    assert "$set" in result
    assert len(result["$set"]) == 6
    assert result["$set"]["metadata"] == valid_metadata_dict


def test_set_with_invalid_types():
    """Test that invalid types in set operations raise TypeError or ValueTypeError."""
    update = Update(User)

    # Our implementation wraps ValueTypeError as TypeError
    with pytest.raises(TypeError):
        update.set(update.fields.name, 123)

    with pytest.raises(TypeError):
        update.set(update.fields.age, "thirty")

    with pytest.raises(TypeError):
        update.set(update.fields.active, "yes")

    with pytest.raises(TypeError):
        update.set(update.fields.tags, "not-a-list")

    with pytest.raises(TypeError):
        update.set(update.fields.tags, [1, 2, 3])

    with pytest.raises(TypeError):
        update.set(update.fields.metadata, ["not", "a", "dict"])

    # Test setting metadata with a dict missing required fields
    # This validation happens within the recursive check for dicts matching classes
    with pytest.raises(
        TypeError, match="missing required field 'key1'"
    ):
        update.set(update.fields.metadata, {"key2": 123, "flag": False})

    # Test setting metadata with a dict having wrong types for fields
    with pytest.raises(
        TypeError, match="expected type int"
    ):
        update.set(update.fields.metadata, {"key1": "val", "key2": "123", "flag": False})


def test_set_with_invalid_field():
    """Test that non-existent fields in set operations raise InvalidPathError."""
    update = Update(User)

    with pytest.raises(TypeError):
        update.set("non_existent_field", "value")


def test_set_with_optional_fields():
    """Test that None is accepted for Optional fields."""
    update = Update(User)

    update.set(update.fields.email, None)
    update.set(update.fields.email, "valid@example.com")

    # Setting a required field to None should raise TypeError
    with pytest.raises(TypeError):
        update.set(update.fields.name, None)


def test_set_with_union_types():
    """Test that Union type validation works correctly."""
    update = Update(ModelWithUnions)

    update.set(update.fields.field, "string value")
    update.set(update.fields.field, 42)
    update.set(update.fields.container, ["string", 42, True])

    # Check that incompatible types raise TypeError
    with pytest.raises(TypeError):
        update.set(update.fields.field, [])

    with pytest.raises(TypeError):
        update.set(update.fields.container, ["string", 42, {}])


def test_set_with_nested_fields():
    """Test set operations with nested field access."""
    update = Update(User)

    update.set(update.fields.metadata.key1, "value1")
    update.set(update.fields.metadata.key2, 99)
    update.set(update.fields.metadata.flag, False)

    # Test invalid nested field type
    with pytest.raises(
        TypeError, match="expected type int"
    ):
        update.set(update.fields.metadata.key2, "not-an-int")

    # Test invalid nested field path
    with pytest.raises(
        TypeError, match="does not exist in type Metadata"
    ):
        update.set(update.fields.metadata.non_existent, "value")

    # Test setting on non-existent parent
    with pytest.raises(
        TypeError, match="Field 'non_existent' does not exist"
    ):
        update.set("non_existent.field", "value")


def test_set_with_complex_nested_validations():
    """Test validation with complex nested structures."""
    update = Update(NestedTypes)

    update.set(update.fields.simple_list, [1, 2, 3])
    update.set(update.fields.str_list, ["a", "b", "c"])
    update.set(update.fields.dict_field, {"key": "value"})

    inner = Inner(42)
    outer = Outer(inner)
    update.set(update.fields.nested, outer)
    update.set(update.fields.nested, {"inner": {"val": 50}})
    update.set(update.fields.nested.inner, {"val": 60})
    update.set(update.fields.nested.inner.val, 70)

    item_instance = ComplexItem("test", 42)
    update.set(update.fields.complex_list, [item_instance])
    update.set(
        update.fields.complex_list,
        [{"name": "item1", "value": 100}, {"name": "item2", "value": 200}],
    )

    # Invalid operations - expect TypeError now
    with pytest.raises(
        TypeError, match="expected type int"
    ):
        update.set(update.fields.simple_list, ["not", "integers"])

    with pytest.raises(
        TypeError, match="expected type str"
    ):
        update.set(update.fields.dict_field, {"key": 123})

    with pytest.raises(
        TypeError, match="expected type int"
    ):
        update.set(update.fields.nested.inner.val, "not_int")

    with pytest.raises(
        TypeError, match="expected type str"
    ):
        update.set(update.fields.complex_list, [{"name": 123, "value": 42}])

    with pytest.raises(
        TypeError, match="missing required field 'value'"
    ):
        update.set(update.fields.complex_list, [{"name": "incomplete"}])


def test_set_without_model_type():
    """Test that set works without a model type (no validation)."""
    update = Update()

    update.set("any_field", "any_value")
    update.set("number", 123)
    update.set("nested.field", "nested value")

    result = update.build()
    assert "$set" in result
    assert result["$set"]["any_field"] == "any_value"
    assert result["$set"]["number"] == 123
    assert result["$set"]["nested.field"] == "nested value"