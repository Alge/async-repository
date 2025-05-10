# tests/base/update/test_set.py

import pytest
from async_repository.base.update import (
    Update,
    SetOperation,
    InvalidPathError,
    ValueTypeError,
)
from tests.base.conftest import (
    User,
    ModelWithUnions,
    NestedTypes,
    Inner,
    Outer,
    ComplexItem,
    Metadata, # Assuming Metadata is in conftest
    assert_operation_present
)
from async_repository.base.utils import prepare_for_storage
from tests.conftest import Entity


# --- Tests for `set` with valid types (User model) ---

def test_set_valid_string_field():
    update = Update(User)
    update.set(update.fields.name, "John")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "name", {"value": "John"})

def test_set_valid_int_field():
    update = Update(User)
    update.set(update.fields.age, 30)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "age", {"value": 30})

def test_set_valid_optional_string_field_with_value():
    update = Update(User)
    update.set(update.fields.email, "john@example.com")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "email", {"value": "john@example.com"})

def test_set_valid_bool_field():
    update = Update(User)
    update.set(update.fields.active, False)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "active", {"value": False})

def test_set_valid_list_str_field():
    update = Update(User)
    update.set(update.fields.tags, ["tag1", "tag2"])
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "tags", {"value": ["tag1", "tag2"]})

def test_set_valid_nested_object_field_with_dict():
    update = Update(User)
    # User.metadata is Type[Metadata]
    valid_metadata_dict = {"key1": "some_value", "key2": 42, "flag": True}
    update.set(update.fields.metadata, valid_metadata_dict)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "metadata", {"value": valid_metadata_dict})

def test_set_valid_nested_object_field_with_instance():
    update = Update(User)
    metadata_instance = Metadata(key1="instance_key", key2=99, flag=False)
    serialized_metadata = prepare_for_storage(metadata_instance)
    update.set(update.fields.metadata, metadata_instance)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "metadata", {"value": serialized_metadata})


# --- Tests for `set` with invalid types (User model) ---

def test_set_invalid_type_for_string_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.set(update.fields.name, 123) # name expects str
    assert len(update.build()) == 0

def test_set_invalid_type_for_int_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.set(update.fields.age, "thirty") # age expects int
    assert len(update.build()) == 0

def test_set_invalid_type_for_bool_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.set(update.fields.active, "yes") # active expects bool
    assert len(update.build()) == 0

def test_set_invalid_type_for_list_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError):
        update.set(update.fields.tags, "not-a-list") # tags expects List
    assert len(update.build()) == 0

def test_set_invalid_item_type_for_list_str_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError): # tags expects List[str]
        update.set(update.fields.tags, [1, 2, 3]) # items are int
    assert len(update.build()) == 0

def test_set_invalid_type_for_nested_object_field_raises_error():
    update = Update(User)
    with pytest.raises(ValueTypeError): # metadata expects Metadata or dict matching Metadata
        update.set(update.fields.metadata, ["not", "a", "dict", "or", "Metadata"])
    assert len(update.build()) == 0

def test_set_incomplete_dict_for_nested_object_raises_error():
    update = Update(User)
    # metadata (Type[Metadata]) expects key1, key2, flag
    with pytest.raises(ValueTypeError, match="missing required field 'key1'"):
        update.set(update.fields.metadata, {"key2": 123, "flag": False})
    assert len(update.build()) == 0

def test_set_wrong_sub_type_in_dict_for_nested_object_raises_error():
    update = Update(User)
    # metadata.key2 expects int
    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(update.fields.metadata, {"key1": "val", "key2": "123_str", "flag": False})
    assert len(update.build()) == 0


# --- Test for `set` with invalid field path ---

def test_set_with_invalid_field_path_raises_error():
    """Test that non-existent fields in set operations raise InvalidPathError."""
    update = Update(User)
    with pytest.raises(InvalidPathError):
        update.set("non_existent_field", "value")
    assert len(update.build()) == 0


# --- Tests for `set` with Optional fields (User model) ---

def test_set_optional_field_to_none_is_valid():
    update = Update(User)
    update.set(update.fields.email, None)  # email is Optional[str]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "email", {"value": None})

def test_set_optional_field_to_valid_value_is_valid():
    update = Update(User)
    update.set(update.fields.email, "valid@example.com") # email is Optional[str]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "email", {"value": "valid@example.com"})

def test_set_non_optional_field_to_none_raises_error():
    update = Update(User)
    # name is str (not Optional[str])
    with pytest.raises(ValueTypeError, match="received None but expected str"):
        update.set(update.fields.name, None)
    assert len(update.build()) == 0


# --- Tests for `set` with Union types (ModelWithUnions) ---

def test_set_union_field_to_first_type_valid():
    update = Update(ModelWithUnions)
    update.set(update.fields.field, "string value")  # field is Union[str, int]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "field", {"value": "string value"})

def test_set_union_field_to_second_type_valid():
    update = Update(ModelWithUnions)
    update.set(update.fields.field, 42)  # field is Union[str, int]
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "field", {"value": 42})

def test_set_list_of_union_field_valid():
    update = Update(ModelWithUnions)
    # container is List[Union[str, int, bool]]
    update.set(update.fields.container, ["string", 42, True])
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "container", {"value": ["string", 42, True]})

def test_set_union_field_to_invalid_type_raises_error():
    update = Update(ModelWithUnions)
    with pytest.raises(ValueTypeError): # field is Union[str, int], [] is not valid
        update.set(update.fields.field, [])
    assert len(update.build()) == 0

def test_set_list_of_union_with_invalid_item_type_raises_error():
    update = Update(ModelWithUnions)
    # container is List[Union[str, int, bool]], {} is not valid for item
    with pytest.raises(ValueTypeError):
        update.set(update.fields.container, ["string", 42, {}])
    assert len(update.build()) == 0


# --- Tests for `set` with nested fields (User and NestedTypes models) ---

def test_set_nested_sub_field_valid():
    update = Update(User)
    update.set(update.fields.metadata.key1, "value1_updated")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "metadata.key1", {"value": "value1_updated"})

def test_set_nested_sub_field_invalid_type_raises_error():
    update = Update(User)
    # metadata.key2 expects int
    with pytest.raises(ValueTypeError, match="expected type int"):
        update.set(update.fields.metadata.key2, "not_an_int")
    assert len(update.build()) == 0

def test_set_nested_sub_field_non_existent_raises_error():
    update = Update(User)
    with pytest.raises(InvalidPathError, match="Field 'non_existent_sub_key' does not exist in type Metadata"):
        update.set(update.fields.metadata.non_existent_sub_key, "value")
    assert len(update.build()) == 0

def test_set_nested_on_non_existent_parent_raises_error():
    update = Update(User)
    with pytest.raises(InvalidPathError, match="Field 'non_existent_parent' does not exist in type User"):
        update.set("non_existent_parent.field", "value")
    assert len(update.build()) == 0


# --- Tests for `set` with complex nested structures and validations (NestedTypes model) ---

def test_set_nested_simple_list_valid():
    update = Update(NestedTypes)
    update.set(update.fields.simple_list, [10, 20, 30])
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "simple_list", {"value": [10, 20, 30]})

def test_set_nested_dict_field_valid():
    update = Update(NestedTypes)
    update.set(update.fields.dict_field, {"new_key": "new_value"})
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "dict_field", {"value": {"new_key": "new_value"}})

def test_set_nested_object_with_instance_valid():
    update = Update(NestedTypes)
    inner_instance = Inner(val=420)
    outer_instance = Outer(inner=inner_instance)
    serialized_outer = prepare_for_storage(outer_instance)
    update.set(update.fields.nested, outer_instance)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "nested", {"value": serialized_outer})

def test_set_nested_object_with_dict_valid():
    update = Update(NestedTypes)
    update.set(update.fields.nested, {"inner": {"val": 500}}) # NestedTypes.nested is Outer
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "nested", {"value": {"inner": {"val": 500}}})

def test_set_deeply_nested_object_sub_field_with_dict_valid():
    update = Update(NestedTypes)
    update.set(update.fields.nested.inner, {"val": 600}) # NestedTypes.nested.inner is Inner
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "nested.inner", {"value": {"val": 600}})

def test_set_very_deeply_nested_primitive_valid():
    update = Update(NestedTypes)
    update.set(update.fields.nested.inner.val, 700) # NestedTypes.nested.inner.val is int
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "nested.inner.val", {"value": 700})

def test_set_nested_list_of_complex_items_with_instances_valid():
    update = Update(NestedTypes)
    item_instance = ComplexItem(name="item_x", value=420)
    serialized_item = prepare_for_storage(item_instance)
    update.set(update.fields.complex_list, [item_instance])
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "complex_list", {"value": [serialized_item]})

def test_set_nested_list_of_complex_items_with_dicts_valid():
    update = Update(NestedTypes)
    list_of_dicts = [{"name": "item_y", "value": 1000}, {"name": "item_z", "value": 2000}]
    update.set(update.fields.complex_list, list_of_dicts)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "complex_list", {"value": list_of_dicts})

# Complex Nested Validation Error Cases
def test_set_nested_simple_list_invalid_item_type_raises_error():
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="expected type int"): # simple_list is List[int]
        update.set(update.fields.simple_list, ["not", "integers"])
    assert len(update.build()) == 0

def test_set_nested_dict_field_invalid_value_type_raises_error():
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="expected type str"): # dict_field is Dict[str, str]
        update.set(update.fields.dict_field, {"key": 123})
    assert len(update.build()) == 0

def test_set_very_deeply_nested_primitive_invalid_type_raises_error():
    update = Update(NestedTypes)
    with pytest.raises(ValueTypeError, match="expected type int"): # nested.inner.val is int
        update.set(update.fields.nested.inner.val, "not_int_value")
    assert len(update.build()) == 0

def test_set_nested_list_of_complex_items_invalid_sub_type_in_dict_raises_error():
    update = Update(NestedTypes)
    # complex_list is List[ComplexItem], ComplexItem.name is str
    with pytest.raises(ValueTypeError, match="expected type str"):
        update.set(update.fields.complex_list, [{"name": 12345, "value": 42}])
    assert len(update.build()) == 0

def test_set_nested_list_of_complex_items_missing_field_in_dict_raises_error():
    update = Update(NestedTypes)
    # complex_list is List[ComplexItem], ComplexItem requires 'value'
    with pytest.raises(ValueTypeError, match="missing required field 'value'"):
        update.set(update.fields.complex_list, [{"name": "incomplete_item"}])
    assert len(update.build()) == 0


# --- Tests for `set` without model type (no validation) ---

def test_set_any_field_no_model():
    update = Update()
    update.set("any_field_name", "any_value_type")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "any_field_name", {"value": "any_value_type"})

def test_set_nested_path_no_model():
    update = Update()
    update.set("some_nested.field", "a_nested_value")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "some_nested.field", {"value": "a_nested_value"})

def test_set_list_value_no_model():
    update = Update()
    update.set("a_list_field", [10, {"sub_key": 20}])
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "a_list_field", {"value": [10, {"sub_key": 20}]})


# --- Test `set` with DSL-like path ---

def test_update_dsl_nested_set():
    """Test set operation on nested fields using string path."""
    update = Update(Entity).set("metadata.settings.theme", "dark_theme")
    result = update.build()
    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, SetOperation)
    assert operation.field_path == "metadata.settings.theme"
    assert operation.value == "dark_theme"