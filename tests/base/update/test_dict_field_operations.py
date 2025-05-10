# tests/base/update/test_dict_field_operations.py
from typing import Dict, Any, Optional, List

import pytest
from async_repository.base.update import Update, InvalidPathError, ValueTypeError, \
    IncrementOperation, PushOperation, PopOperation, PullOperation, MinOperation, \
    SetOperation, UnsetOperation
from tests.base.conftest import assert_operation_present # Your models

class ModelWithDict:
    data_any: Dict[str, Any]
    data_int: Dict[str, int]
    data_list_str: Dict[str, List[str]]
    optional_data_any: Optional[Dict[str, Any]]

    def __init__(self):
        self.data_any = {}
        self.data_int = {}
        self.data_list_str = {}
        self.optional_data_any = None

# --- Increment/Decrement on Dict Sub-paths ---
def test_increment_on_dict_any_subpath_valid_amount():
    update = Update(ModelWithDict)
    # 'data_any.counter' is not explicitly typed as numeric by validator for sub-path,
    # but operation should be allowed because data_any is a Dict.
    # The increment method's own check for numeric amount still applies.
    update.increment("data_any.counter", 5)
    result = update.build()
    assert_operation_present(result, IncrementOperation, "data_any.counter", {"amount": 5})

def test_increment_on_dict_any_subpath_invalid_amount_type_error():
    update = Update(ModelWithDict)
    with pytest.raises(TypeError, match="Increment amount must be numeric"):
        update.increment("data_any.counter", "not_a_number")

def test_decrement_on_dict_any_subpath_valid_amount():
    update = Update(ModelWithDict)
    update.decrement("data_any.another_counter", 2)
    result = update.build()
    assert_operation_present(result, IncrementOperation, "data_any.another_counter", {"amount": -2})

# --- Push/Pop/Pull on Dict Sub-paths (targeting list-like structures within the dict) ---
def test_push_to_dict_list_str_subpath():
    update = Update(ModelWithDict)
    # data_list_str.my_list is not explicitly List[str] via validator for sub-path,
    # but push should be allowed. The item pushed should match what the DB expects.
    update.push("data_list_str.my_list", "new_item")
    result = update.build()
    assert_operation_present(result, PushOperation, "data_list_str.my_list", {"items": ["new_item"]})

def test_pop_from_dict_list_str_subpath():
    update = Update(ModelWithDict)
    update.pop("data_list_str.another_list")
    result = update.build()
    assert_operation_present(result, PopOperation, "data_list_str.another_list", {"position": 1})

def test_pull_from_dict_list_str_subpath_literal():
    update = Update(ModelWithDict)
    update.pull("data_list_str.key_list", "item_to_pull")
    result = update.build()
    assert_operation_present(result, PullOperation, "data_list_str.key_list", {"value_or_condition": "item_to_pull"})

def test_pull_from_dict_any_subpath_operator_dict():
    update = Update(ModelWithDict)
    condition = {"$exists": True}
    update.pull("data_any.some_list_key", condition)
    result = update.build()
    assert_operation_present(result, PullOperation, "data_any.some_list_key", {"value_or_condition": condition})


# --- Min/Max/Mul on Dict Sub-paths ---
def test_min_on_dict_any_subpath_valid_value():
    update = Update(ModelWithDict)
    update.min("data_any.min_val", 10)
    result = update.build()
    assert_operation_present(result, MinOperation, "data_any.min_val", {"value": 10})

def test_min_on_dict_any_subpath_invalid_value_type_error():
    update = Update(ModelWithDict)
    with pytest.raises(TypeError, match="Min value must be numeric"):
        update.min("data_any.min_val", "not_a_number")

# ... similar tests for max and mul ...


# --- Set/Unset on Dict Sub-paths (These DO NOT use _is_nested_path_in_dict) ---
def test_set_on_dict_int_subpath_valid():
    update = Update(ModelWithDict)
    # ModelValidator should be able to validate "data_int.key" against int type
    update.set("data_int.new_key", 123)
    result = update.build()
    assert_operation_present(result, SetOperation, "data_int.new_key", {"value": 123})

def test_set_on_dict_int_subpath_invalid_type_value_error():
    update = Update(ModelWithDict)
    # ModelValidator should raise ValueTypeError if it can validate dict sub-keys
    with pytest.raises(ValueTypeError): # Assuming validator can check Dict[str, int] sub-keys
        update.set("data_int.another_key", "not_an_int")

def test_set_on_dict_any_subpath_no_validation_error():
    update = Update(ModelWithDict)
    # For Dict[str, Any], validator might not raise ValueTypeError for the sub-key's value type
    update.set("data_any.flexible_key", "any_value_type_here")
    result = update.build()
    assert_operation_present(result, SetOperation, "data_any.flexible_key", {"value": "any_value_type_here"})

def test_unset_on_dict_any_subpath():
    update = Update(ModelWithDict)
    update.unset("data_any.key_to_remove")
    result = update.build()
    assert_operation_present(result, UnsetOperation, "data_any.key_to_remove")

# Test with Optional[Dict]
def test_increment_on_optional_dict_subpath():
    update = Update(ModelWithDict)
    update.increment("optional_data_any.counter", 1)
    result = update.build()
    assert_operation_present(result, IncrementOperation, "optional_data_any.counter", {"amount": 1})