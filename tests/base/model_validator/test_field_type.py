# tests/base/model_validator/test_field_type.py
import pytest
from typing import Any, Dict, List, Optional, Union, Set, Tuple

from pydantic import BaseModel

from async_repository.base.model_validator import ModelValidator, InvalidPathError

from .conftest import (
    Inner,
    OuterDataClass,
    Node,
    PydanticInner,
)

# --- get_field_type Tests ---


def test_get_field_type_simple(simple_validator):
    """Test getting types for simple fields."""
    assert simple_validator.get_field_type("name") is str
    assert simple_validator.get_field_type("age") is int
    assert simple_validator.get_field_type("is_active") is bool


def test_get_field_type_nested(nested_validator):
    """Test getting types for nested fields."""
    assert nested_validator.get_field_type("outer") is OuterDataClass
    assert nested_validator.get_field_type("outer.inner") is Inner
    assert nested_validator.get_field_type("outer.inner.val") is int
    assert nested_validator.get_field_type("outer.items") is List[int]
    assert nested_validator.get_field_type("config") is Dict[str, Any]
    assert nested_validator.get_field_type("maybe_inner") is Optional[Inner]
    # Type within Optional
    assert nested_validator.get_field_type("maybe_inner.val") is int
    assert nested_validator.get_field_type("maybe_inner.description") is Optional[str]


def test_get_field_type_pydantic(pydantic_validator):
    """Test getting types for fields in a Pydantic model."""
    assert pydantic_validator.get_field_type("id") is str
    assert pydantic_validator.get_field_type("count") is int
    assert pydantic_validator.get_field_type("nested") is PydanticInner
    # Pydantic aliases are not resolved by get_type_hints directly
    # The validator gets the declared type, not the alias type
    assert pydantic_validator.get_field_type("nested.p_val") is float
    assert pydantic_validator.get_field_type("optional_list") is Optional[List[str]]
    assert pydantic_validator.get_field_type("union_field") is Union[int, str]
    assert pydantic_validator.get_field_type("any_field") is Any
    assert pydantic_validator.get_field_type("simple_list") is list
    assert pydantic_validator.get_field_type("typed_dict") is Dict[str, int]
    assert pydantic_validator.get_field_type("tuple_field") is Tuple[int, str, bool]
    assert pydantic_validator.get_field_type("set_field") is Set[float]


def test_get_field_type_list_index(nested_validator, pydantic_validator):
    """Test getting types for items within lists using index notation."""
    # Dataclass list
    assert nested_validator.get_field_type("outer.items.0") is int
    # Pydantic list
    # Optional[List[str]] -> List[str] -> str
    assert pydantic_validator.get_field_type("optional_list.0") is str
    assert pydantic_validator.get_field_type("simple_list.0") is Any  # list -> Any
    # Tuple behaves like list index here
    assert pydantic_validator.get_field_type("tuple_field.0") is int
    # Set behaves like list index here
    assert pydantic_validator.get_field_type("set_field.0") is float


def test_get_field_type_dict_key(nested_validator, pydantic_validator):
    """Test getting types for values within dicts using key notation."""
    # Any dict
    assert nested_validator.get_field_type("config.some_key") is Any
    # Typed dict
    assert pydantic_validator.get_field_type("typed_dict.some_key") is int


def test_get_field_type_any(pydantic_validator, nested_validator):
    """Test traversing through Any type."""
    assert pydantic_validator.get_field_type("any_field") is Any
    assert pydantic_validator.get_field_type("any_field.nested.path") is Any
    assert nested_validator.get_field_type("config.key.nested") is Any


def test_get_field_type_forward_ref():
    """Test resolving forward references."""
    validator = ModelValidator[Node](Node)  # Specify generic type
    assert validator.get_field_type("value") is int
    assert validator.get_field_type("next_node") is Optional[Node]
    assert validator.get_field_type("next_node.value") is int
    # Deeper nesting
    assert validator.get_field_type("next_node.next_node.value") is int


def test_get_field_type_invalid_path(simple_validator, nested_validator):
    """Test getting types for invalid paths raises InvalidPathError."""
    with pytest.raises(InvalidPathError):
        simple_validator.get_field_type("non_existent")
    with pytest.raises(InvalidPathError):
        # Accessing attribute on str
        simple_validator.get_field_type("name.nested")
    with pytest.raises(InvalidPathError):
        nested_validator.get_field_type("outer.non_existent")
    with pytest.raises(InvalidPathError):
        # Accessing attribute on list element
        nested_validator.get_field_type("outer.items.invalid")
    with pytest.raises(InvalidPathError):
        # Accessing attribute on int
        nested_validator.get_field_type("outer.inner.val.nested")


def test_get_field_type_invalid_list_index(simple_validator):
    """Test accessing list index on non-list field."""
    with pytest.raises(InvalidPathError):
        simple_validator.get_field_type("name.0")


def test_get_field_type_invalid_dict_key_type(pydantic_validator):
    """Test traversing Dict path with non-string key type."""

    class DictWithIntKey(BaseModel):
        data: Dict[int, str]

    validator = ModelValidator[DictWithIntKey](DictWithIntKey)  # Specify generic type
    # This path is invalid because '123' is not a string key,
    # even though the key type is int. Dot notation implies string keys.
    with pytest.raises(InvalidPathError, match="non-string key type"):
        validator.get_field_type("data.123")


def test_get_field_type_empty_path(simple_validator):
    """Test that an empty field path raises ValueError."""
    with pytest.raises(ValueError, match="field_path cannot be empty"):
        simple_validator.get_field_type("")
