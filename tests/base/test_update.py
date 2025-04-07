import pytest
from typing import List, Dict, Optional, Union
from datetime import datetime

from repositories.base.update import Update


# Define test model classes
class Address:
    street: str
    city: str
    zipcode: str

    def __init__(self, street="", city="", zipcode=""):
        self.street = street
        self.city = city
        self.zipcode = zipcode


# Define a class for metadata to handle nested field validation
class Metadata:
    key1: str
    key2: int
    flag: bool

    def __init__(self, key1="", key2=0, flag=False):
        self.key1 = key1
        self.key2 = key2
        self.flag = flag


class User:
    name: str
    age: int
    email: Optional[str]
    active: bool
    tags: List[str]
    addresses: List[Address]
    metadata: Metadata

    def __init__(self, name="", age=0, email=None, active=True, tags=None, addresses=None, metadata=None):
        self.name = name
        self.age = age
        self.email = email
        self.active = active
        self.tags = tags or []
        self.addresses = addresses or []
        self.metadata = metadata or Metadata()


class Inner:
    val: int

    def __init__(self, val=0):
        self.val = val


class Outer:
    inner: Inner

    def __init__(self, inner=None):
        self.inner = inner or Inner()


class ComplexItem:
    name: str
    value: int

    def __init__(self, name="", value=0):
        self.name = name
        self.value = value


class NestedTypes:
    simple_list: List[int]
    str_list: List[str]
    dict_field: Dict[str, str]
    nested: Outer
    complex_list: List[ComplexItem]

    def __init__(self):
        self.simple_list = []
        self.str_list = []
        self.dict_field = {}
        self.nested = Outer()
        self.complex_list = []


# Test basic validation
def test_update_with_valid_types():
    """Test that valid types in Update operations pass validation."""
    update = Update(User)

    # Test valid operations
    update.set("name", "John")
    update.set("age", 30)
    update.set("email", "john@example.com")
    update.set("active", False)
    update.set("tags", ["tag1", "tag2"])
    update.set("metadata", {"key": "value", "count": 123})

    # Build should succeed without errors
    result = update.build()
    assert "$set" in result
    assert len(result["$set"]) == 6


def test_update_with_invalid_types():
    """Test that invalid types in Update operations raise TypeError."""
    update = Update(User)

    # Test invalid field type
    with pytest.raises(TypeError):
        update.set("name", 123)  # Name should be string, not int

    with pytest.raises(TypeError):
        update.set("age", "thirty")  # Age should be int, not string

    with pytest.raises(TypeError):
        update.set("active", "yes")  # Active should be bool, not string

    with pytest.raises(TypeError):
        update.set("tags", "not-a-list")  # Tags should be a list, not string

    with pytest.raises(TypeError):
        update.set("tags", [1, 2, 3])  # Tags should be List[str], not List[int]

    with pytest.raises(TypeError):
        update.set("metadata", ["not", "a", "dict"])  # Metadata should be dict, not list


def test_update_with_invalid_field():
    """Test that non-existent fields in Update operations raise TypeError."""
    update = Update(User)

    with pytest.raises(TypeError):
        update.set("non_existent_field", "value")


def test_update_with_optional_fields():
    """Test that None is accepted for Optional fields."""
    update = Update(User)

    # Email is Optional[str]
    update.set("email", None)  # Should not raise error
    update.set("email", "valid@example.com")  # Should not raise error

    with pytest.raises(TypeError):
        update.set("name", None)  # Name is not optional


# Test nested fields
def test_update_with_nested_fields():
    """Test update operations with nested field access."""
    update = Update(User)

    # Valid nested operations
    update.set("metadata.key1", "value1")

    # Test invalid nested field
    with pytest.raises(TypeError):
        update.set("metadata.key2", [])  # Should be int, not list

    with pytest.raises(TypeError):
        update.set("non_existent.field", "value")


# Test list/container operations
def test_update_push_with_type_validation():
    """Test that push operations are type validated."""
    update = Update(User)

    # Valid push
    update.push("tags", "new_tag")

    # Invalid push (wrong type)
    with pytest.raises(TypeError):
        update.push("tags", 123)  # tags should contain strings

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.push("name", "value")  # name is str, not a list

    # Non-existent field
    with pytest.raises(TypeError):
        update.push("non_existent", "value")


def test_update_pop_with_type_validation():
    """Test that pop operations are type validated."""
    update = Update(User)

    # Valid pop
    update.pop("tags")
    update.pop("tags", 1)
    update.pop("tags", -1)

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pop("name")

    # Non-existent field
    with pytest.raises(TypeError):
        update.pop("non_existent")

    # Invalid direction
    with pytest.raises(ValueError):
        update.pop("tags", 2)  # direction must be 1 or -1


def test_update_pull_with_type_validation():
    """Test that pull operations are type validated."""
    update = Update(User)

    # Valid pull
    update.pull("tags", "tag_to_remove")

    # Invalid pull (wrong type)
    with pytest.raises(TypeError):
        update.pull("tags", 123)  # tags should contain strings

    # Invalid field (not a list)
    with pytest.raises(TypeError):
        update.pull("name", "value")

    # Non-existent field
    with pytest.raises(TypeError):
        update.pull("non_existent", "value")


def test_update_unset_with_type_validation():
    """Test that unset operations are validated for field existence."""
    update = Update(User)

    # Valid unset
    update.unset("email")
    update.unset("metadata.key1")

    # Non-existent field
    with pytest.raises(TypeError):
        update.unset("non_existent")

    with pytest.raises(TypeError):
        update.unset("metadata.non_existent")


# Test complex nested validations
def test_complex_nested_validations():
    """Test validation with complex nested structures."""
    update = Update(NestedTypes)

    # Valid operations
    update.set("simple_list", [1, 2, 3])
    update.set("str_list", ["a", "b", "c"])
    update.set("dict_field", {"key": "value"})

    # Create proper nested objects
    inner = Inner(42)
    outer = Outer(inner)
    update.set("nested", outer)

    # Create a proper complex item
    item = ComplexItem("test", 42)
    update.set("complex_list", [item])

    # Invalid operations
    with pytest.raises(TypeError):
        update.set("simple_list", ["not", "integers"])

    with pytest.raises(TypeError):
        update.set("dict_field", {"key": 123})  # Value should be string

    with pytest.raises(TypeError):
        # Use proper structure but wrong type
        invalid_inner = Inner("not_int")  # Inner.val should be int, not string
        update.set("nested.inner.val", "not_int")

    with pytest.raises(TypeError):
        # Create an invalid complex item with name as int (should be string)
        update.set("complex_list", [ComplexItem(123, 42)])


# Test without type validation
def test_update_without_model_type():
    """Test that Update works without a model type (no validation)."""
    update = Update()  # No model type

    # All these operations should work without errors
    update.set("any_field", "any_value")
    update.set("number", 123)
    update.push("list_field", "item")
    update.pop("another_list")
    update.pull("some_list", "value")
    update.unset("field_to_remove")

    # Build should succeed
    result = update.build()
    assert "$set" in result
    assert "$push" in result
    assert "$pop" in result
    assert "$pull" in result
    assert "$unset" in result


# Test repr functionality
def test_update_repr():
    """Test the string representation of Update includes model type."""
    # Without model
    update1 = Update()
    assert "Update()" == repr(update1)

    # With model
    update2 = Update(User)
    assert "Update(User)" == repr(update2)

    # With operations
    update3 = Update(User).set("name", "John").set("age", 30)
    assert "Update(User).set" in repr(update3)
    assert "\"John\"" in repr(update3)
    assert "30" in repr(update3)


# Test integration with documented examples
def test_integration_with_examples():
    """Test the Update class using examples from the class docstring."""

    class User:
        name: str
        age: int
        tags: List[str]

    # Valid operations
    update = Update(User).set("name", "John").set("age", 30)
    result = update.build()
    assert result["$set"]["name"] == "John"
    assert result["$set"]["age"] == 30

    # Invalid operation
    with pytest.raises(TypeError):
        Update(User).set("name", 123)


# Test from existing code examples in provided tests
def test_existing_code_compatibility():
    """Test compatibility with existing code from test file."""
    # From the provided test file examples
    update = Update().set("name", "Updated Name").set("value", 200)
    result = update.build()
    assert result["$set"]["name"] == "Updated Name"
    assert result["$set"]["value"] == 200

    update = Update().push("tags", "new_tag")
    result = update.build()
    assert result["$push"]["tags"] == "new_tag"

    update = Update().pop("tags", 1)
    result = update.build()
    assert result["$pop"]["tags"] == 1

    update = Update().unset("metadata.note")
    result = update.build()
    assert result["$unset"]["metadata.note"] == ""

    update = Update().pull("tags", "y")
    result = update.build()
    assert result["$pull"]["tags"] == "y"


# Test combined operations in a single Update
def test_combined_operations():
    """Test multiple different operations in a single Update instance."""
    update = (Update(User)
              .set("name", "John")
              .push("tags", "new_tag")
              .pop("addresses", 1)
              .unset("email")
              .pull("tags", "old_tag"))

    result = update.build()
    assert "$set" in result
    assert "$push" in result
    assert "$pop" in result
    assert "$unset" in result
    assert "$pull" in result

    assert result["$set"]["name"] == "John"
    assert result["$push"]["tags"] == "new_tag"
    assert result["$pop"]["addresses"] == 1
    assert result["$unset"]["email"] == ""
    assert result["$pull"]["tags"] == "old_tag"


# Test with Union types
def test_update_with_union_types():
    """Test that Union type validation works correctly."""

    class ModelWithUnions:
        field: Union[str, int]
        container: List[Union[str, int, bool]]

    update = Update(ModelWithUnions)

    # Valid values for Union[str, int]
    update.set("field", "string value")
    update.set("field", 42)

    # Valid values for List[Union[str, int, bool]]
    update.set("container", ["string", 42, True])

    # Invalid value for Union[str, int]
    with pytest.raises(TypeError):
        update.set("field", [])  # [] is neither str nor int

    # Invalid value in List[Union[str, int, bool]]
    with pytest.raises(TypeError):
        update.set("container", ["string", 42, {}])  # {} is not in the Union