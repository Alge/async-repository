import pytest
from async_repository.base.update import Update
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


class TestNestedOperations:
    """
    Comprehensive tests for all operations on nested objects with type checking.
    """

    def test_set_on_nested_objects(self):
        """Test set operations on various nested object structures."""
        # Test User.metadata (nested object)
        update = Update(User)
        update.set("metadata.key1", "new_value")
        update.set("metadata.key2", 42)
        update.set("metadata.flag", True)

        # Test NestedTypes.nested.inner (deeply nested object)
        update = Update(NestedTypes)
        update.set("nested.inner.val", 100)

        # Test incorrect types
        with pytest.raises(TypeError):
            update.set("nested.inner.val", "not_an_int")

        with pytest.raises(TypeError):
            Update(User).set("metadata.key2", "not_an_int")

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(User).set("metadata.non_existent", "value")

        with pytest.raises(TypeError):
            Update(NestedTypes).set("nested.inner.non_existent", 42)

        # Test setting complete nested objects
        metadata = Metadata("new_key1", 99, False)
        Update(User).set("metadata", metadata)

        inner = Inner(500)
        outer = Outer(inner)
        Update(NestedTypes).set("nested", outer)

        # Test setting complete nested objects with invalid types
        with pytest.raises(TypeError):
            Update(User).set("metadata", "not_a_metadata")

        with pytest.raises(TypeError):
            Update(NestedTypes).set("nested", "not_an_outer")

    def test_push_on_nested_lists(self):
        """Test push operations on nested lists."""
        # Test User.addresses (list of Address objects)
        update = Update(User)
        address = Address("123 Main St", "Anytown", "12345")
        update.push("addresses", address)

        # Test Organization.departments[0].members (nested list of strings)
        update = Update(Organization)
        update.push("departments.0.members", "new_member")

        # Test Organization.departments[0].categories[0].items (deeply nested list of strings)
        update.push("departments.0.categories.0.items", "new_item")

        # Test Organization.departments[0].categories[0].counts (deeply nested list of ints)
        update.push("departments.0.categories.0.counts", 99)

        # Test incorrect types
        with pytest.raises(TypeError):
            Update(User).push("addresses", "not_an_address")

        with pytest.raises(TypeError):
            Update(Organization).push("departments.0.categories.0.items", 42)

        with pytest.raises(TypeError):
            Update(Organization).push("departments.0.categories.0.counts", "not_an_int")

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(Organization).push("departments.0.non_existent", "value")

        # Test pushing to non-list fields
        with pytest.raises(TypeError):
            Update(Organization).push("departments.0.name", "value")

    def test_pop_on_nested_lists(self):
        """Test pop operations on nested lists."""
        # Test User.addresses (list of Address objects)
        update = Update(User)
        update.pop("addresses")  # Default (last)
        update.pop("addresses", -1)  # First

        # Test Organization.departments[0].members (nested list of strings)
        update = Update(Organization)
        update.pop("departments.0.members")

        # Test Organization.departments[0].categories[0].items (deeply nested list of strings)
        update.pop("departments.0.categories.0.items", 1)

        # Test Organization.departments[0].categories[0].counts (deeply nested list of ints)
        update.pop("departments.0.categories.0.counts", -1)

        # Test invalid direction
        with pytest.raises(ValueError):
            Update(User).pop("addresses", 2)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(Organization).pop("departments.0.non_existent")

        # Test popping from non-list fields
        with pytest.raises(TypeError):
            Update(Organization).pop("departments.0.name")

    def test_pull_on_nested_lists(self):
        """Test pull operations on nested lists."""
        # Test User.tags (list of strings)
        update = Update(User)
        update.pull("tags", "tag_to_remove")

        # Test Organization.departments[0].members (nested list of strings)
        update = Update(Organization)
        update.pull("departments.0.members", "member1")

        # Test Organization.departments[0].categories[0].items (deeply nested list of strings)
        update.pull("departments.0.categories.0.items", "item1")

        # Test Organization.departments[0].categories[0].counts (deeply nested list of ints)
        update.pull("departments.0.categories.0.counts", 2)

        # Test incorrect types
        with pytest.raises(TypeError):
            Update(User).pull("tags", 42)

        with pytest.raises(TypeError):
            Update(Organization).pull("departments.0.categories.0.items", 42)

        with pytest.raises(TypeError):
            Update(Organization).pull("departments.0.categories.0.counts", "not_an_int")

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(Organization).pull("departments.0.non_existent", "value")

        # Test pulling from non-list fields
        with pytest.raises(TypeError):
            Update(Organization).pull("departments.0.name", "value")

    def test_unset_on_nested_fields(self):
        """Test unset operations on nested fields."""
        # Test User.metadata fields
        update = Update(User)
        update.unset("metadata.key1")
        update.unset("metadata.key2")
        update.unset("metadata.flag")

        # Test NestedTypes deeply nested fields
        update = Update(NestedTypes)
        update.unset("nested.inner.val")

        # Test Organization nested fields
        update = Update(Organization)
        update.unset("departments.0.name")
        update.unset("departments.0.categories.0.name")

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(User).unset("metadata.non_existent")

        with pytest.raises(TypeError):
            Update(NestedTypes).unset("nested.inner.non_existent")

    def test_increment_on_nested_fields(self):
        """Test increment operations on nested numeric fields."""
        # Test NestedTypes.nested.inner.val (deeply nested int)
        update = Update(NestedTypes)
        update.increment("nested.inner.val", 5)
        update.increment("counter", 10)

        # Test Organization.departments[0].categories[0].counts directly (not valid for increment)
        with pytest.raises(TypeError):
            Update(Organization).increment("departments.0.categories.0.counts", 5)

        # Test non-numeric fields
        with pytest.raises(TypeError):
            Update(User).increment("name", 5)

        with pytest.raises(TypeError):
            Update(NestedTypes).increment("nested.inner", 5)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(NestedTypes).increment("nested.inner.non_existent", 5)

        # Test invalid increment amount type
        with pytest.raises(TypeError):
            Update(NestedTypes).increment("nested.inner.val", "not_a_number")

    def test_decrement_on_nested_fields(self):
        """Test decrement operations on nested numeric fields."""
        # Test NestedTypes.nested.inner.val (deeply nested int)
        update = Update(NestedTypes)
        update.decrement("nested.inner.val", 5)
        update.decrement("counter", 10)

        # Test non-numeric fields
        with pytest.raises(TypeError):
            Update(User).decrement("name", 5)

        with pytest.raises(TypeError):
            Update(NestedTypes).decrement("nested.inner", 5)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(NestedTypes).decrement("nested.inner.non_existent", 5)

        # Test invalid decrement amount type
        with pytest.raises(TypeError):
            Update(NestedTypes).decrement("nested.inner.val", "not_a_number")

    def test_min_on_nested_fields(self):
        """Test min operations on nested numeric fields."""
        # Test NestedTypes.nested.inner.val (deeply nested int)
        update = Update(NestedTypes)
        update.min("nested.inner.val", 5)
        update.min("counter", 0)

        # Test non-numeric fields
        with pytest.raises(TypeError):
            Update(User).min("name", 5)

        with pytest.raises(TypeError):
            Update(NestedTypes).min("nested.inner", 5)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(NestedTypes).min("nested.inner.non_existent", 5)

        # Test invalid min value type
        with pytest.raises(TypeError):
            Update(NestedTypes).min("nested.inner.val", "not_a_number")

    def test_max_on_nested_fields(self):
        """Test max operations on nested numeric fields."""
        # Test NestedTypes.nested.inner.val (deeply nested int)
        update = Update(NestedTypes)
        update.max("nested.inner.val", 100)
        update.max("counter", 1000)

        # Test non-numeric fields
        with pytest.raises(TypeError):
            Update(User).max("name", 100)

        with pytest.raises(TypeError):
            Update(NestedTypes).max("nested.inner", 100)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(NestedTypes).max("nested.inner.non_existent", 100)

        # Test invalid max value type
        with pytest.raises(TypeError):
            Update(NestedTypes).max("nested.inner.val", "not_a_number")

    def test_mul_on_nested_fields(self):
        """Test mul operations on nested numeric fields."""
        # Test NestedTypes.nested.inner.val (deeply nested int)
        update = Update(NestedTypes)
        update.mul("nested.inner.val", 2)
        update.mul("counter", 1.5)

        # Test non-numeric fields
        with pytest.raises(TypeError):
            Update(User).mul("name", 2)

        with pytest.raises(TypeError):
            Update(NestedTypes).mul("nested.inner", 2)

        # Test non-existent nested paths
        with pytest.raises(TypeError):
            Update(NestedTypes).mul("nested.inner.non_existent", 2)

        # Test invalid mul factor type
        with pytest.raises(TypeError):
            Update(NestedTypes).mul("nested.inner.val", "not_a_number")

    def test_combined_nested_operations(self):
        """Test combined operations on nested structures."""
        update = Update(Organization)

        # Multiple operations on the same nested structure
        update.set("name", "New Organization")
        update.set("departments.0.name", "Updated Department")
        update.push("departments.0.members", "new_member")
        update.pull("departments.0.members", "member1")
        update.push("departments.0.categories.0.items", "new_item")
        update.pop("departments.0.categories.0.counts")
        update.unset("departments.0.categories.0.name")

        # Verify the resulting update document
        result = update.build()

        assert "$set" in result
        assert "$push" in result
        assert "$pull" in result
        assert "$pop" in result
        assert "$unset" in result

        assert result["$set"]["name"] == "New Organization"
        assert result["$set"]["departments.0.name"] == "Updated Department"
        assert result["$push"]["departments.0.members"] == "new_member"
        assert result["$pull"]["departments.0.members"] == "member1"
        assert result["$push"]["departments.0.categories.0.items"] == "new_item"
        assert result["$pop"]["departments.0.categories.0.counts"] == 1
        assert result["$unset"]["departments.0.categories.0.name"] == ""
