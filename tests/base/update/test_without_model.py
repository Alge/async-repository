import pytest
from async_repository.base.update import Update


def test_update_without_model():
    """Test basic operations without a model class."""
    update = Update()  # No model provided

    # Basic operations with string fields
    update.set("name", "John Doe")
    update.increment("counter", 5)
    update.push("tags", "new_tag")

    result = update.build()
    assert "$set" in result
    assert "$inc" in result
    assert "$push" in result
    assert result["$set"]["name"] == "John Doe"
    assert result["$inc"]["counter"] == 5
    assert result["$push"]["tags"] == "new_tag"


def test_fields_proxy_without_model():
    """Test using the fields proxy without a model class."""
    update = Update()  # No model provided

    # Using dynamic fields proxy
    update.set(update.fields.name, "John Doe")
    update.increment(update.fields.counter, 5)
    update.push(update.fields.tags, "new_tag")

    # Nested fields access
    update.set(update.fields.user.profile.age, 30)
    update.set(update.fields.preferences.theme, "dark")

    # Multi-level nesting
    update.push(update.fields.posts.comments.replies, "New reply")

    result = update.build()
    assert "$set" in result
    assert "$inc" in result
    assert "$push" in result
    assert result["$set"]["name"] == "John Doe"
    assert result["$set"]["user.profile.age"] == 30
    assert result["$set"]["preferences.theme"] == "dark"
    assert result["$inc"]["counter"] == 5
    assert result["$push"]["tags"] == "new_tag"
    assert result["$push"]["posts.comments.replies"] == "New reply"


def test_complex_operations_without_model():
    """Test more complex operations without a model class."""
    update = Update()  # No model provided

    # Min/max/mul operations
    update.min(update.fields.score, 0)
    update.max(update.fields.score, 100)
    update.mul(update.fields.multiplier, 1.5)

    # Array operations
    update.push(update.fields.items, {"id": 1, "name": "Item 1"})
    update.pop(update.fields.recent_items, -1)  # Remove first item
    update.pull(update.fields.tags, "old_tag")

    # Nested operations
    update.unset(update.fields.user.temporary_data)
    update.increment(update.fields.stats.visits, 1)

    result = update.build()
    assert "$min" in result
    assert "$max" in result
    assert "$mul" in result
    assert "$push" in result
    assert "$pop" in result
    assert "$pull" in result
    assert "$unset" in result
    assert "$inc" in result

    assert result["$min"]["score"] == 0
    assert result["$max"]["score"] == 100
    assert result["$mul"]["multiplier"] == 1.5
    assert result["$push"]["items"] == {"id": 1, "name": "Item 1"}
    assert result["$pop"]["recent_items"] == -1
    assert result["$pull"]["tags"] == "old_tag"
    assert result["$unset"]["user.temporary_data"] == ""
    assert result["$inc"]["stats.visits"] == 1


def test_increment_restrictions_without_model():
    """Test that increment restrictions still apply without a model."""
    update = Update()  # No model provided

    # First increment is fine
    update.increment(update.fields.counter, 5)

    # Second increment on same field should be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.increment(update.fields.counter, 3)

    # Decrement after increment should also be rejected
    with pytest.raises(ValueError, match="already has an increment"):
        update.decrement(update.fields.counter, 2)

    # But increment on different field is okay
    update.increment(update.fields.another_counter, 10)

    result = update.build()
    assert result["$inc"]["counter"] == 5
    assert result["$inc"]["another_counter"] == 10