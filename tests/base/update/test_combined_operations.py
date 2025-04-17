import pytest
from async_repository.base.update import Update
from .conftest import User, NumericModel, NestedTypes


def test_min_max_mul_combined():
    """Test min, max, and mul operations together."""
    update = (
        Update(NumericModel)
        .min("int_field", 10)
        .max("float_field", 100.0)
        .mul("union_numeric", 2)
    )

    result = update.build()
    assert "$min" in result
    assert "$max" in result
    assert "$mul" in result
    assert result["$min"]["int_field"] == 10
    assert result["$max"]["float_field"] == 100.0
    assert result["$mul"]["union_numeric"] == 2


def test_with_existing_operations():
    """Test min, max, and mul combined with other operations."""
    update = (
        Update(NumericModel)
        .set("optional_int", 42)
        .min("int_field", 10)
        .increment("int_field", 5)
        .increment("int_field", 3)  # Second increment accumulates
        .max("float_field", 100.0)
        .mul("union_numeric", 2)
    )

    result = update.build()
    assert "$set" in result
    assert "$min" in result
    assert "$inc" in result
    assert "$max" in result
    assert "$mul" in result
    assert result["$set"]["optional_int"] == 42
    assert result["$min"]["int_field"] == 10
    assert result["$inc"]["int_field"] == 8  # 5 + 3 = 8
    assert result["$max"]["float_field"] == 100.0
    assert result["$mul"]["union_numeric"] == 2


def test_nested_combined_operations():
    """Test combined operations on nested fields."""
    update = (
        Update(NestedTypes)
        .min("nested.inner.val", 5)
        .max("nested.inner.val", 100)
        .mul("counter", 2)
    )

    result = update.build()
    assert "$min" in result
    assert "$max" in result
    assert "$mul" in result
    assert result["$min"]["nested.inner.val"] == 5
    assert result["$max"]["nested.inner.val"] == 100
    assert result["$mul"]["counter"] == 2


def test_all_operations_combined():
    """Test all supported Update operations in a single update."""
    update = (
        Update(User)
        .set("name", "John Doe")
        .push("tags", "active")
        .pop("addresses", -1)
        .pull("tags", "inactive")
        .unset("metadata.key1")
        .increment("points", 50)
        .decrement("balance", 10.5)
        .min("points", 0)
        .max("score", 100)
        .mul("points", 1.1)
    )

    result = update.build()

    # Verify all operation types are present
    assert "$set" in result
    assert "$push" in result
    assert "$pop" in result
    assert "$pull" in result
    assert "$unset" in result
    assert "$inc" in result
    assert "$min" in result
    assert "$max" in result
    assert "$mul" in result

    # Verify values
    assert result["$set"]["name"] == "John Doe"
    assert result["$push"]["tags"] == "active"
    assert result["$pop"]["addresses"] == -1
    assert result["$pull"]["tags"] == "inactive"
    assert result["$unset"]["metadata.key1"] == ""
    assert result["$inc"]["points"] == 50
    assert result["$inc"]["balance"] == -10.5
    assert result["$min"]["points"] == 0
    assert result["$max"]["score"] == 100
    assert result["$mul"]["points"] == 1.1