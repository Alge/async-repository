# tests/base/update/test_complex_types.py

import pytest
from typing import List, Optional

# Import Update builder and relevant components
from async_repository.base.update import (
    Update,
    PushOperation,
    PopOperation,
    PullOperation,
    InvalidPathError,
    ValueTypeError,
)

# --- Test Model ---
class ModelWithOptionalList:
    maybe_list: Optional[List[str]] = None
    another_field: int = 0


# --- Test Functions ---

def test_push_on_optional_list_valid_item():
    """
    Tests that a push operation with a valid item type on an
    Optional[List[str]] field is correctly added.
    """
    update = Update(ModelWithOptionalList)
    update.push("maybe_list", "item1")
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PushOperation)
    assert operation.field_path == "maybe_list"
    assert operation.items == ["item1"]


def test_pop_default_on_optional_list():
    """
    Tests that a pop operation (default position 1) on an
    Optional[List[str]] field is correctly added.
    """
    update = Update(ModelWithOptionalList)
    update.pop("maybe_list")  # Default position for pop is 1.
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PopOperation)
    assert operation.field_path == "maybe_list"
    assert operation.position == 1


def test_pull_on_optional_list_valid_item():
    """
    Tests that a pull operation with a valid item on an
    Optional[List[str]] field is correctly added.
    """
    update = Update(ModelWithOptionalList)
    update.pull("maybe_list", "item_to_remove")
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PullOperation)
    assert operation.field_path == "maybe_list"
    assert operation.value_or_condition == "item_to_remove"


def test_pop_negative_position_on_optional_list():
    """
    Tests that a pop operation (position -1, from the end) on an
    Optional[List[str]] field is correctly added.
    """
    update = Update(ModelWithOptionalList)
    update.pop("maybe_list", -1)
    result = update.build()

    assert len(result) == 1
    operation = result[0]
    assert isinstance(operation, PopOperation)
    assert operation.field_path == "maybe_list"
    assert operation.position == -1


def test_push_on_optional_list_invalid_item_type_raises_value_type_error():
    """
    Tests that pushing an item of an invalid type to an Optional[List[str]]
    field raises ValueTypeError. Validation uses the inner list item type (str).
    """
    update = Update(ModelWithOptionalList)
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'maybe_list'"
    ):
        update.push("maybe_list", 123)  # 123 is int, expected str based on List[str]

    # Ensure the invalid operation was not added.
    assert len(update.build()) == 0


def test_pop_on_non_list_field_raises_invalid_path_error():
    """
    Tests that a pop operation on a non-list field raises InvalidPathError.
    """
    update = Update(ModelWithOptionalList)
    with pytest.raises(InvalidPathError):
        update.pop("another_field") # 'another_field' is int, not a list.

    # Ensure the invalid operation was not added.
    assert len(update.build()) == 0


def test_pull_on_non_list_field_raises_invalid_path_error():
    """
    Tests that a pull operation on a non-list field raises InvalidPathError.
    """
    update = Update(ModelWithOptionalList)
    with pytest.raises(InvalidPathError):
        update.pull("another_field", 0) # 'another_field' is int, not a list.

    # Ensure the invalid operation was not added.
    assert len(update.build()) == 0