# tests/base/update/test_complex_types.py

import pytest
from typing import List, Optional  # Added imports

# Import Update builder and relevant components
from async_repository.base.update import (
    Update,
    PushOperation,
    PopOperation,
    PullOperation,
    InvalidPathError,
    ValueTypeError,
)

from tests.base.conftest import find_operations


# --- Test Model ---
class ModelWithOptionalList:
    maybe_list: Optional[List[str]] = None
    another_field: int = 0


# --- Test Function ---
def test_ops_on_optional_list():
    """
    Tests list operations (push, pop, pull) on Optional[List[T]] fields.
    Ensures validation uses the inner list item type correctly.
    """
    update = Update(ModelWithOptionalList)

    # These operations should pass validation based on the inner List[str] type hint
    update.push("maybe_list", "item1")  # Op 0
    update.pop("maybe_list")  # Op 1 (position=1)
    update.pull("maybe_list", "item_to_remove")  # Op 2
    update.pop("maybe_list", -1)  # Op 3 (position=-1)

    result = update.build()

    assert isinstance(result, list)
    assert len(result) == 4  # All operations should be added

    # --- CORRECTED ASSERTIONS ---
    # Verify individual operations are present and correct
    assert (
        isinstance(result[0], PushOperation)
        and result[0].field_path == "maybe_list"
        and result[0].items == ["item1"]
    )
    assert (
        isinstance(result[1], PopOperation)
        and result[1].field_path == "maybe_list"
        and result[1].position == 1
    )
    assert (
        isinstance(result[2], PullOperation)
        and result[2].field_path == "maybe_list"
        and result[2].value_or_condition == "item_to_remove"
    )
    assert (
        isinstance(result[3], PopOperation)
        and result[3].field_path == "maybe_list"
        and result[3].position == -1
    )

    # Alternatively, using the helpers more carefully:
    push_ops = find_operations(result, PushOperation, "maybe_list")
    assert len(push_ops) == 1
    assert push_ops[0].items == ["item1"]

    pop_ops = find_operations(result, PopOperation, "maybe_list")
    assert len(pop_ops) == 2
    assert pop_ops[0].position == 1  # First pop added
    assert pop_ops[1].position == -1  # Second pop added

    pull_ops = find_operations(result, PullOperation, "maybe_list")
    assert len(pull_ops) == 1
    assert pull_ops[0].value_or_condition == "item_to_remove"
    # --- END CORRECTED ASSERTIONS ---

    # Test pushing invalid type - should fail validation against inner str type
    with pytest.raises(
        ValueTypeError, match="Invalid value type for push to 'maybe_list'"
    ):
        update.push("maybe_list", 123)

    # Test popping/pulling from a non-list field still fails
    with pytest.raises(InvalidPathError):
        update.pop("another_field")
    with pytest.raises(InvalidPathError):
        update.pull("another_field", 0)
