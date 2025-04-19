# tests/base/update/test_complex_types.py

import pytest
from typing import List, Optional, Type, TypeVar, Union # Added imports

# Import Update builder and relevant components
from async_repository.base.update import (
    Update,
    UpdateOperation,
    PushOperation,
    PopOperation,
    PullOperation,
    InvalidPathError,
    ValueTypeError,
)
# Import necessary conftest models if needed, or define test models here
# from .conftest import ... # If using shared models


# --- Test Helper (Copied or Imported) ---
OpT = TypeVar('OpT', bound=UpdateOperation)

def find_operation(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str
) -> Optional[OpT]:
    """Finds the first operation of a specific type and field path."""
    for op in operations:
        if isinstance(op, op_type) and op.field_path == field_path:
            return op
    return None

# Helper to find *all* operations for a path
def find_operations(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: Optional[str] = None # Optional field path filtering
) -> List[OpT]:
    """Finds all operations of a specific type, optionally filtered by field path."""
    found = []
    for op in operations:
        if isinstance(op, op_type):
             if field_path is None or op.field_path == field_path:
                found.append(op)
    return found

def assert_operation_present(
    operations: List[UpdateOperation],
    op_type: Type[OpT],
    field_path: str,
    expected_attrs: Optional[dict] = None # Check specific attributes like value, amount
):
    """Asserts that a specific operation exists and optionally checks its attributes."""
    # This helper still finds the first match. Use find_operations for multiple.
    op = find_operation(operations, op_type, field_path)
    assert op is not None, f"{op_type.__name__} for field '{field_path}' not found in {operations}"
    if expected_attrs:
        for attr, expected_value in expected_attrs.items():
            assert hasattr(op, attr), f"Operation {op!r} missing attribute '{attr}'"
            actual_value = getattr(op, attr)
            if isinstance(expected_value, float):
                 import pytest
                 assert actual_value == pytest.approx(expected_value), \
                     f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
            else:
                 if isinstance(expected_value, (dict, list)) and isinstance(actual_value, (dict, list)):
                     assert actual_value == expected_value, \
                        f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                 elif attr == 'items' and isinstance(expected_value, list) and isinstance(actual_value, list):
                     assert actual_value == expected_value, \
                        f"Attribute 'items' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                 else:
                     assert actual_value == expected_value, \
                        f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
# --- End Test Helper ---


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
    update.push("maybe_list", "item1")         # Op 0
    update.pop("maybe_list")                   # Op 1 (position=1)
    update.pull("maybe_list", "item_to_remove")# Op 2
    update.pop("maybe_list", -1)               # Op 3 (position=-1)

    result = update.build()
    assert isinstance(result, list)
    assert len(result) == 4 # All operations should be added

    # --- CORRECTED ASSERTIONS ---
    # Verify individual operations are present and correct
    assert isinstance(result[0], PushOperation) and result[0].field_path == "maybe_list" and result[0].items == ["item1"]
    assert isinstance(result[1], PopOperation) and result[1].field_path == "maybe_list" and result[1].position == 1
    assert isinstance(result[2], PullOperation) and result[2].field_path == "maybe_list" and result[2].value_or_condition == "item_to_remove"
    assert isinstance(result[3], PopOperation) and result[3].field_path == "maybe_list" and result[3].position == -1

    # Alternatively, using the helpers more carefully:
    push_ops = find_operations(result, PushOperation, "maybe_list")
    assert len(push_ops) == 1
    assert push_ops[0].items == ["item1"]

    pop_ops = find_operations(result, PopOperation, "maybe_list")
    assert len(pop_ops) == 2
    assert pop_ops[0].position == 1 # First pop added
    assert pop_ops[1].position == -1 # Second pop added

    pull_ops = find_operations(result, PullOperation, "maybe_list")
    assert len(pull_ops) == 1
    assert pull_ops[0].value_or_condition == "item_to_remove"
    # --- END CORRECTED ASSERTIONS ---

    # Test pushing invalid type - should fail validation against inner str type
    with pytest.raises(ValueTypeError, match="Invalid value type for push to 'maybe_list'"):
         update.push("maybe_list", 123)

    # Test popping/pulling from a non-list field still fails
    with pytest.raises(InvalidPathError):
        update.pop("another_field")
    with pytest.raises(InvalidPathError):
        update.pull("another_field", 0)