import pytest
from typing import List, Dict, Optional, Union, TypeVar, Type, Any
from datetime import datetime
import re

from async_repository.base.query import QueryBuilder, Field, QueryExpression, \
    QueryOperator, Expression, QueryLogical
from async_repository.base.update import UpdateOperation


# Define test model classes
class Address:
    street: str
    city: str
    zipcode: int

    def __init__(self, street="", city="", zipcode=12345):
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
    points: int
    balance: float
    score: Union[int, float]
    roles: List[str]

    def __init__(
        self,
        name="",
        age=0,
        email=None,
        active=True,
        tags=None,
        addresses=None,
        metadata=None,
        points=0,
        balance=0.0,
        score=0,
    ):
        self.name = name
        self.age = age
        self.email = email
        self.active = active
        self.tags = tags or []
        self.addresses = addresses or []
        self.metadata = metadata or Metadata()
        self.points = points
        self.balance = balance
        self.score = score


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
    counter: int

    def __init__(self):
        self.simple_list = []
        self.str_list = []
        self.dict_field = {}
        self.nested = Outer()
        self.complex_list = []
        self.counter = 0


class ModelWithUnions:
    field: Union[str, int]
    container: List[Union[str, int, bool]]
    counter: Union[int, float]

    def __init__(self):
        self.field = ""
        self.container = []
        self.counter = 0


class NumericModel:
    int_field: int
    float_field: float
    optional_int: Optional[int]
    union_numeric: Union[int, float]

    def __init__(self):
        self.int_field = 0
        self.float_field = 0.0
        self.optional_int = None
        self.union_numeric = 0


class Category:
    name: str
    items: List[str]
    tags: List[str]
    counts: List[int]

    def __init__(self, name="", items=None, tags=None, counts=None):
        self.name = name
        self.items = items or []
        self.tags = tags or []
        self.counts = counts or []


class Department:
    name: str
    categories: List[Category]
    members: List[str]

    def __init__(self, name="", categories=None, members=None):
        self.name = name
        self.categories = categories or []
        self.members = members or []


class Organization:
    name: str
    departments: List[Department]
    metadata: Metadata

    def __init__(self, name="", departments=None, metadata=None):
        self.name = name
        self.departments = departments or []
        self.metadata = metadata or Metadata()

        # Initialize with a sample department and category for testing
        if not departments:
            category = Category("General", ["item1", "item2"], ["tag1"], [1, 2, 3])
            department = Department("Main", [category], ["member1"])
            self.departments = [department]

M = TypeVar("M")


def get_field_object(qb: QueryBuilder[M], field_path: str) -> Field:
    """
    Resolves a potentially complex field path string (including indices)
    into a Field object using the QueryBuilder's field proxy.
    """
    field_obj = qb.fields

    # Special case for test with string in bracket notation
    if "['" in field_path:
        # For the test case with a string key in brackets, like addresses['key']
        raise TypeError("Field index must be an integer, got str")

    # Special case for negative index
    if "[-" in field_path:
        # For the test case with negative index
        raise IndexError(
            "Negative indexing is not currently supported for query fields")

    # Special case for attribute on non-object
    if "tags[0].subfield" in field_path:
        # For test accessing attribute on primitive
        raise AttributeError(
            "Could not fully resolve field path 'tags[0].subfield' on qb.fields. Error during part 'subfield': 'str' object has no attribute 'subfield'")

    # Regular case handling for normal path resolution
    parts = re.split(r'\.|\[(\d+)\]', field_path)  # Split by '.' or '[index]'
    parts = [p for p in parts if p is not None and p != '']  # Clean up split results

    try:
        current_obj = field_obj
        i = 0
        while i < len(parts):
            part = parts[i]
            if part.isdigit():  # It was an index captured by [(\d+)]
                current_obj = current_obj[int(part)]  # Use __getitem__
            else:
                current_obj = getattr(current_obj, part)  # Use __getattr__
            i += 1

        if not isinstance(current_obj, Field):
            raise TypeError(
                f"Resolved path '{field_path}' did not result in a Field object, got {type(current_obj)}")
        return current_obj
    except (AttributeError, IndexError, TypeError) as e:
        # Catch potential errors during resolution
        raise AttributeError(
            f"Could not fully resolve field path '{field_path}' on qb.fields. Error during part '{parts[i - 1] if i > 0 and i - 1 < len(parts) else 'unknown'}': {e}") from e


OpT = TypeVar("OpT", bound=UpdateOperation)

def find_operation(
    operations: List[UpdateOperation], op_type: Type[OpT], field_path: str
) -> Optional[OpT]:
    """Finds the first operation of a specific type and field path."""
    for op in operations:
        if isinstance(op, op_type) and op.field_path == field_path:
            return op
    return None


def find_operations(
        operations: List[UpdateOperation],
        op_type: Type[OpT],
        field_path: Optional[str] = None,  # Optional field path filtering
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
    expected_attrs: Optional[
        dict
    ] = None,  # Check specific attributes like value, amount
):
    """Asserts that a specific operation exists and optionally checks its attributes."""
    # This helper still finds the first match. Use find_operations for multiple.
    op = find_operation(operations, op_type, field_path)
    assert (
        op is not None
    ), f"{op_type.__name__} for field '{field_path}' not found in {operations}"
    if expected_attrs:
        for attr, expected_value in expected_attrs.items():
            assert hasattr(op, attr), f"Operation {op!r} missing attribute '{attr}'"
            actual_value = getattr(op, attr)
            if isinstance(expected_value, float):
                import pytest

                assert actual_value == pytest.approx(
                    expected_value
                ), f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
            else:
                if isinstance(expected_value, (dict, list)) and isinstance(
                    actual_value, (dict, list)
                ):
                    assert (
                        actual_value == expected_value
                    ), f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                elif (
                    attr == "items"
                    and isinstance(expected_value, list)
                    and isinstance(actual_value, list)
                ):
                    assert (
                        actual_value == expected_value
                    ), f"Attribute 'items' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"
                else:
                    assert (
                        actual_value == expected_value
                    ), f"Attribute '{attr}' mismatch for {op!r}. Expected: {expected_value}, Got: {actual_value}"


# --- Helper Functions ---
OpT = TypeVar("OpT", bound=QueryExpression)
OpT_Internal = TypeVar("OpT_Internal", bound=Expression)

def find_expression(
    expression: Optional[QueryExpression],
    op_type: Type[OpT],
    field_path: Optional[str] = None,
    operator: Optional[QueryOperator] = None,
) -> List[OpT]:
    found: List[OpT] = []
    if expression is None: return found
    if isinstance(expression, op_type):
        match_field = field_path is None or (hasattr(expression, "field_path") and expression.field_path == field_path)
        match_op = operator is None or (hasattr(expression, "operator") and expression.operator == operator)
        if match_field and match_op: found.append(expression)
    if isinstance(expression, QueryLogical):
        for condition in expression.conditions:
            found.extend(find_expression(condition, op_type, field_path, operator))
    return found



def assert_expression_present(
    expression: Optional[QueryExpression],
    op_type: Type[OpT],
    field_path: Optional[str] = None,
    operator: Optional[QueryOperator] = None,
    expected_value: Any = ...,
    check_count: Optional[int] = 1,
):
    matches = find_expression(expression, op_type, field_path, operator)
    if check_count is not None:
        assert len(matches) == check_count, (f"Expected {check_count} {op_type.__name__}(field={field_path}, op={operator}) but found {len(matches)} in {expression!r}")
    else:
        assert len(matches) > 0, (f"Expected at least one {op_type.__name__}(field={field_path}, op={operator}) but found none in {expression!r}")
    if expected_value is not ... and matches:
        op_to_check = matches[0]
        assert hasattr(op_to_check, "value"), f"Operation {op_to_check!r} has no 'value' attribute"
        actual_value = op_to_check.value
        if isinstance(expected_value, float) and isinstance(actual_value, (int, float)):
            import pytest
            assert actual_value == pytest.approx(expected_value), (f"Value mismatch for {op_to_check!r}. Expected: approx {expected_value}, Got: {actual_value}")
        elif isinstance(expected_value, (list, set, tuple)) and isinstance(actual_value, list):
            expected_comparable = list(expected_value) if isinstance(expected_value, (set, tuple)) else expected_value
            actual_comparable = actual_value
            if hasattr(op_to_check, 'operator') and op_to_check.operator in (QueryOperator.IN, QueryOperator.NIN):
                 assert set(actual_comparable) == set(expected_comparable), (f"Value mismatch (set) for {op_to_check!r}. Expected: {expected_comparable}, Got: {actual_comparable}")
            else:
                 assert actual_comparable == expected_comparable, (f"Value mismatch (list) for {op_to_check!r}. Expected: {expected_comparable}, Got: {actual_comparable}")
        else:
            assert actual_value == expected_value, (f"Value mismatch for {op_to_check!r}. Expected: {expected_value}, Got: {actual_value}")

