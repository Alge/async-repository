# tests/base/query/test_query_builder.py

import pytest
from pydantic import BaseModel
from typing import List, Optional, Set, Tuple, Union, Callable, Any
import re # For flexible error message matching

# Import necessary components
from async_repository.base.query import QueryBuilder, Field, QueryOptions, FilterCondition, CombinedCondition, Expression
from async_repository.base.model_validator import ValueTypeError, InvalidPathError


# --- Test Model Definition ---

class Address(BaseModel):
    street: str
    zip_code: int
    tags: Optional[List[str]] = None

class User(BaseModel):
    name: str
    age: int
    score: Optional[float] = None
    is_active: bool
    roles: List[str] = []
    address: Optional[Address] = None
    tags: Set[str] = set()
    settings: Tuple[str, int] = ("default", 0)
    permissions: Tuple[str, ...] = ()


# --- Fixtures ---

@pytest.fixture
def qb() -> QueryBuilder:
    """Provides a QueryBuilder instance with the User model."""
    return QueryBuilder(model=User)

@pytest.fixture
def qb_no_model() -> QueryBuilder:
    """Provides a QueryBuilder instance without a model."""
    return QueryBuilder(model=None)

# --- Helper Function for Creating Conditions ---

def _create_filter_condition(field_obj: Field, op_name: str, value: Any) -> FilterCondition:
    """Creates a FilterCondition based on a string operator name."""
    op_map: Dict[str, Callable[[Any], FilterCondition]] = {
        'eq': field_obj.__eq__,
        'ne': field_obj.ne,
        'gt': field_obj.__gt__,
        'lt': field_obj.__lt__,
        'ge': field_obj.__ge__,
        'le': field_obj.__le__,
        'in': field_obj.in_,
        'nin': field_obj.nin,
        'like': field_obj.like,
        'contains': field_obj.contains,
        'startswith': field_obj.startswith,
        'endswith': field_obj.endswith,
        'exists': field_obj.exists,
        'regex': field_obj.regex,
    }
    if op_name not in op_map:
        raise ValueError(f"Unsupported operation name for helper: {op_name}")
    return op_map[op_name](value)

# --- Tests for Basic Filter Operator Expression Generation ---

@pytest.mark.parametrize(
    "field_path, op_name, value, expected_value_in_expr",
    [
        # Equality & Inequality
        ("name",      "eq", "Alice",            "Alice"),
        ("age",       "ne", 30,                 30),
        # Comparisons
        ("age",       "gt", 20,                 20),
        ("age",       "lt", 50,                 50),
        ("age",       "ge", 21,                 21),
        ("age",       "le", 49,                 49),
        # Collection Membership (Note: Field methods convert input to list for expression)
        ("roles",     "in", ["admin", "editor"],["admin", "editor"]),
        ("tags",      "nin",{"guest", "temp"},  ["guest", "temp"]),
        ("age",       "in", (18, 21, 30),       [18, 21, 30]),
        # String Matching
        ("name",      "like", "%lic%",          "%lic%"),
        ("roles",     "contains","moderator",   "moderator"), # 'roles' is List[str]
        ("name",      "startswith","Al",        "Al"),
        ("name",      "endswith", "ice",        "ice"),
        ("name",      "regex", r"^A.*e$",       r"^A.*e$"),
        # Existence
        ("score",     "exists", True,           True),    # Optional field
        ("address",   "exists", False,          False),   # Optional field
    ],
    ids=[ # Provide clear IDs for each test case
        "eq_str", "ne_int", "gt_int", "lt_int", "ge_int", "le_int",
        "in_list", "nin_set", "in_tuple",
        "like_str", "contains_list_item", "startswith_str", "endswith_str", "regex_str",
        "exists_true", "exists_false"
    ]
)
def test_single_filter_operator_expression(qb: QueryBuilder, field_path: str, op_name: str, value: Any, expected_value_in_expr: Any):
    """Verify that individual filter operators generate the correct expression dictionary."""
    # Dynamically get the Field object (handles nested paths like "address.zip_code")
    field_obj = qb
    for part in field_path.split('.'):
        field_obj = getattr(field_obj, part)

    condition = _create_filter_condition(field_obj, op_name, value)

    # Use a fresh builder for isolation in parametrized tests
    options = QueryBuilder(model=User).filter(condition).build()

    expected_expression = {field_path: {"operator": op_name, "value": expected_value_in_expr}}
    assert options.expression == expected_expression, f"Failed for {op_name} on {field_path}"


# --- Tests for Combined Filter Expressions ---

def test_combined_and_expression(qb: QueryBuilder):
    """Verify combining two conditions with '&' generates an 'and' expression."""
    condition1 = qb.name == "Alice"
    condition2 = qb.age > 25
    combined = condition1 & condition2
    options = QueryBuilder(model=User).filter(combined).build() # Fresh builder
    expected = {
        "and": [
            {"name": {"operator": "eq", "value": "Alice"}},
            {"age": {"operator": "gt", "value": 25}},
        ]
    }
    assert options.expression == expected

def test_combined_or_expression(qb: QueryBuilder):
    """Verify combining two conditions with '|' generates an 'or' expression."""
    condition1 = qb.roles.in_(["admin", "editor"])
    condition2 = qb.is_active == False
    combined = condition1 | condition2
    options = QueryBuilder(model=User).filter(combined).build() # Fresh builder
    expected = {
        "or": [
            {"roles": {"operator": "in", "value": ["admin", "editor"]}},
            {"is_active": {"operator": "eq", "value": False}},
        ]
    }
    assert options.expression == expected

def test_multiple_sequential_filters_expression(qb: QueryBuilder):
    """Verify that chaining .filter() calls combines them with 'and'."""
    options = QueryBuilder(model=User).filter(qb.name == "Bob").filter(qb.age < 40).build()
    # Expects structure like: {"and": [{"name": ...}, {"age": ...}]}
    assert "and" in options.expression
    assert len(options.expression["and"]) == 2
    assert options.expression["and"][0] == {"name": {"operator": "eq", "value": "Bob"}}
    assert options.expression["and"][1] == {"age": {"operator": "lt", "value": 40}}


# --- Tests for Query Options ---

@pytest.mark.parametrize(
    "option_method, value, expected_attr",
    [
        ("limit", 50, "limit"),
        ("offset", 10, "offset"),
        ("set_timeout", 5.5, "timeout"),
    ]
)
def test_pagination_and_timeout_options(qb: QueryBuilder, option_method: str, value: Any, expected_attr: str):
    """Verify setting limit, offset, and timeout options correctly."""
    builder_method = getattr(qb, option_method)
    options = builder_method(value).build()
    assert getattr(options, expected_attr) == value

def test_sort_by_ascending_option(qb: QueryBuilder):
    """Verify setting ascending sort order."""
    options = qb.sort_by("age").build()
    assert options.sort_by == "age"
    assert options.sort_desc is False

def test_sort_by_descending_option(qb: QueryBuilder):
    """Verify setting descending sort order."""
    options = qb.sort_by("name", descending=True).build()
    assert options.sort_by == "name"
    assert options.sort_desc is True

def test_random_order_option(qb: QueryBuilder):
    """Verify setting random order overrides sort_by."""
    options = qb.sort_by("name").random_order().build()
    assert options.random_order is True
    assert options.sort_by is None

def test_combined_query_options(qb: QueryBuilder):
    """Verify setting multiple different options together."""
    options = (
        qb.filter(qb.is_active == True)
        .sort_by("score", descending=True)
        .limit(25)
        .offset(5)
        .set_timeout(10.0)
        .build()
    )
    # Check multiple attributes on the final QueryOptions object
    assert options.expression == {"is_active": {"operator": "eq", "value": True}}
    assert options.sort_by == "score"
    assert options.sort_desc is True
    assert options.limit == 25
    assert options.offset == 5
    assert options.timeout == 10.0
    assert options.random_order is False


# --- Tests for Field Access and Validation ---

def test_valid_field_access_returns_field_object(qb: QueryBuilder):
    """Verify accessing valid fields returns Field instances with correct names."""
    assert isinstance(qb.name, Field) and qb.name.name == "name"
    assert isinstance(qb.address.street, Field) and qb.address.street.name == "address.street"

def test_invalid_field_access_raises_attribute_error(qb: QueryBuilder):
    """Verify accessing non-existent fields raises AttributeError appropriately."""
    # Top-level invalid field should raise immediately from QueryBuilder.__getattr__
    with pytest.raises(AttributeError, match=re.escape("'User' has no attribute 'nonexistent_field'.")):
        _ = qb.nonexistent_field

    # Nested invalid field access returns a Field object initially...
    invalid_nested_field = qb.address.city
    assert isinstance(invalid_nested_field, Field) and invalid_nested_field.name == "address.city"

    # ...but using it in filter raises ValueError (wrapping InvalidPathError)
    with pytest.raises(ValueError, match=re.escape("Invalid field path used in filter: Field 'city' does not exist in type Address")):
        QueryBuilder(model=User).filter(invalid_nested_field == "SomeCity").build()

    # ...and using it in sort_by raises AttributeError
    with pytest.raises(AttributeError, match=re.escape("'User' has no sortable attribute or valid path 'address.city'.")):
        QueryBuilder(model=User).sort_by("address.city").build()


# --- Tests for Value Validation during Filtering ---

def test_filter_with_valid_values_succeeds(qb: QueryBuilder):
    """Check that filtering with type-correct values does not raise validation errors."""
    # Use QueryBuilder directly to ensure test isolation
    try:
        QueryBuilder(model=User).filter(qb.name == "Valid Name").build()
        QueryBuilder(model=User).filter(qb.age > 18).build()
        QueryBuilder(model=User).filter(qb.score == 95.5).build()
        QueryBuilder(model=User).filter(qb.score == None).build()
        QueryBuilder(model=User).filter(qb.is_active == True).build()
        QueryBuilder(model=User).filter(qb.roles.contains("admin")).build()
        QueryBuilder(model=User).filter(qb.tags.in_({"admin", "user"})).build()
        QueryBuilder(model=User).filter(qb.address.zip_code == 90210).build()
        QueryBuilder(model=User).filter(qb.address == None).build()
        QueryBuilder(model=User).filter(qb.age.in_([20, 30, 40])).build()
        QueryBuilder(model=User).filter(qb.permissions.in_(("read", "write"))).build()
    except ValueError as e:
        pytest.fail(f"Valid filter operation raised unexpected ValueError: {e}")


@pytest.mark.parametrize(
    "field_path, op_name, invalid_value, error_fragment", # Use fragments for robustness
    [
        # Basic type mismatches
        ("name", "eq", 123, "expected type str or compatible dict, got 123 (type: int)"),
        ("age", "eq", "twenty", "expected type int or compatible dict, got 'twenty' (type: str)"),
        ("is_active", "eq", "true", "expected type bool or compatible dict, got 'true' (type: str)"),
        ("score", "eq", True, "expected type float or compatible dict, got True (type: bool)"),
        # Invalid item types for collections
        ("roles", "contains", 123, "Value for 'roles' contains': expected type str or compatible dict, got 123 (type: int)"),
        ("age", "in", [10, "twenty", 30], "'age' item 1 for 'in'': expected type int or compatible dict, got 'twenty' (type: str)"),
        ("tags", "in", ["valid", 123], "'tags' item 1 for 'in'': expected type str or compatible dict, got 123 (type: int)"),
        # Operator type mismatches
        ("age", "gt", "twenty", "requires a numeric value (int/float) for comparison, got str"),
        ("name", "gt", "A", "requires a numeric field, but 'name' is not"),
        ("name", "contains", "lice", "requires a List field, but 'name' is not"),
    ],
    ids=[
        "eq_str_int", "eq_int_str", "eq_bool_str", "eq_float_bool",
        "contains_list_int", "in_int_str", "in_set_int",
        "gt_val_str", "gt_field_str", "contains_field_str"
    ]
)
def test_filter_with_invalid_values_raises_value_error(field_path: str, op_name: str, invalid_value: Any, error_fragment: str):
    """Verify filtering with incorrect types or inapplicable operators raises ValueError."""
    # Use a fresh builder instance for each parametrized test run
    qb_instance = QueryBuilder(model=User)
    field_obj = qb_instance
    for part in field_path.split('.'):
        field_obj = getattr(field_obj, part)

    condition = _create_filter_condition(field_obj, op_name, invalid_value)

    # Expect ValueError wrapping the underlying validation error
    with pytest.raises(ValueError) as excinfo:
        qb_instance.filter(condition).build()

    # Check that the expected fragment is present in the error message
    assert error_fragment in str(excinfo.value)


@pytest.mark.parametrize(
    "field_name, method_name, invalid_arg, error_pattern",
    [
        ("age", "in_", 123, r"requires a list, set, or tuple, got <class 'int'>"),
        ("score", "exists", "yes", r"requires a boolean value, got <class 'str'>"),
        ("name", "startswith", 123, r"requires a string substring, got <class 'int'>"),
        ("name", "like", True, r"requires a string pattern, got <class 'bool'>"), # Use bool for variety
        ("name", "endswith", None, r"requires a string substring, got <class 'NoneType'>"),
        ("name", "regex", [], r"requires a string pattern, got <class 'list'>"),
    ],
    ids=["in_int", "exists_str", "startswith_int", "like_bool", "endswith_none", "regex_list"]
)
def test_field_method_direct_type_error(qb: QueryBuilder, field_name: str, method_name: str, invalid_arg: Any, error_pattern: str):
    """Verify that Field methods raise TypeError directly for invalid argument types (before filter validation)."""
    field_obj = qb
    for part in field_name.split('.'):
        field_obj = getattr(field_obj, part)

    method_to_call = getattr(field_obj, method_name)
    with pytest.raises(TypeError, match=error_pattern):
        method_to_call(invalid_arg)


# --- Tests for Builder without Model ---

def test_no_model_field_access_succeeds(qb_no_model: QueryBuilder):
    """Verify field access works without validation when no model is provided."""
    assert qb_no_model.any_field.name == "any_field"
    nested_field = qb_no_model.nested.path.here
    assert isinstance(nested_field, Field) and nested_field.name == "nested.path.here"

def test_no_model_filter_succeeds(qb_no_model: QueryBuilder):
    """Verify filtering works without validation when no model is provided."""
    options = qb_no_model.filter(qb_no_model.status == "active").build()
    expected = {"status": {"operator": "eq", "value": "active"}}
    assert options.expression == expected

def test_no_model_sort_succeeds(qb_no_model: QueryBuilder):
    """Verify sorting works without validation when no model is provided."""
    options = qb_no_model.sort_by("some_field").build()
    assert options.sort_by == "some_field"


# --- Edge Case Test ---

def test_build_without_filters_or_options(qb: QueryBuilder):
    """Verify building QueryOptions with defaults when no methods are called."""
    options = qb.build()
    assert options.expression == {}
    assert options.sort_by is None
    assert options.sort_desc is False
    assert options.limit == 100 # Default
    assert options.offset == 0   # Default
    assert options.timeout is None
    assert options.random_order is False