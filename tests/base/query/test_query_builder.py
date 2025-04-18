# tests/base/query/test_query_builder.py

import pytest
from pydantic import BaseModel
from typing import List, Optional, Set, Tuple, Callable, Any, Dict, Type, TypeVar
import re  # For flexible error message matching
import copy # For deepcopy in test fix
import operator # For helper function
import traceback # For better error reporting in valid tests

# Assuming your QueryBuilder and related classes are in this path
from async_repository.base.query import (
    QueryBuilder,
    Field,
    QueryOptions,
    FilterCondition,
    CombinedCondition,
    InvalidPathError,
    ValueTypeError,
    _PROXY_CACHE, GenericFieldsProxy,  # Import for cache clearing
)

M = TypeVar('M')

# --- Test Model Definition (Using Pydantic) ---

class Address(BaseModel):
    street: Optional[str] = None
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

@pytest.fixture(autouse=True)  # Apply to all tests in this module
def clear_proxy_cache():
    """Clears the proxy cache before each test."""
    _PROXY_CACHE.clear()
    yield  # Run the test
    _PROXY_CACHE.clear()  # Clear after test too


@pytest.fixture
def user_model() -> Type[User]:
    """Provides the User model class."""
    return User


@pytest.fixture
def qb(user_model: Type[User]) -> QueryBuilder[User]:
    """Provides a QueryBuilder instance for the User model."""
    return QueryBuilder(user_model)


# --- Helper Functions ---

def _get_field_object(qb: QueryBuilder[M], field_path: str) -> Field:
    """Dynamically gets the Field object from the QueryBuilder."""
    field_obj = qb.fields
    try:
        for part in field_path.split('.'):
            field_obj = getattr(field_obj, part)
        if not isinstance(field_obj, Field):
            raise TypeError(
                f"Resolved path '{field_path}' did not result in a Field "
                f"object, got {type(field_obj).__name__}."
            )
        return field_obj
    except AttributeError as e:
        raise AttributeError(
            f"Could not resolve field path '{field_path}' on qb.fields. "
            f"Original error: {e}"
        ) from e


def _create_filter_condition(qb: QueryBuilder[M], field_path: str, op_name: str,
                             value: Any) -> FilterCondition:
    """Creates a FilterCondition using the Field object and operator name."""
    field_obj = _get_field_object(qb, field_path)

    # Map operator names to Field methods
    op_map: Dict[str, Callable[..., FilterCondition]] = {
        'eq': field_obj.__eq__, 'ne': field_obj.__ne__, 'gt': field_obj.__gt__,
        'lt': field_obj.__lt__, 'ge': field_obj.__ge__, 'le': field_obj.__le__,
        'in': field_obj.in_, 'nin': field_obj.nin, 'like': field_obj.like,
        'contains': field_obj.contains, 'startswith': field_obj.startswith,
        'endswith': field_obj.endswith, 'exists': field_obj.exists,
    }
    if op_name not in op_map:
        raise ValueError(f"Unsupported operation name for helper: {op_name}")

    # Call the appropriate method on the Field object
    if op_name == 'exists':
        return op_map[op_name](value)
    else:
        return op_map[op_name](value)


def _get_field_name_from_condition(condition_dict):
    """Extracts the field name (key) from a simple condition dictionary."""
    if isinstance(condition_dict, dict) and len(condition_dict) == 1:
        return list(condition_dict.keys())[0]
    return str(condition_dict) # Fallback


# --- Test Parametrization Data ---
# Contains parameters for test_filter_with_invalid_values_raises_error
invalid_values_params = [
    # Basic type mismatches
    ("name", "eq", 123, ValueError, r"Path 'name': expected type str, got 123 \(int\)"),
    ("age", "eq", "twenty", ValueError, r"Path 'age': expected type int, got 'twenty' \(str\)"),
    ("is_active", "eq", "true", ValueError, r"Path 'is_active': expected type bool, got 'true' \(str\)"),
    ("score", "eq", True, ValueError, r"Path 'score': expected type float, got True \(bool\)"),
    ("address.zip_code", "eq", "90210", ValueError, r"Path 'address.zip_code': expected type int, got '90210' \(str\)"),
    # Invalid item types for collections
    ("roles", "contains", 123, ValueError, r"Operator 'contains' on field 'roles' requires value compatible with item type str, got int"),
    ("tags", "contains", 123, ValueError, r"Operator 'contains' on field 'tags' requires value compatible with item type str, got int"),
    ("roles", "in", ["a", 1], ValueError, r"Invalid item in 'in' list for field 'roles'.*Path 'roles \(in item \d+\)': expected type str, got 1 \(int\)"),
    ("tags", "nin", {"a", 1}, ValueError, r"Invalid item in 'nin' list for field 'tags'.*Path 'tags \(nin item \d+\)': expected type str, got 1 \(int\)"),
    ("age", "in", [20, "30"], ValueError, r"Invalid item in 'in' list for field 'age'.*Path 'age \(in item \d+\)': expected type int, got '30' \(str\)"),
    # Operator-specific value type mismatches (TypeError from Field._op)
    ("name", "like", 123, TypeError, r"Operator 'like' requires a string value"),
    ("name", "startswith", True, TypeError, r"Operator 'startswith' requires a string value"),
    ("age", "in", 123, TypeError, r"Operator 'in' requires a list/set/tuple"),
    ("score", "exists", "maybe", TypeError, r"Operator 'exists' requires a boolean value"),
    # Operator vs Field Type Mismatches (ValueError from _validate_expression_runtime)
    ("age", "like", "%test%", ValueError, r"Path 'age': expected type int, got '%test%' \(str\)"), # Value error first
    ("roles", "startswith", "adm", ValueError, r"Path 'roles': expected type list\[str\], got 'adm' \(str\)"), # Value error first, match formatted type name
    ("age", "gt", "10", ValueError, r"Path 'age': expected type int, got '10' \(str\)" ), # Value error first
    # Null checks
    ("name", "eq", None, ValueError, r"Path 'name': expected type str, got None \(NoneType\)"),
    ("age", "gt", None, ValueError, r"Path 'age': expected type int, got None \(NoneType\)"),
]

# Corresponding IDs for the parametrized tests
invalid_values_ids = [
    "eq_str_int", "eq_int_str", "eq_bool_str", "eq_float_bool", "eq_nested_int_str",
    "contains_list_int", "contains_set_int_check",
    "in_list_str_int", "nin_set_str_int", "in_int_list_str",
    "like_val_int", "startswith_val_bool", "in_val_int", "exists_val_str",
    "like_field_int", "startswith_field_list",
    # "gt_field_str_val_str_check", # ID Removed as test case was removed (valid scenario)
    "gt_field_int_val_str",
    "eq_nonoptional_none", "gt_nonoptional_none",
]


# --- Tests ---

@pytest.mark.parametrize(
    "field_path, op_name, value, expected_value_in_expr",
    [
        ("name", "eq", "Alice", "Alice"),
        ("age", "ne", 30, 30),
        ("age", "gt", 20, 20),
        ("age", "lt", 50, 50),
        ("age", "ge", 21, 21),
        ("age", "le", 49, 49),
        ("roles", "in", ["admin", "editor"], ["admin", "editor"]),
        ("tags", "nin", {"guest", "temp"}, ["guest", "temp"]),
        ("age", "in", (18, 21, 30), [18, 21, 30]),
        ("name", "like", "%lic%", "%lic%"),
        ("roles", "contains", "moderator", "moderator"),
        ("name", "startswith", "Al", "Al"),
        ("name", "endswith", "ice", "ice"),
        ("score", "exists", True, True),
        ("address", "exists", False, False),
        ("address.street", "exists", True, True),
    ],
    ids=[
        "eq_str", "ne_int", "gt_int", "lt_int", "ge_int", "le_int",
        "in_list", "nin_set", "in_tuple",
        "like_str", "contains_list_item", "startswith_str", "endswith_str",
        "exists_true_optional_float", "exists_false_optional_model",
        "exists_true_nested_optional_str"
    ]
)
def test_single_filter_operator_expression(user_model: Type[User], field_path: str,
                                           op_name: str, value: Any,
                                           expected_value_in_expr: Any):
    """Verify individual filter operators generate correct expression dict."""
    qb_instance = QueryBuilder(user_model)
    condition = _create_filter_condition(qb_instance, field_path, op_name, value)
    options = qb_instance.filter(condition).build()

    expected_expression = {
        field_path: {"operator": op_name, "value": expected_value_in_expr}
    }
    actual_expression = options.expression

    # Order-insensitive check for in/nin with set/tuple input
    if op_name in ("in", "nin") and isinstance(value, (set, tuple)):
        actual_expr_copy = copy.deepcopy(actual_expression)
        expected_expr_copy = copy.deepcopy(expected_expression)
        try:
            actual_list = actual_expr_copy[field_path]['value']
            expected_list = expected_expr_copy[field_path]['value']
        except (KeyError, TypeError) as e:
            pytest.fail(f"Error extracting lists for {op_name}: {e}")
        assert isinstance(actual_list, list) and isinstance(expected_list, list)
        assert set(actual_list) == set(expected_list), "List items mismatch"
        assert len(actual_list) == len(expected_list), "List lengths mismatch"
        del actual_expr_copy[field_path]['value']
        del expected_expr_copy[field_path]['value']
        assert actual_expr_copy == expected_expr_copy, "Structure mismatch"
    else:
        assert actual_expression == expected_expression, "Expression mismatch"


def test_combined_and_expression(qb: QueryBuilder[User]):
    """Verify combining two conditions with '&' generates 'and' expression."""
    qb_instance = QueryBuilder(qb.model_cls)
    condition1 = qb_instance.fields.name == "Alice"
    condition2 = qb_instance.fields.age > 25
    combined = condition1 & condition2
    options = qb_instance.filter(combined).build()
    expected = {
        "and": [
            {"name": {"operator": "eq", "value": "Alice"}},
            {"age": {"operator": "gt", "value": 25}},
        ]
    }
    actual_expression = options.expression
    assert "and" in actual_expression
    assert isinstance(actual_expression.get("and"), list)
    assert len(actual_expression.get("and", [])) == len(expected.get("and", []))
    actual_sorted_list = sorted(actual_expression.get("and", []), key=_get_field_name_from_condition)
    expected_sorted_list = sorted(expected.get("and", []), key=_get_field_name_from_condition)
    assert actual_sorted_list == expected_sorted_list


def test_combined_or_expression(qb: QueryBuilder[User]):
    """Verify combining two conditions with '|' generates 'or' expression."""
    qb_instance = QueryBuilder(qb.model_cls)
    condition1 = qb_instance.fields.roles.in_(["admin", "editor"])
    condition2 = qb_instance.fields.is_active == False
    combined = condition1 | condition2
    options = qb_instance.filter(combined).build()
    expected = {
        "or": [
            {"roles": {"operator": "in", "value": ["admin", "editor"]}},
            {"is_active": {"operator": "eq", "value": False}},
        ]
    }
    actual_expression = options.expression
    assert "or" in actual_expression
    assert isinstance(actual_expression.get("or"), list)
    assert len(actual_expression.get("or", [])) == len(expected.get("or", []))
    actual_sorted_list = sorted(actual_expression.get("or", []), key=_get_field_name_from_condition)
    expected_sorted_list = sorted(expected.get("or", []), key=_get_field_name_from_condition)
    assert actual_sorted_list == expected_sorted_list


def test_multiple_sequential_filters_expression(qb: QueryBuilder[User]):
    """Verify chaining .filter() calls combines them with 'and'."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = (
        qb_instance
        .filter(qb_instance.fields.name == "Bob")
        .filter(qb_instance.fields.age < 40)
        .build()
    )
    expected_conds_list = [
        {"name": {"operator": "eq", "value": "Bob"}},
        {"age": {"operator": "lt", "value": 40}}
    ]
    actual_expression = options.expression
    assert "and" in actual_expression
    assert isinstance(actual_expression.get("and"), list)
    assert len(actual_expression.get("and", [])) == 2
    actual_sorted_list = sorted(actual_expression.get("and", []), key=_get_field_name_from_condition)
    expected_sorted_list = sorted(expected_conds_list, key=_get_field_name_from_condition)
    assert actual_sorted_list == expected_sorted_list


@pytest.mark.parametrize(
    "option_method_name, value, expected_attr",
    [("limit", 50, "limit"), ("offset", 10, "offset"), ("timeout", 5.5, "timeout")]
)
def test_pagination_and_timeout_options(user_model: Type[User], option_method_name: str,
                                        value: Any, expected_attr: str):
    """Verify setting limit, offset, and timeout options correctly."""
    qb_instance = QueryBuilder(user_model)
    builder_method = getattr(qb_instance, option_method_name)
    options = builder_method(value).build()
    assert getattr(options, expected_attr) == value


def test_sort_by_ascending_option(qb: QueryBuilder[User]):
    """Verify setting ascending sort order."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = qb_instance.sort_by(qb_instance.fields.age).build()
    assert options.sort_by == "age"
    assert options.sort_desc is False


def test_sort_by_descending_option(qb: QueryBuilder[User]):
    """Verify setting descending sort order."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = qb_instance.sort_by(qb_instance.fields.name, descending=True).build()
    assert options.sort_by == "name"
    assert options.sort_desc is True


def test_random_order_option(qb: QueryBuilder[User]):
    """Verify setting random order overrides sort_by."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = qb_instance.sort_by(qb_instance.fields.name).random_order().build()
    assert options.random_order is True
    assert options.sort_by is None
    assert options.sort_desc is False


def test_combined_query_options(qb: QueryBuilder[User]):
    """Verify setting multiple different options together."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = (
        qb_instance
        .filter(qb_instance.fields.is_active == True)
        .sort_by(qb_instance.fields.score, descending=True)
        .limit(25)
        .offset(5)
        .timeout(10.0)
        .build()
    )
    assert options.expression == {"is_active": {"operator": "eq", "value": True}}
    assert options.sort_by == "score"
    assert options.sort_desc is True
    assert options.limit == 25
    assert options.offset == 5
    assert options.timeout == 10.0
    assert options.random_order is False


def test_valid_field_access_returns_field_object(qb: QueryBuilder[User]):
    """Verify accessing valid fields returns Field instances with correct paths."""
    assert isinstance(qb.fields.name, Field)
    assert qb.fields.name.path == "name"
    assert isinstance(qb.fields.address, Field) # Now returns Field
    assert qb.fields.address.path == "address"
    assert isinstance(qb.fields.address.zip_code, Field) # Access through Field
    assert qb.fields.address.zip_code.path == "address.zip_code"


def test_invalid_field_access_raises_attribute_error(qb: QueryBuilder[User]):
    """Verify accessing non-existent top-level fields raises AttributeError."""
    # Use the error message generated by the SimpleNamespace proxy
    with pytest.raises(AttributeError, match=r"'?SimpleNamespace'? object has no attribute 'nonexistent_field'"):
        _ = qb.fields.nonexistent_field
    # Note: Nested invalid access (e.g., qb.fields.address.city) now returns a Field object
    # and the error is caught later during validation (filter/sort_by).


def test_invalid_field_path_in_filter_raises_value_error(user_model: Type[User]):
    """Verify using an invalid field path (constructed manually) in filter raises ValueError."""
    qb_instance = QueryBuilder(user_model)
    invalid_field = Field("address.city")
    # Expect the ValueError wrapping the InvalidPathError, match the full message
    expected_regex = (
        r"Invalid filter expression: Path 'address.city': "
        r"Cannot resolve part 'city' in path 'address\.city' for type Address\. "
        r"Available fields: \['street', 'zip_code', 'tags'\]\."
    )
    with pytest.raises(ValueError, match=expected_regex):
        qb_instance.filter(invalid_field == "SomeCity") # Error raised here


def test_invalid_field_path_in_sort_raises_attribute_error(user_model: Type[User]):
    """Verify using an invalid field path (via manual Field) in sort_by raises AttributeError."""
    qb_instance = QueryBuilder(user_model)
    invalid_field = Field("address.city")
     # Expect the AttributeError wrapping the InvalidPathError, match the full message
    expected_regex = (
        r"Invalid sort field path: address\.city\. Error: Path 'address.city': "
        r"Cannot resolve part 'city' in path 'address\.city' for type Address\. "
        r"Available fields: \['street', 'zip_code', 'tags'\]\."
    )
    with pytest.raises(AttributeError, match=expected_regex):
        qb_instance.sort_by(invalid_field) # Error raised here


def test_filter_with_valid_values_succeeds(qb: QueryBuilder[User]):
    """Check that filtering with type-correct values does not raise validation errors."""
    model = qb.model_cls
    fields = qb.fields # Use fields from the fixture qb
    try:
        QueryBuilder(model).filter(fields.name == "Valid Name").build()
        QueryBuilder(model).filter(fields.age > 18).build()
        QueryBuilder(model).filter(fields.score == 95.5).build()
        QueryBuilder(model).filter(fields.score == None).build()
        QueryBuilder(model).filter(fields.is_active == True).build()
        QueryBuilder(model).filter(fields.roles.contains("admin")).build()
        QueryBuilder(model).filter(fields.tags.contains("admin")).build()
        QueryBuilder(model).filter(fields.tags.in_({"admin", "user"})).build()
        QueryBuilder(model).filter(fields.address.zip_code == 90210).build()
        QueryBuilder(model).filter(fields.address == None).build()
        QueryBuilder(model).filter(fields.age.in_([20, 30, 40])).build()
        QueryBuilder(model).filter(fields.address.tags.contains("vip")).build()
        QueryBuilder(model).filter(fields.address.tags == None).build()
        QueryBuilder(model).filter(fields.tags.exists(True)).build()
        QueryBuilder(model).filter(fields.address.exists(True)).build()
        QueryBuilder(model).filter(fields.address.street == None).build()
        QueryBuilder(model).filter(fields.address.street.exists(False)).build()
    except (ValueError, TypeError) as e:
        pytest.fail(f"Valid filter operation raised unexpected error: {e}\n{traceback.format_exc()}")


@pytest.mark.parametrize(
    "field_path, op_name, invalid_value, error_type, error_fragment",
    invalid_values_params, # Use the corrected params list from above
    ids=invalid_values_ids
)
def test_filter_with_invalid_values_raises_error(user_model: Type[User],
                                                 field_path: str,
                                                 op_name: str,
                                                 invalid_value: Any,
                                                 error_type: Type[Exception],
                                                 error_fragment: str):
    """Verify filtering with incorrect types or inapplicable operators raises expected error."""
    qb_instance = QueryBuilder(user_model)

    with pytest.raises(error_type) as excinfo:
        condition = _create_filter_condition(qb_instance, field_path, op_name, invalid_value)
        qb_instance.filter(condition) # Error is raised here

    error_str = str(excinfo.value)
    # Check that the expected fragment is present in the error message
    assert re.search(error_fragment, error_str, re.IGNORECASE | re.DOTALL) is not None, \
        (f"Failed for op '{op_name}' on path '{field_path}' with value {invalid_value!r}.\n"
         f"  Expected error type: {error_type.__name__}\n"
         f"  Expected fragment (regex): r'{error_fragment}'\n"
         f"  Actual error message: '{error_str}'")


def test_build_without_filters_or_options(qb: QueryBuilder[User]):
    """Verify building QueryOptions with defaults when no methods are called."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = qb_instance.build()
    assert isinstance(options, QueryOptions)
    assert options.expression == {}
    assert options.sort_by is None
    assert options.sort_desc is False
    assert options.limit == 100
    assert options.offset == 0
    assert options.timeout is None
    assert options.random_order is False


def test_repr_methods(qb: QueryBuilder[User]):
    """Test the __repr__ methods for better debugging."""
    qb_instance = QueryBuilder(qb.model_cls)
    filter_cond = qb_instance.fields.name == "Test"
    assert repr(filter_cond) == "FilterCondition('name', 'eq', 'Test')"

    combined_cond = (qb_instance.fields.age > 18) & (qb_instance.fields.is_active == True)
    assert re.match(r"CombinedCondition\('and', .*, .*\)", repr(combined_cond))
    assert "FilterCondition('age', 'gt', 18)" in repr(combined_cond)
    assert "FilterCondition('is_active', 'eq', True)" in repr(combined_cond)

    field_obj = qb_instance.fields.address.zip_code
    assert re.match(r"Field(\[.*\])?\(path='address.zip_code'\)", repr(field_obj))

    options = qb_instance.filter(filter_cond).limit(10).build()
    options_repr = repr(options)
    assert options_repr.startswith("QueryOptions(")
    assert "expression={'name': {'operator': 'eq', 'value': 'Test'}}" in options_repr
    assert "limit=10" in options_repr
    assert "offset=0" in options_repr

# Example test for model-less builder (Optional)
# Test for model-less builder
def test_model_less_builder():
    """Verify builder works without a model (no validation)."""
    qb_generic = QueryBuilder() # No model class
    assert qb_generic._validator is None
    assert isinstance(qb_generic.fields, GenericFieldsProxy)

    # Accessing fields creates them dynamically
    field1 = qb_generic.fields.any_field
    assert isinstance(field1, Field)
    assert field1.path == "any_field"

    # --- FIX: Correctly test nested field creation and path ---
    # Access the full desired nested path first using attribute chaining
    deep_field = qb_generic.fields.level1.level2.final_name
    # Check the result is a Field instance
    assert isinstance(deep_field, Field)
    # Check the .path property of the *resulting* Field instance
    assert deep_field.path == "level1.level2.final_name"
    # --- END FIX ---


    # Filtering works without validation errors for paths/types
    try:
        options = (
            qb_generic
            .filter(qb_generic.fields.some_field == "value")
            .filter(qb_generic.fields.num_field > 100)
            # Use a nested field created dynamically in the filter
            .filter(qb_generic.fields.data.payload.status.contains("OK"))
            .sort_by(qb_generic.fields.num_field, descending=True)
            .limit(10)
            .build()
        )
        # Verify the expression structure (adjust based on how AND is nested)
        # This structure depends on ((a & b) & c)
        assert options.expression['and'][0]['and'][0] == {'some_field': {'operator': 'eq', 'value': 'value'}}
        assert options.expression['and'][0]['and'][1] == {'num_field': {'operator': 'gt', 'value': 100}}
        assert options.expression['and'][1] == {'data.payload.status': {'operator': 'contains', 'value': 'OK'}}

        assert options.sort_by == "num_field"
        assert options.sort_desc is True
        assert options.limit == 10

    except (ValueError, TypeError, AttributeError) as e: # Added AttributeError just in case
         pytest.fail(f"Model-less builder failed unexpectedly: {e}\n{traceback.format_exc()}")

    # Check that Field._op still raises TypeError for wrong value type for operator
    with pytest.raises(TypeError, match="Operator 'like' requires a string value"):
        qb_generic.filter(qb_generic.fields.foo.like(123))