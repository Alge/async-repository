import pytest
from pydantic import BaseModel
from typing import List, Optional, Set, Tuple, Callable, Any, Dict, Type, TypeVar
import re  # For flexible error message matching
import copy  # For deepcopy in test fix
import operator  # For helper function
import traceback  # For better error reporting in valid tests

from async_repository.base.query import (
    QueryBuilder,
    Field,
    QueryOptions,
    FilterCondition,
    CombinedCondition,  # Added import
    InvalidPathError,  # Added import
    ValueTypeError,  # Added import
    _PROXY_CACHE,  # Import for cache clearing
)

M = TypeVar("M")

# --- Test Model Definition (Using Pydantic as per original tests) ---


class Address(BaseModel):
    street: Optional[str] = (
        None  # Make street optional to match User.address usage better
    )
    zip_code: int
    tags: Optional[List[str]] = None


class User(BaseModel):
    name: str
    age: int
    score: Optional[float] = None
    is_active: bool
    roles: List[str] = []
    address: Optional[Address] = None  # Nested Optional Model
    tags: Set[str] = set()
    # Note: Tuples might be harder to query into depending on backend,
    # treat them as simple fields for these tests unless specific tuple ops are added.
    settings: Tuple[str, int] = ("default", 0)
    permissions: Tuple[str, ...] = ()  # Variable length tuple


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
    # Instantiate with the specific model type
    return QueryBuilder(user_model)


# --- Helper Function for Creating Conditions (Adapted) ---


def _get_field_object(qb: QueryBuilder[M], field_path: str) -> Field:
    """Dynamically gets the Field object from the QueryBuilder."""
    field_obj = qb.fields
    try:
        for part in field_path.split("."):
            field_obj = getattr(field_obj, part)
        # Check if the final resolved object is a Field instance
        if not isinstance(field_obj, Field):
            # This case should ideally not happen with the simplified proxy generation
            # but is kept for robustness against future changes.
            raise TypeError(
                f"Resolved path '{field_path}' did not result in a Field object, got {type(field_obj).__name__}."
            )
        return field_obj
    except AttributeError as e:
        # Re-raise clearly indicating the path failed resolution on the proxy
        raise AttributeError(
            f"Could not resolve field path '{field_path}' on qb.fields. Original error: {e}"
        ) from e


def _create_filter_condition(
    qb: QueryBuilder[M], field_path: str, op_name: str, value: Any
) -> FilterCondition:
    """Creates a FilterCondition using the Field object and operator name."""
    field_obj = _get_field_object(qb, field_path)

    # Map operator names to Field methods
    op_map: Dict[str, Callable[..., FilterCondition]] = {
        "eq": field_obj.__eq__,
        "ne": field_obj.__ne__,
        "gt": field_obj.__gt__,
        "lt": field_obj.__lt__,
        "ge": field_obj.__ge__,
        "le": field_obj.__le__,
        "in": field_obj.in_,
        "nin": field_obj.nin,
        "like": field_obj.like,
        "contains": field_obj.contains,
        "startswith": field_obj.startswith,
        "endswith": field_obj.endswith,
        "exists": field_obj.exists,
        # 'regex': field_obj.regex, # Remove if regex not implemented
    }
    if op_name not in op_map:
        raise ValueError(f"Unsupported operation name for helper: {op_name}")

    # Call the appropriate method on the Field object
    # Handle 'exists' which might not take a value in the helper call if default is used
    if op_name == "exists":
        return op_map[op_name](value)  # exists() takes the boolean value
    else:
        return op_map[op_name](value)


# Helper function to get the field name key from a condition dict (for combined tests)
def _get_field_name_from_condition(condition_dict):
    """Extracts the field name (key) from a simple condition dictionary."""
    if isinstance(condition_dict, dict) and len(condition_dict) == 1:
        return list(condition_dict.keys())[0]
    # Fallback for unexpected format - will likely fail assertion anyway
    return str(condition_dict)


# --- Test Parametrization Data ---
invalid_values_params = [
    # Basic type mismatches
    ("name", "eq", 123, ValueError, r"Path 'name': expected type str, got 123 \(int\)"),
    (
        "age",
        "eq",
        "twenty",
        ValueError,
        r"Path 'age': expected type int, got 'twenty' \(str\)",
    ),
    (
        "is_active",
        "eq",
        "true",
        ValueError,
        r"Path 'is_active': expected type bool, got 'true' \(str\)",
    ),
    (
        "score",
        "eq",
        True,
        ValueError,
        r"Path 'score': expected type float, got True \(bool\)",
    ),
    (
        "address.zip_code",
        "eq",
        "90210",
        ValueError,
        r"Path 'address.zip_code': expected type int, got '90210' \(str\)",
    ),
    # Invalid item types for collections
    (
        "roles",
        "contains",
        123,
        ValueError,
        r"Operator 'contains' on field 'roles' requires value compatible with item type str, got int",
    ),
    (
        "tags",
        "contains",
        123,
        ValueError,
        r"Operator 'contains' on field 'tags' requires value compatible with item type str, got int",
    ),
    (
        "roles",
        "in",
        ["a", 1],
        ValueError,
        r"Invalid item in 'in' list for field 'roles'.*Path 'roles \(in item \d+\)': expected type str, got 1 \(int\)",
    ),
    (
        "tags",
        "nin",
        {"a", 1},
        ValueError,
        r"Invalid item in 'nin' list for field 'tags'.*Path 'tags \(nin item \d+\)': expected type str, got 1 \(int\)",
    ),
    (
        "age",
        "in",
        [20, "30"],
        ValueError,
        r"Invalid item in 'in' list for field 'age'.*Path 'age \(in item \d+\)': expected type int, got '30' \(str\)",
    ),
    # Operator-specific value type mismatches
    ("name", "like", 123, TypeError, r"like' requires a string value, got int"),
    (
        "name",
        "startswith",
        True,
        TypeError,
        r"startswith' requires a string value, got bool",
    ),
    ("age", "in", 123, TypeError, r"in' requires a list, set, or tuple value, got int"),
    (
        "score",
        "exists",
        "maybe",
        TypeError,
        r"exists' requires a boolean value, got str",
    ),
    # Operator vs Field Type Mismatches
    (
        "age",
        "like",
        "%test%",
        ValueError,
        r"Path 'age': expected type int, got '%test%' \(str\)",
    ),  # Value error first
    (
        "roles",
        "startswith",
        "adm",
        ValueError,
        r"Path 'roles': expected type List, got 'adm' \(str\)",
    ),  # Value error first
    # ("name", "gt", "a", ValueError, ... ), # Removed: This is a valid comparison
    (
        "age",
        "gt",
        "10",
        ValueError,
        r"Path 'age': expected type int, got '10' \(str\)",
    ),  # Value error first - FIXED FRAGMENT
    # Null checks
    (
        "name",
        "eq",
        None,
        ValueError,
        r"Path 'name': expected type str, got None \(NoneType\)",
    ),
    (
        "age",
        "gt",
        None,
        ValueError,
        r"Path 'age': expected type int, got None \(NoneType\)",
    ),
]

invalid_values_ids = [
    "eq_str_int",
    "eq_int_str",
    "eq_bool_str",
    "eq_float_bool",
    "eq_nested_int_str",
    "contains_list_int",
    "contains_set_int_check",
    "in_list_str_int",
    "nin_set_str_int",
    "in_int_list_str",
    "like_val_int",
    "startswith_val_bool",
    "in_val_int",
    "exists_val_str",
    "like_field_int",
    "startswith_field_list",
    # "gt_field_str_val_str_check", # Removed ID
    "gt_field_int_val_str",
    "eq_nonoptional_none",
    "gt_nonoptional_none",
]


# --- Tests ---


@pytest.mark.parametrize(
    "field_path, op_name, value, expected_value_in_expr",
    [
        # Equality & Inequality
        ("name", "eq", "Alice", "Alice"),
        ("age", "ne", 30, 30),
        # Comparisons
        ("age", "gt", 20, 20),
        ("age", "lt", 50, 50),
        ("age", "ge", 21, 21),
        ("age", "le", 49, 49),
        # Collection Membership (Field methods convert input to list for expression dict)
        ("roles", "in", ["admin", "editor"], ["admin", "editor"]),  # Input is list
        ("tags", "nin", {"guest", "temp"}, ["guest", "temp"]),  # Input is set
        ("age", "in", (18, 21, 30), [18, 21, 30]),  # Input is tuple
        # String Matching
        ("name", "like", "%lic%", "%lic%"),
        (
            "roles",
            "contains",
            "moderator",
            "moderator",
        ),  # 'roles' is List[str], check item
        ("name", "startswith", "Al", "Al"),
        ("name", "endswith", "ice", "ice"),
        # Existence
        ("score", "exists", True, True),  # Optional float field
        ("address", "exists", False, False),  # Optional model field
        ("address.street", "exists", True, True),  # Nested optional str field
    ],
    ids=[  # Provide clear IDs for each test case
        "eq_str",
        "ne_int",
        "gt_int",
        "lt_int",
        "ge_int",
        "le_int",
        "in_list",
        "nin_set",
        "in_tuple",  # Note: expected_value_in_expr is a list for set/tuple inputs
        "like_str",
        "contains_list_item",
        "startswith_str",
        "endswith_str",
        "exists_true_optional_float",
        "exists_false_optional_model",
        "exists_true_nested_optional_str",
    ],
)
def test_single_filter_operator_expression(
    user_model: Type[User],
    field_path: str,
    op_name: str,
    value: Any,
    expected_value_in_expr: Any,
):
    """Verify that individual filter operators generate the correct expression dictionary."""
    # Use a fresh builder for isolation in parametrized tests
    qb_instance = QueryBuilder(user_model)
    condition = _create_filter_condition(qb_instance, field_path, op_name, value)
    options = qb_instance.filter(
        condition
    ).build()  # Filter and build on the same instance

    expected_expression = {
        field_path: {"operator": op_name, "value": expected_value_in_expr}
    }

    # --- Order-insensitive check for in/nin with set/tuple input ---
    actual_expression = options.expression

    # Check if we need order-insensitive comparison for the 'value' list
    # The condition applies if the OPERATOR is in/nin AND the original INPUT VALUE was a set/tuple
    if op_name in ("in", "nin") and isinstance(value, (set, tuple)):
        # Make copies to avoid modifying the original dicts
        actual_expr_copy = copy.deepcopy(actual_expression)
        expected_expr_copy = copy.deepcopy(expected_expression)

        # Extract the lists (assuming structure {field: {op:..., value:[...]}} is correct based on previous validation steps)
        try:
            actual_list = actual_expr_copy[field_path]["value"]
            expected_list = expected_expr_copy[field_path]["value"]
        except (KeyError, TypeError) as e:
            pytest.fail(
                f"Could not extract lists for comparison for {op_name} on {field_path}. "
                f"Actual: {actual_expression}, Expected: {expected_expression}. Error: {e}"
            )

        # Assert they are lists before trying to sort/convert to set
        assert isinstance(
            actual_list, list
        ), f"Actual value for {op_name}/{field_path} is not a list in expression: {actual_list}"
        assert isinstance(
            expected_list, list
        ), f"Expected value for {op_name}/{field_path} is not a list in test parameters: {expected_list}"

        # Compare the lists order-insensitively (using sets is easiest for item presence)
        assert set(actual_list) == set(expected_list), (
            f"Failed for {op_name} on {field_path}: list items mismatch (order ignored).\n"
            f"  Expected items: {set(expected_list)}\n"
            f"  Got items:      {set(actual_list)}\n"
            f"  (Original input value was type: {type(value).__name__})"
        )

        # Also check length in case of duplicate items in original input (sets handle this implicitly, but good practice)
        assert len(actual_list) == len(expected_list), (
            f"Failed for {op_name} on {field_path}: list length mismatch.\n"
            f"  Expected length: {len(expected_list)}\n"
            f"  Got length:      {len(actual_list)}\n"
            f"  (Original input value was type: {type(value).__name__})"
        )

        # Remove the 'value' key from both copies for structural comparison of the rest
        del actual_expr_copy[field_path]["value"]
        del expected_expr_copy[field_path]["value"]

        # Assert the rest of the structure matches (field path and operator)
        assert actual_expr_copy == expected_expr_copy, (
            f"Failed for {op_name} on {field_path}: structure mismatch after checking list content.\n"
            f"  Expected structure: {expected_expr_copy}\n"
            f"  Got structure:      {actual_expr_copy}"
        )

    else:
        # Standard comparison for other operators or when the input 'value' was already a list
        assert actual_expression == expected_expression, (
            f"Failed for {op_name} on {field_path}.\n"
            f"  Expected expression: {expected_expression}\n"
            f"  Got expression:      {actual_expression}"
        )
    # --- End Order-insensitive check ---


def test_combined_and_expression(qb: QueryBuilder[User]):
    """Verify combining two conditions with '&' generates an 'and' expression."""
    # Use a fresh builder for isolation if modifying state (filter does)
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

    # --- Order-insensitive check for combined expressions ---
    actual_expression = options.expression
    assert "and" in actual_expression, "Expected 'and' key in expression"
    assert isinstance(
        actual_expression.get("and"), list
    ), "'and' value should be a list"
    assert len(actual_expression.get("and", [])) == len(
        expected.get("and", [])
    ), "List lengths should match"

    # Sort the actual and expected lists using the field name as the key
    # Use .get() with default empty list to avoid errors if key is missing
    actual_sorted_list = sorted(
        actual_expression.get("and", []), key=_get_field_name_from_condition
    )
    expected_sorted_list = sorted(
        expected.get("and", []), key=_get_field_name_from_condition
    )

    # Now compare the sorted lists directly
    assert (
        actual_sorted_list == expected_sorted_list
    ), f"Combined 'and' expression mismatch.\nExpected (sorted): {expected_sorted_list}\nActual (sorted):   {actual_sorted_list}"
    # --- End Order-insensitive check ---


def test_combined_or_expression(qb: QueryBuilder[User]):
    """Verify combining two conditions with '|' generates an 'or' expression."""
    # Use a fresh builder
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

    # --- Order-insensitive check for combined expressions ---
    actual_expression = options.expression
    assert "or" in actual_expression, "Expected 'or' key in expression"
    assert isinstance(actual_expression.get("or"), list), "'or' value should be a list"
    assert len(actual_expression.get("or", [])) == len(
        expected.get("or", [])
    ), "List lengths should match"

    actual_sorted_list = sorted(
        actual_expression.get("or", []), key=_get_field_name_from_condition
    )
    expected_sorted_list = sorted(
        expected.get("or", []), key=_get_field_name_from_condition
    )

    # Compare the sorted lists directly
    assert (
        actual_sorted_list == expected_sorted_list
    ), f"Combined 'or' expression mismatch.\nExpected (sorted): {expected_sorted_list}\nActual (sorted):   {actual_sorted_list}"
    # --- End Order-insensitive check ---


def test_multiple_sequential_filters_expression(qb: QueryBuilder[User]):
    """Verify that chaining .filter() calls combines them with 'and'."""
    # Re-get field objects from the specific qb instance used for filtering
    qb_instance = QueryBuilder(qb.model_cls)
    options = (
        qb_instance.filter(qb_instance.fields.name == "Bob")
        .filter(qb_instance.fields.age < 40)
        .build()
    )
    # Expects structure like: {"and": [{"name": ...}, {"age": ...}]}
    assert "and" in options.expression, "Expected 'and' key in expression"
    assert isinstance(
        options.expression.get("and"), list
    ), "'and' value should be a list"
    assert len(options.expression.get("and", [])) == 2, "Should have 2 conditions"

    # --- Order-insensitive check for combined expressions ---
    expected_conds_list = [
        {"name": {"operator": "eq", "value": "Bob"}},
        {"age": {"operator": "lt", "value": 40}},
    ]

    actual_sorted_list = sorted(
        options.expression.get("and", []), key=_get_field_name_from_condition
    )
    expected_sorted_list = sorted(
        expected_conds_list, key=_get_field_name_from_condition
    )

    assert (
        actual_sorted_list == expected_sorted_list
    ), f"Sequential filters 'and' expression mismatch.\nExpected (sorted): {expected_sorted_list}\nActual (sorted):   {actual_sorted_list}"
    # --- End Order-insensitive check ---


@pytest.mark.parametrize(
    "option_method_name, value, expected_attr",
    [
        ("limit", 50, "limit"),
        ("offset", 10, "offset"),
        ("timeout", 5.5, "timeout"),
    ],
)
def test_pagination_and_timeout_options(
    user_model: Type[User], option_method_name: str, value: Any, expected_attr: str
):
    """Verify setting limit, offset, and timeout options correctly."""
    qb_instance = QueryBuilder(user_model)
    builder_method = getattr(qb_instance, option_method_name)
    options = builder_method(value).build()
    assert getattr(options, expected_attr) == value


def test_sort_by_ascending_option(qb: QueryBuilder[User]):
    """Verify setting ascending sort order using a Field object."""
    qb_instance = QueryBuilder(qb.model_cls)
    options = qb_instance.sort_by(qb_instance.fields.age).build()
    assert options.sort_by == "age"
    assert options.sort_desc is False


def test_sort_by_descending_option(qb: QueryBuilder[User]):
    """Verify setting descending sort order using a Field object."""
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
        qb_instance.filter(qb_instance.fields.is_active == True)
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
    assert isinstance(qb.fields.address, Field)
    assert qb.fields.address.path == "address"
    assert isinstance(qb.fields.address.zip_code, Field)
    assert qb.fields.address.zip_code.path == "address.zip_code"


def test_invalid_field_access_raises_attribute_error(qb: QueryBuilder[User]):
    """Verify accessing non-existent top-level fields raises AttributeError."""
    with pytest.raises(
        AttributeError,
        match=r"'?SimpleNamespace'? object has no attribute 'nonexistent_field'",
    ):
        _ = qb.fields.nonexistent_field
    # Note: Nested invalid access check removed as Field.__getattr__ allows it; validation happens on use.


def test_invalid_field_path_in_filter_raises_value_error(user_model: Type[User]):
    """Verify using an invalid field path (constructed manually) in filter raises ValueError."""
    qb_instance = QueryBuilder(user_model)
    invalid_field = Field("address.city")
    with pytest.raises(
        ValueError,
        match=r"Invalid filter expression: Cannot resolve part 'city' in path 'address\.city' for type Address",
    ):
        qb_instance.filter(invalid_field == "SomeCity").build()


def test_invalid_field_path_in_sort_raises_attribute_error(user_model: Type[User]):
    """Verify using an invalid field path (via manual Field) in sort_by raises AttributeError."""
    qb_instance = QueryBuilder(user_model)
    invalid_field = Field("address.city")
    with pytest.raises(
        AttributeError,
        match=r"Invalid sort field path: address\.city\. Error: Cannot resolve part 'city' in path 'address\.city' for type Address",
    ):
        qb_instance.sort_by(invalid_field).build()


def test_filter_with_valid_values_succeeds(qb: QueryBuilder[User]):
    """Check that filtering with type-correct values does not raise validation errors."""
    model = qb.model_cls
    fields = qb.fields
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
        pytest.fail(
            f"Valid filter operation raised unexpected error: {e}\n{traceback.format_exc()}"
        )


@pytest.mark.parametrize(
    "field_path, op_name, invalid_value, error_type, error_fragment",
    invalid_values_params,
    ids=invalid_values_ids,
)
def test_filter_with_invalid_values_raises_error(
    user_model: Type[User],
    field_path: str,
    op_name: str,
    invalid_value: Any,
    error_type: Type[Exception],
    error_fragment: str,
):
    """Verify filtering with incorrect types or inapplicable operators raises expected error."""
    qb_instance = QueryBuilder(user_model)

    with pytest.raises(error_type) as excinfo:
        condition = _create_filter_condition(
            qb_instance, field_path, op_name, invalid_value
        )
        qb_instance.filter(condition)

    error_str = str(excinfo.value)
    assert (
        re.search(error_fragment, error_str, re.IGNORECASE | re.DOTALL) is not None
    ), (
        f"Failed for op '{op_name}' on path '{field_path}' with value {invalid_value!r}.\n"
        f"  Expected error type: {error_type.__name__}\n"
        f"  Expected fragment (regex): r'{error_fragment}'\n"
        f"  Actual error message: '{error_str}'"
    )


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

    combined_cond = (qb_instance.fields.age > 18) & (
        qb_instance.fields.is_active == True
    )
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
