# tests/base/query/test_query_builder.py

import pytest
from typing import List, Optional, Callable, Any, Dict, Type, TypeVar
import re
import traceback

from async_repository.base.query import (
    QueryBuilder, Field, QueryOptions, Expression, FilterCondition,
    QueryExpression, QueryFilter, QueryLogical, QueryOperator,
    _PROXY_CACHE, GenericFieldsProxy
)

from tests.base.conftest import User, Address, Metadata, get_field_object

from tests.base.conftest import assert_expression_present

M = TypeVar("M")


# --- Fixtures ---
@pytest.fixture(autouse=True)
def clear_proxy_cache():
    _PROXY_CACHE.clear()
    yield
    _PROXY_CACHE.clear()

@pytest.fixture
def user_model() -> Type[User]:
    return User

@pytest.fixture
def qb(user_model: Type[User]) -> QueryBuilder[User]:
    return QueryBuilder(user_model)



def _create_internal_condition(qb: QueryBuilder[M], field_path: str, op_name: str, value: Any) -> FilterCondition:
    """Creates an internal FilterCondition using the resolved Field object."""
    try:
        field_obj = get_field_object(qb, field_path)
    except AttributeError as e:
        # If path resolution fails, re-raise clearly for test debugging
        raise AttributeError(f"Failed to get Field object for path '{field_path}': {e}") from e

    op_map: Dict[str, Callable[..., FilterCondition]] = {
        "eq": field_obj.__eq__, "ne": field_obj.__ne__, "gt": field_obj.__gt__,
        "lt": field_obj.__lt__, "ge": field_obj.__ge__, "le": field_obj.__le__,
        "in": field_obj.in_, "nin": field_obj.nin, "like": field_obj.like,
        "contains": field_obj.contains, "startswith": field_obj.startswith,
        "endswith": field_obj.endswith, "exists": field_obj.exists,
    }
    if op_name not in op_map:
        raise ValueError(f"Unsupported operation name for helper: {op_name}")
    try:
        return op_map[op_name](value)
    except Exception as e:
        # Catch errors during operator application (e.g., Field's type checks)
        raise type(e)(f"Error applying operator '{op_name}' to Field('{field_obj.path}') with value {value!r}: {e}") from e


# --- Test Parametrization Data ---
# Update invalid_values_params for new __getitem__ related errors
invalid_values_params = [
    # Basic type mismatches (validator -> ValueTypeError -> ValueError)
    ("name", "eq", 123, ValueError, r"Invalid filter expression: Path 'name': expected type str, got 123 \(int\)\."),
    ("age", "eq", "twenty", ValueError, r"Invalid filter expression: Path 'age': expected type int, got 'twenty' \(str\)\."),
    ("active", "eq", "true", ValueError, r"Invalid filter expression: Path 'active': expected type bool, got 'true' \(str\)\."),
    ("score", "eq", True, ValueError, r"Invalid filter expression: Path 'score': bool invalid for Union.*allowing int but not bool\."),

    # Operator-specific type mismatches (validator -> ValueTypeError -> re-raise ValueTypeError -> ValueError)
    ("roles", "contains", 123, ValueError, r"Invalid filter expression: Operator 'contains' on field 'roles' requires value compatible with item type str, got int\. Original error: Path 'roles \(contains value\)': expected type str, got 123 \(int\)\."),
    ("tags", "contains", 123, ValueError, r"Invalid filter expression: Operator 'contains' on field 'tags' requires value compatible with item type str, got int\. Original error: Path 'tags \(contains value\)': expected type str, got 123 \(int\)\."),
    ("roles", "in", ["a", 1], ValueError, r"Invalid filter expression: Invalid item in 'in' list for field 'roles'\. Path 'roles \(in item 1\)': expected type str, got 1 \(int\)\."),
    ("tags", "nin", {"a", 1}, ValueError, r"Invalid filter expression: Invalid item in 'nin' list for field 'tags'\. Path 'tags \(nin item \d+\)': expected type str, got 1 \(int\)\."), # Index may vary
    ("age", "in", [20, "30"], ValueError, r"Invalid filter expression: Invalid item in 'in' list for field 'age'\. Path 'age \(in item 1\)': expected type int, got '30' \(str\)\."),

    # Operator value type checks (Field._op -> TypeError)
    ("name", "like", 123, TypeError, r"Operator 'like' requires a string value"),
    ("name", "startswith", True, TypeError, r"Operator 'startswith' requires a string value"),
    ("age", "in", 123, TypeError, r"Operator 'in' requires a list/set/tuple"),
    ("score", "exists", "maybe", TypeError, r"Operator 'exists' requires a boolean value"),

    # Operator-Field compatibility checks (validator -> ValueTypeError -> ValueError)
    ("age", "like", "%test%", ValueError, r"Invalid filter expression: Path 'age': expected type int, got '%test%' \(str\)\."), # Fails basic validation first
    ("roles", "startswith", "adm", ValueError, r"Invalid filter expression: Path 'roles': expected type List, got 'adm' \(str\)\."),
    ("age", "gt", "10", ValueError, r"Invalid filter expression: Path 'age': expected type int, got '10' \(str\)\."),

    ("addresses[0].zipcode", "eq", "abc", ValueError, r"Invalid filter expression: Path 'addresses\.0\.zipcode': expected type int, got 'abc' \(str\)\."), # Valid path, invalid value type
    # This case now raises ValueError directly from the filter method due to InvalidPathError
    ("addresses[0].nonexistent", "eq", 1, ValueError, r"Invalid filter expression: Field 'nonexistent' does not exist in type Address\. Path: 'addresses\.0\.nonexistent'\. in model User"), # Invalid sub-path after index

]
invalid_values_ids = [ f"{path.replace('.', '_').replace('[','_').replace(']','')}_{op}_{type(val).__name__}" for path, op, val, _, _ in invalid_values_params ]

# Separate tests for errors happening during path *resolution* vs validation
invalid_path_resolution_params = [
     # Path using non-integer index in __getitem__
    ("addresses['key']", TypeError, r"Field index must be an integer, got str"),
    # Path using negative index in __getitem__
    ("addresses[-1].zipcode", IndexError, r"Negative indexing is not currently supported"),
    # Path accessing attribute on non-object/dict (after index)
    ("tags[0].subfield", AttributeError, r"Could not fully resolve field path.*part 'subfield': 'str' object has no attribute 'subfield'"),
]
invalid_path_resolution_ids = [ f"{path.replace('.', '_').replace('[','_').replace(']','')}" for path, _, _ in invalid_path_resolution_params ]


# --- Tests ---
@pytest.mark.parametrize(
    "field_path, op_name, value, expected_op_enum, expected_value_in_filter",
    [
        ("name", "eq", "Alice", QueryOperator.EQ, "Alice"),
        ("age", "ne", 30, QueryOperator.NE, 30),
        ("age", "gt", 20, QueryOperator.GT, 20),
        ("age", "lt", 50, QueryOperator.LT, 50),
        ("age", "ge", 21, QueryOperator.GTE, 21),
        ("age", "le", 49, QueryOperator.LTE, 49),
        ("roles", "in", ["admin", "editor"], QueryOperator.IN, ["admin", "editor"]),
        ("tags", "nin", {"guest", "temp"}, QueryOperator.NIN, ["guest", "temp"]),
        ("age", "in", (18, 21, 30), QueryOperator.IN, [18, 21, 30]),
        ("name", "like", "%lic%", QueryOperator.LIKE, "%lic%"),
        ("roles", "contains", "moderator", QueryOperator.CONTAINS, "moderator"),
        ("name", "startswith", "Al", QueryOperator.STARTSWITH, "Al"),
        ("name", "endswith", "ice", QueryOperator.ENDSWITH, "ice"),
        ("score", "exists", True, QueryOperator.EXISTS, True),
        ("addresses", "exists", False, QueryOperator.EXISTS, False),
        ("email", "eq", None, QueryOperator.EQ, None),
        ("addresses[0].zipcode", "eq", 90210, QueryOperator.EQ, 90210),
        ("addresses[1].street", "ne", "Street street", QueryOperator.NE, "Street street"),
    ],
)
def test_single_filter_operator_expression(
    user_model: Type[User],
    field_path: str,
    op_name: str,
    value: Any,
    expected_op_enum: QueryOperator,
    expected_value_in_filter: Any,
):
    """Tests creating simple filters with various operators and paths."""
    qb_instance = QueryBuilder(user_model)
    # Create the condition using the helper which now handles indexed paths
    internal_condition = _create_internal_condition(qb_instance, field_path, op_name, value)
    # Build and assert
    options = qb_instance.filter(internal_condition).build()
    # Use the public path string for assertion
    public_path = field_path.replace('[', '.').replace(']', '') # Convert test path to validator format
    assert_expression_present(
        options.expression, QueryFilter, field_path=public_path,
        operator=expected_op_enum, expected_value=expected_value_in_filter
    )

# --- NEW Test for list index specific functionality ---
def test_query_specific_list_index(qb: QueryBuilder[User]):
    """Explicitly tests the __getitem__ syntax for list field access."""
    # 1. Access via __getitem__ and then __getattr__
    condition = qb.fields.addresses[0].zipcode == 12345
    options = QueryBuilder(User).filter(condition).build()

    # Assert path is correct
    assert_expression_present(
        options.expression, QueryFilter,
        field_path="addresses.0.zipcode",
        operator=QueryOperator.EQ,
        expected_value=12345
    )

    # 2. Test chaining __getitem__ (if model had nested lists)
    # Example (requires model change):
    # condition_nested = qb.fields.matrix[1][2] == "value"
    # options_nested = QueryBuilder(MatrixModel).filter(condition_nested).build()
    # assert_expression_present(options_nested.expression, QueryFilter, field_path="matrix.1.2")

# --- Combined expression tests (remain the same) ---
def test_combined_and_expression(qb: QueryBuilder[User]):
    condition1 = qb.fields.name == "Alice"
    condition2 = qb.fields.age > 25
    combined = condition1 & condition2
    options = qb.filter(combined).build()
    assert isinstance(options.expression, QueryLogical)
    assert options.expression.operator == "and"
    assert len(options.expression.conditions) == 2
    assert_expression_present(options.expression, QueryFilter, field_path="name", operator=QueryOperator.EQ, expected_value="Alice")
    assert_expression_present(options.expression, QueryFilter, field_path="age", operator=QueryOperator.GT, expected_value=25)

def test_combined_or_expression(qb: QueryBuilder[User]):
    condition1 = qb.fields.roles.in_(["admin", "editor"])
    condition2 = qb.fields.active == False
    combined = condition1 | condition2
    options = qb.filter(combined).build()
    assert isinstance(options.expression, QueryLogical)
    assert options.expression.operator == "or"
    assert len(options.expression.conditions) == 2
    assert_expression_present(options.expression, QueryFilter, field_path="roles", operator=QueryOperator.IN, expected_value=["admin", "editor"])
    assert_expression_present(options.expression, QueryFilter, field_path="active", operator=QueryOperator.EQ, expected_value=False)

def test_multiple_sequential_filters_expression(qb: QueryBuilder[User]):
    options = ( qb.filter(qb.fields.name == "Bob").filter(qb.fields.age < 40).build() )
    assert isinstance(options.expression, QueryLogical)
    assert options.expression.operator == "and"
    assert len(options.expression.conditions) == 2 # Still 2 direct conditions before translation flattening
    final_expr = options.expression
    # After translation, ANDs might be flattened, so check individual filters exist
    assert_expression_present(final_expr, QueryFilter, field_path="name", operator=QueryOperator.EQ, expected_value="Bob")
    assert_expression_present(final_expr, QueryFilter, field_path="age", operator=QueryOperator.LT, expected_value=40)

# --- Option tests (remain the same) ---
@pytest.mark.parametrize(
    "option_method_name, value, expected_attr",
    [("limit", 50, "limit"), ("offset", 10, "offset"), ("timeout", 5.5, "timeout")],
)
def test_pagination_and_timeout_options(user_model: Type[User], option_method_name: str, value: Any, expected_attr: str ):
    qb_instance = QueryBuilder(user_model)
    builder_method = getattr(qb_instance, option_method_name)
    options = builder_method(value).build()
    assert getattr(options, expected_attr) == value

def test_sort_by_ascending_option(qb: QueryBuilder[User]):
    options = qb.sort_by(qb.fields.age).build()
    assert options.sort_by == "age" and options.sort_desc is False

def test_sort_by_descending_option(qb: QueryBuilder[User]):
    options = qb.sort_by(qb.fields.name, descending=True).build()
    assert options.sort_by == "name" and options.sort_desc is True

def test_random_order_option(qb: QueryBuilder[User]):
    options = qb.sort_by(qb.fields.name).random_order().build()
    assert options.random_order is True and options.sort_by is None and options.sort_desc is False

def test_combined_query_options(qb: QueryBuilder[User]):
    options = ( qb.filter(qb.fields.active == True).sort_by(qb.fields.score, descending=True).limit(25).offset(5).timeout(10.0).build() )
    assert_expression_present(options.expression, QueryFilter, field_path="active", operator=QueryOperator.EQ, expected_value=True)
    assert options.sort_by == "score" and options.sort_desc is True
    assert options.limit == 25 and options.offset == 5 and options.timeout == 10.0 and options.random_order is False

# --- Field Access and Path Validation Tests ---
def test_valid_field_access_returns_field_object(qb: QueryBuilder[User]):
    # Direct attributes
    assert isinstance(qb.fields.name, Field) and qb.fields.name.path == "name"
    assert isinstance(qb.fields.addresses, Field) and qb.fields.addresses.path == "addresses"
    # Nested attributes
    assert isinstance(qb.fields.metadata.timestamp, Field) and qb.fields.metadata.timestamp.path == "metadata.timestamp"
    # --- NEW: Indexed access ---
    assert isinstance(qb.fields.addresses[0], Field) and qb.fields.addresses[0].path == "addresses.0"
    assert isinstance(qb.fields.addresses[10].street, Field) and qb.fields.addresses[10].street.path == "addresses.10.street"


def test_invalid_field_access_raises_attribute_error(qb: QueryBuilder[User]):
    # Dot notation for a non-existent top-level field should fail immediately
    with pytest.raises(AttributeError, match=r"object has no attribute 'nonexistent_field'"):
        _ = qb.fields.nonexistent_field

# --- Adjusted path validation tests ---
def test_invalid_field_path_in_filter_raises_value_error(user_model: Type[User]):
    """ Tests errors caught during QueryBuilder.filter() validation step """
    qb_instance = QueryBuilder(user_model)
    # 1. Invalid attribute after valid index
    invalid_field_obj = qb_instance.fields.addresses[0].nonexistent
    expected_regex_1 = (
        r"Invalid filter expression: Field 'nonexistent' does not exist in type Address. "
        r"Path: 'addresses\.0\.nonexistent'\. in model User"
    )
    with pytest.raises(ValueError, match=expected_regex_1):
        qb_instance.filter(invalid_field_obj == "SomeCity")

    # 2. Invalid direct attribute
    invalid_field_obj_2 = Field("invalid_root_field")
    expected_regex_2 = r"Invalid filter expression: Field 'invalid_root_field' does not exist.*Path: 'invalid_root_field'\. in model User"
    with pytest.raises(ValueError, match=expected_regex_2):
         qb_instance.filter(invalid_field_obj_2 == "value")

def test_invalid_field_path_in_sort_raises_attribute_error(user_model: Type[User]):
    """ Tests errors caught during QueryBuilder.sort_by() validation step """
    qb_instance = QueryBuilder(user_model)
    # 1. Invalid attribute after valid index
    invalid_field_obj = qb_instance.fields.addresses[0].nonexistent
    expected_regex_1 = (
        r"Invalid sort field path: addresses\.0\.nonexistent\. Error: "
        r"Field 'nonexistent' does not exist in type Address. Path: 'addresses\.0\.nonexistent'\. in model User"
    )
    with pytest.raises(AttributeError, match=expected_regex_1):
        qb_instance.sort_by(invalid_field_obj)

    # 2. Invalid direct attribute
    invalid_field_obj_2 = Field("invalid_root_field")
    expected_regex_2 = r"Invalid sort field path: invalid_root_field\. Error: Field 'invalid_root_field' does not exist.*Path: 'invalid_root_field'\. in model User"
    with pytest.raises(AttributeError, match=expected_regex_2):
         qb_instance.sort_by(invalid_field_obj_2)

# --- NEW: Tests for path resolution errors (before validation) ---
@pytest.mark.parametrize(
    "invalid_path_str, error_type, error_fragment",
    invalid_path_resolution_params,
    ids=invalid_path_resolution_ids,
)
def test_invalid_path_resolution_raises_error(qb: QueryBuilder[User], invalid_path_str: str, error_type: Type[Exception], error_fragment: str):
    """ Tests errors that occur during Field path construction itself """
    with pytest.raises(error_type, match=error_fragment):
        # Attempt to resolve the path using the helper, which triggers __getitem__/__getattr__
        get_field_object(qb, invalid_path_str)


# --- Value Validation Tests ---
def test_filter_with_valid_values_succeeds(qb: QueryBuilder[User]):
    """ Tests various valid filter expressions that should pass validation """
    model = qb.model_cls; fields = qb.fields
    try:
        QueryBuilder(model).filter(fields.name == "Valid Name").build()
        QueryBuilder(model).filter(fields.age > 18).build()
        QueryBuilder(model).filter(fields.score == 95.5).build()
        QueryBuilder(model).filter(fields.score == 10).build()
        QueryBuilder(model).filter(fields.email != None).build()
        QueryBuilder(model).filter(fields.active == True).build()
        QueryBuilder(model).filter(fields.roles.contains("admin")).build()
        QueryBuilder(model).filter(fields.tags.contains("admin")).build()
        QueryBuilder(model).filter(fields.tags.in_({"admin", "user"})).build()
        # --- NEW: Valid indexed access ---
        QueryBuilder(model).filter(fields.addresses[0].zipcode == 90210).build()
        QueryBuilder(model).filter(fields.addresses[1].street == "Street street 123").build()
        QueryBuilder(model).filter(fields.addresses[0].city != "Unknown").build()
        # ---------------------------------
        QueryBuilder(model).filter(fields.addresses == [Address(), Address(), Address()]).build()
        QueryBuilder(model).filter(fields.age.in_([20, 30, 40])).build()
        QueryBuilder(model).filter(fields.tags.exists(True)).build()
        QueryBuilder(model).filter(fields.addresses.exists(True)).build()

    except (ValueError, TypeError, AttributeError) as e:
        pytest.fail(f"Valid filter operation raised unexpected error: {e}\n{traceback.format_exc()}")


@pytest.mark.parametrize(
    "field_path, op_name, invalid_value, error_type, error_fragment",
    invalid_values_params,
    ids=invalid_values_ids,
)
def test_filter_with_invalid_values_or_path_raises_error( # Renamed for clarity
    user_model: Type[User],
    field_path: str,
    op_name: str,
    invalid_value: Any,
    error_type: Type[Exception],
    error_fragment: str,
):
    """ Tests filters that should fail during QueryBuilder.filter validation """
    qb_instance = QueryBuilder(user_model)
    if field_path == "is_active": field_path = "active" # Legacy correction

    with pytest.raises(error_type) as excinfo:
        # Helper now resolves path first, then creates condition
        internal_condition = _create_internal_condition(qb_instance, field_path, op_name, invalid_value)
        # The filter method performs the final validation
        qb_instance.filter(internal_condition)

    error_str = str(excinfo.value)
    assert re.search(error_fragment, error_str, re.IGNORECASE | re.DOTALL) is not None, (
        f"Failed for op '{op_name}' on path '{field_path}' with value {invalid_value!r}.\n"
        f"  Expected error type: {error_type.__name__}\n"
        f"  Expected fragment (regex): r'{error_fragment}'\n"
        f"  Actual error message: '{error_str}'"
    )


def test_build_without_filters_or_options(qb: QueryBuilder[User]):
    options = qb.build()
    assert isinstance(options, QueryOptions) and options.expression is None
    assert options.sort_by is None and options.sort_desc is False
    assert options.limit == 100 and options.offset == 0
    assert options.timeout is None and options.random_order is False


def test_repr_methods(qb: QueryBuilder[User]):
    # Internal FilterCondition
    filter_cond_internal = qb.fields.name == "Test"
    assert repr(filter_cond_internal) == "FilterCondition('name', 'eq', 'Test')"

    # Internal CombinedCondition
    combined_cond_internal = (qb.fields.age > 18) & (qb.fields.active == True)
    repr_combined = repr(combined_cond_internal)
    assert repr_combined.startswith("CombinedCondition('and', ")
    assert "FilterCondition('age', 'gt', 18)" in repr_combined
    assert "FilterCondition('active', 'eq', True)" in repr_combined

    # Field object (including index)
    field_obj_indexed = qb.fields.addresses[1].zipcode
    assert repr(field_obj_indexed) == "Field(path='addresses.1.zipcode')"

    # Final QueryOptions
    options = qb.filter(qb.fields.addresses[0].city == "LA").limit(10).build()
    options_repr = repr(options)
    assert options_repr.startswith("QueryOptions(")
    assert "expression=QueryFilter(field_path='addresses.0.city', operator=<QueryOperator.EQ: 'eq'>, value='LA')" in options_repr
    assert "limit=10" in options_repr
    assert "offset=0" in options_repr # Default included

def test_model_less_builder():
    qb_generic = QueryBuilder() # No model
    assert qb_generic._validator is None and isinstance(qb_generic.fields, GenericFieldsProxy)

    # Basic dot notation
    field1 = qb_generic.fields.any_field
    assert isinstance(field1, Field) and field1.path == "any_field"
    deep_field = qb_generic.fields.level1.level2.final_name
    assert isinstance(deep_field, Field) and deep_field.path == "level1.level2.final_name"

    # --- NEW: Indexed access without model ---
    indexed_field = qb_generic.fields.some_list[5].value
    assert isinstance(indexed_field, Field) and indexed_field.path == "some_list.5.value"

    # Build query without validation
    try:
        options = (
            qb_generic.filter(qb_generic.fields.some_field == "value")
            .filter(qb_generic.fields.num_field > 100)
            .filter(qb_generic.fields.data.payload.status.contains("OK"))
            .filter(qb_generic.fields.items[2].name == "Widget") # Add indexed query
            .sort_by(qb_generic.fields.num_field, descending=True).limit(10).build()
        )

        # --- Start of corrected assertions ---
        assert isinstance(options.expression, QueryLogical)
        assert options.expression.operator == "and"
        # Check that all individual filters are present somewhere in the nested structure
        assert_expression_present(options.expression, QueryFilter, field_path='some_field', operator=QueryOperator.EQ, expected_value='value', check_count=1)
        assert_expression_present(options.expression, QueryFilter, field_path='num_field', operator=QueryOperator.GT, expected_value=100, check_count=1)
        assert_expression_present(options.expression, QueryFilter, field_path='data.payload.status', operator=QueryOperator.CONTAINS, expected_value='OK', check_count=1)
        assert_expression_present(options.expression, QueryFilter, field_path='items.2.name', operator=QueryOperator.EQ, expected_value='Widget', check_count=1)
        # --- End of corrected assertions ---

        assert options.sort_by == "num_field" and options.sort_desc is True and options.limit == 10
    except (ValueError, TypeError, AttributeError) as e:
        pytest.fail(f"Model-less builder failed unexpectedly: {e}\n{traceback.format_exc()}")

    # TypeErrors from Field._op still work
    with pytest.raises(TypeError, match="Operator 'like' requires a string value"):
        qb_generic.filter(qb_generic.fields.foo.like(123))

    # TypeErrors from Field.__getitem__ still work
    with pytest.raises(TypeError, match="Field index must be an integer"):
        _ = qb_generic.fields.bar["key"]
    with pytest.raises(IndexError, match="Negative indexing"):
        _ = qb_generic.fields.bar[-1]