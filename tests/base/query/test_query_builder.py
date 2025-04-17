# tests/base/query/test_query_builder.py

import pytest
from pydantic import BaseModel
from typing import List, Optional, Set, Tuple, Union
import re # Import regex for more flexible matching

# Import necessary components from the module under test
from async_repository.base.query import (
    QueryBuilder,
    Expression,
    FilterCondition,
    CombinedCondition,
    QueryOptions,
    Field,
)
from async_repository.base.model_validator import (
    ValueTypeError,
    InvalidPathError
)


# --- Test Model ---

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
    settings: Tuple[str, int] = ("default", 0) # Fixed-length tuple
    permissions: Tuple[str, ...] = () # Variable-length tuple


# --- Fixtures ---

@pytest.fixture
def qb() -> QueryBuilder:
    """Provides a fresh QueryBuilder instance with the User model for each test."""
    return QueryBuilder(model=User)

@pytest.fixture
def qb_no_model() -> QueryBuilder:
    """Provides a QueryBuilder without a model for testing validation bypass."""
    return QueryBuilder(model=None)


# --- Basic Operator Tests (Unchanged from previous correct version) ---

def test_eq_operator(qb: QueryBuilder):
    condition = qb.name == "Alice"
    options = qb.filter(condition).build()
    expected = {"name": {"operator": "eq", "value": "Alice"}}
    assert options.expression == expected

def test_ne_operator(qb: QueryBuilder):
    condition = qb.age.ne(30)
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "ne", "value": 30}}
    assert options.expression == expected

def test_gt_operator(qb: QueryBuilder):
    condition = qb.age > 20
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "gt", "value": 20}}
    assert options.expression == expected

def test_lt_operator(qb: QueryBuilder):
    condition = qb.age < 50
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "lt", "value": 50}}
    assert options.expression == expected

def test_ge_operator(qb: QueryBuilder):
    condition = qb.age >= 21
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "ge", "value": 21}}
    assert options.expression == expected

def test_le_operator(qb: QueryBuilder):
    condition = qb.age <= 49
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "le", "value": 49}}
    assert options.expression == expected

def test_in_operator(qb: QueryBuilder):
    condition = qb.roles.in_(["admin", "editor"])
    options = qb.filter(condition).build()
    expected = {"roles": {"operator": "in", "value": ["admin", "editor"]}}
    assert options.expression == expected

def test_nin_operator(qb: QueryBuilder):
    condition = qb.tags.nin({"guest", "temp"})
    options = qb.filter(condition).build()
    expected = {"tags": {"operator": "nin", "value": ["guest", "temp"]}}
    assert options.expression == expected
    qb_tuple = QueryBuilder(model=User)
    condition_tuple = qb_tuple.age.nin((18, 21))
    options_tuple = qb_tuple.filter(condition_tuple).build()
    expected_tuple = {"age": {"operator": "nin", "value": [18, 21]}}
    assert options_tuple.expression == expected_tuple

def test_like_operator(qb: QueryBuilder):
    condition = qb.name.like("%lic%")
    options = qb.filter(condition).build()
    expected = {"name": {"operator": "like", "value": "%lic%"}}
    assert options.expression == expected

def test_contains_operator(qb: QueryBuilder):
    condition = qb.roles.contains("moderator")
    options = qb.filter(condition).build()
    expected = {"roles": {"operator": "contains", "value": "moderator"}}
    assert options.expression == expected

def test_startswith_operator(qb: QueryBuilder):
    condition = qb.name.startswith("Al")
    options = qb.filter(condition).build()
    expected = {"name": {"operator": "startswith", "value": "Al"}}
    assert options.expression == expected

def test_endswith_operator(qb: QueryBuilder):
    condition = qb.name.endswith("ice")
    options = qb.filter(condition).build()
    expected = {"name": {"operator": "endswith", "value": "ice"}}
    assert options.expression == expected

def test_exists_operator_true(qb: QueryBuilder):
    condition = qb.score.exists(True)
    options = qb.filter(condition).build()
    expected = {"score": {"operator": "exists", "value": True}}
    assert options.expression == expected

def test_exists_operator_false(qb: QueryBuilder):
    condition = qb.address.exists(False)
    options = qb.filter(condition).build()
    expected = {"address": {"operator": "exists", "value": False}}
    assert options.expression == expected

def test_regex_operator(qb: QueryBuilder):
    condition = qb.name.regex(r"^A.*e$")
    options = qb.filter(condition).build()
    expected = {"name": {"operator": "regex", "value": r"^A.*e$"}}
    assert options.expression == expected

# --- Combined Condition Tests (Unchanged from previous correct version) ---

def test_combined_and_condition(qb: QueryBuilder):
    condition1 = qb.name == "Alice"
    condition2 = qb.age > 25
    combined = condition1 & condition2
    options = qb.filter(combined).build()
    expected = {
        "and": [
            {"name": {"operator": "eq", "value": "Alice"}},
            {"age": {"operator": "gt", "value": 25}},
        ]
    }
    assert options.expression == expected

def test_combined_or_condition(qb: QueryBuilder):
    condition1 = qb.roles.in_(["admin", "editor"])
    condition2 = qb.is_active == False
    combined = condition1 | condition2
    options = qb.filter(combined).build()
    expected = {
        "or": [
            {"roles": {"operator": "in", "value": ["admin", "editor"]}},
            {"is_active": {"operator": "eq", "value": False}},
        ]
    }
    assert options.expression == expected

def test_multiple_filters_chained(qb: QueryBuilder):
    options = qb.filter(qb.name == "Bob").filter(qb.age < 40).build()
    expected = {
        "and": [
            {"name": {"operator": "eq", "value": "Bob"}},
            {"age": {"operator": "lt", "value": 40}},
        ]
    }
    assert options.expression == expected

# --- Query Options Tests (Unchanged from previous correct version) ---

def test_sort_by_ascending(qb: QueryBuilder):
    options = qb.sort_by("age").build()
    assert options.sort_by == "age"
    assert options.sort_desc is False
    assert options.expression == {}

def test_sort_by_descending(qb: QueryBuilder):
    options = qb.sort_by("name", descending=True).build()
    assert options.sort_by == "name"
    assert options.sort_desc is True

def test_limit(qb: QueryBuilder):
    options = qb.limit(50).build()
    assert options.limit == 50
    assert options.offset == 0

def test_offset(qb: QueryBuilder):
    options = qb.offset(10).build()
    assert options.offset == 10
    assert options.limit == 100

def test_set_timeout(qb: QueryBuilder):
    options = qb.set_timeout(5.5).build()
    assert options.timeout == 5.5

def test_random_order(qb: QueryBuilder):
    options = qb.sort_by("name").random_order().build()
    assert options.random_order is True
    assert options.sort_by is None

def test_combined_options(qb: QueryBuilder):
    options = (
        qb.filter(qb.is_active == True)
        .sort_by("score", descending=True)
        .limit(25)
        .offset(5)
        .set_timeout(10.0)
        .build()
    )
    assert options.expression == {"is_active": {"operator": "eq", "value": True}}
    assert options.sort_by == "score"
    assert options.sort_desc is True
    assert options.limit == 25
    assert options.offset == 5
    assert options.timeout == 10.0
    assert options.random_order is False


# --- Validation Tests (With Model) ---

def test_valid_field_access(qb: QueryBuilder):
    """Test accessing valid fields does not raise AttributeError."""
    assert isinstance(qb.name, Field)
    assert qb.name.name == "name"
    assert isinstance(qb.age, Field)
    assert qb.age.name == "age"
    nested_field = qb.address.street
    assert isinstance(nested_field, Field)
    assert nested_field.name == "address.street"

# --- UPDATED test_invalid_field_access ---
def test_invalid_field_access(qb: QueryBuilder):
    """Test accessing non-existent fields raises AttributeError."""
    # Test top-level invalid field - __getattr__ checks base 'nonexistent_field'
    with pytest.raises(AttributeError, match=re.escape("'User' has no attribute 'nonexistent_field'.")):
        _ = qb.nonexistent_field

    # Test nested invalid field - __getattr__ checks base 'address' (valid)
    # Field.__getattr__ creates Field('address.city') without error here.
    # The error will be caught later during filter/sort validation.
    invalid_nested_field = qb.address.city
    assert isinstance(invalid_nested_field, Field)
    assert invalid_nested_field.name == "address.city"

    # Now test that using this invalid path fails during filter/sort
    with pytest.raises(ValueError, match=re.escape("Invalid field path used in filter: Field 'city' does not exist in type Address")):
        qb.filter(invalid_nested_field == "SomeCity").build()

    # Use a new builder instance to avoid state issues
    qb_sort = QueryBuilder(model=User)
    with pytest.raises(AttributeError, match=re.escape("'User' has no sortable attribute or valid path 'address.city'.")):
        qb_sort.sort_by("address.city").build()
# --- END UPDATE ---

def test_filter_valid_value_types(qb: QueryBuilder):
    """Test filtering with values matching the model's type hints."""
    QueryBuilder(model=User).filter(qb.name == "Valid Name").build()
    QueryBuilder(model=User).filter(qb.age > 18).build()
    QueryBuilder(model=User).filter(qb.score == 95.5).build()
    QueryBuilder(model=User).filter(qb.score == None).build()
    QueryBuilder(model=User).filter(qb.is_active == True).build()
    QueryBuilder(model=User).filter(qb.roles.contains("admin")).build()
    QueryBuilder(model=User).filter(qb.tags.nin({"temp"})).build()
    QueryBuilder(model=User).filter(qb.address.zip_code == 90210).build()
    QueryBuilder(model=User).filter(qb.address == None).build()
    QueryBuilder(model=User).filter(qb.age.in_([20, 30, 40])).build()
    QueryBuilder(model=User).filter(qb.permissions.in_(("read", "write"))).build()
    QueryBuilder(model=User).filter(qb.tags.in_({"admin", "user"})).build()


def test_filter_invalid_value_type(qb: QueryBuilder):
    """Test filtering with values mismatching the model's type hints."""
    qb_name = QueryBuilder(model=User)
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path 'name': expected type str or compatible dict, got 123 (type: int).")):
        qb_name.filter(qb_name.name == 123).build()

    qb_age = QueryBuilder(model=User)
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path 'age': expected type int or compatible dict, got 'twenty' (type: str).")):
        qb_age.filter(qb_age.age == "twenty").build()

    qb_active = QueryBuilder(model=User)
    # --- UPDATED Regex for bool ---
    # The fallback error message includes "or compatible dict", even for bool
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path 'is_active': expected type bool or compatible dict, got 'true' (type: str).")):
        qb_active.filter(qb_active.is_active == "true").build()
    # --- END UPDATE ---

    qb_score = QueryBuilder(model=User)
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path 'score': expected type float or compatible dict, got True (type: bool).")):
        qb_score.filter(qb_score.score == True).build()


def test_filter_numeric_op_on_non_numeric_field(qb: QueryBuilder):
    """Test using numeric operators (> < >= <=) on non-numeric fields."""
    with pytest.raises(ValueError, match=r"Operator 'gt' requires a numeric field, but 'name' is not"):
        qb.filter(qb.name > "A").build()

def test_filter_numeric_op_with_non_numeric_value(qb: QueryBuilder):
    """Test using numeric operators (> < >= <=) with non-numeric values."""
    with pytest.raises(ValueError, match=r"Operator 'gt' requires a numeric value \(int/float\) for comparison, got str."):
        qb.filter(qb.age > "twenty").build()

def test_filter_contains_on_non_list_field(qb: QueryBuilder):
    """Test using .contains() on a field that is not a List."""
    with pytest.raises(ValueError, match=r"Operator 'contains' requires a List field, but 'name' is not"):
        qb.filter(qb.name.contains("lice")).build()

def test_filter_contains_with_wrong_item_type(qb: QueryBuilder):
    """Test using .contains() with a value type mismatching the list item type."""
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path 'Value for 'roles' contains': expected type str or compatible dict, got 123 (type: int).")):
        qb.filter(qb.roles.contains(123)).build()

def test_filter_in_with_wrong_item_type(qb: QueryBuilder):
    """Test using .in_() with item types mismatching the field type."""
    with pytest.raises(ValueError, match=re.escape("Invalid filter condition: Path ''age' item 1 for 'in'': expected type int or compatible dict, got 'twenty' (type: str).")):
        qb.filter(qb.age.in_([10, "twenty", 30])).build()

def test_filter_in_with_non_iterable(qb: QueryBuilder):
    """Test using .in_() with a value that isn't a list/set/tuple."""
    with pytest.raises(TypeError, match=r"in_ operator requires a list, set, or tuple, got <class 'int'>"):
        qb.age.in_(123) # type: ignore

def test_filter_exists_with_non_bool(qb: QueryBuilder):
    """Test using .exists() with a non-boolean value."""
    with pytest.raises(TypeError, match=r"exists operator requires a boolean value, got <class 'str'>"):
        qb.score.exists("yes") # type: ignore

def test_filter_string_op_with_non_string(qb: QueryBuilder):
    """Test using string operators (like, startswith, etc.) with non-string values."""
    with pytest.raises(TypeError, match=r"startswith operator requires a string substring, got <class 'int'>"):
        qb.name.startswith(123) # type: ignore

    with pytest.raises(TypeError, match=r"like operator requires a string pattern, got <class 'int'>"):
        qb.name.like(123) # type: ignore

    with pytest.raises(TypeError, match=r"endswith operator requires a string substring, got <class 'int'>"):
        qb.name.endswith(123) # type: ignore

    with pytest.raises(TypeError, match=r"regex operator requires a string pattern, got <class 'int'>"):
        qb.name.regex(123) # type: ignore


def test_sort_by_invalid_field(qb: QueryBuilder):
    """Test sorting by a non-existent field raises AttributeError."""
    with pytest.raises(AttributeError, match=re.escape("'User' has no sortable attribute or valid path 'nonexistent'.")):
        qb.sort_by("nonexistent").build()

    with pytest.raises(AttributeError, match=re.escape("'User' has no sortable attribute or valid path 'address.city'.")):
        qb.sort_by("address.city").build() # Nested invalid path


# --- Validation Tests (Without Model) ---

def test_no_model_field_access(qb_no_model: QueryBuilder):
    """Test field access works without a model (no validation)."""
    assert qb_no_model.any_field.name == "any_field"
    nested_field = qb_no_model.nested.path.here
    assert isinstance(nested_field, Field)
    assert nested_field.name == "nested.path.here"

def test_no_model_filter(qb_no_model: QueryBuilder):
    """Test filtering works without a model (no validation)."""
    options = qb_no_model.filter(qb_no_model.status == "active").build()
    expected = {"status": {"operator": "eq", "value": "active"}}
    assert options.expression == expected

def test_no_model_sort(qb_no_model: QueryBuilder):
    """Test sorting works without a model (no validation)."""
    options = qb_no_model.sort_by("some_field").build()
    assert options.sort_by == "some_field"


# --- Edge Case Tests ---

def test_build_empty(qb: QueryBuilder):
    """Test building QueryOptions without any filters or options."""
    options = qb.build()
    assert options.expression == {}
    assert options.sort_by is None
    assert options.sort_desc is False
    assert options.limit == 100 # Default
    assert options.offset == 0   # Default
    assert options.timeout is None
    assert options.random_order is False