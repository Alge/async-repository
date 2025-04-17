import pytest
import re
from pydantic import BaseModel

from async_repository.base.query import QueryOptions, QueryBuilder



# Dummy Pydantic model for validation.
class User(BaseModel):
    name: str
    age: int
    status: str
    description: str = ""
    username: str = ""
    email: str = ""
    code: str = ""
    role: str = ""
    category: str = ""


@pytest.fixture
def qb():
    # Returns a fresh QueryBuilder for tests that don't need multiple conditions.
    return QueryBuilder(model=User)


def test_eq_operator(qb):
    condition = qb.name == "Alice"
    options = qb.filter(condition).build()
    expected = {"name": "Alice"}
    assert options.expression == expected


def test_ne_operator(qb):
    # Assuming that the DSL now supports 'ne' via qb.age.ne(30)
    condition = qb.age.ne(30)
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "ne", "value": 30}}
    assert options.expression == expected


def test_comparison_operator(qb):
    condition = qb.age > 20
    options = qb.filter(condition).build()
    expected = {"age": {"operator": "gt", "value": 20}}
    assert options.expression == expected


def test_combined_and_condition(qb):
    condition1 = qb.name == "Alice"
    condition2 = qb.age > 25
    combined = condition1 & condition2
    options = qb.filter(combined).build()
    expected = {"and": [{"name": "Alice"}, {"age": {"operator": "gt", "value": 25}}]}
    assert options.expression == expected


def test_combined_or_condition(qb):
    condition1 = qb.status.in_(["active", "pending"])
    condition2 = qb.role.ne("guest")
    combined = condition1 | condition2
    options = qb.filter(combined).build()
    expected = {
        "or": [
            {"status": {"operator": "in", "value": ["active", "pending"]}},
            {"role": {"operator": "ne", "value": "guest"}},
        ]
    }
    assert options.expression == expected


def test_string_operators():
    # Each condition uses a fresh QueryBuilder instance to avoid state accumulation.

    # like operator
    qb_like = QueryBuilder(model=User)
    condition = qb_like.name.like("%Ali%")
    options = qb_like.filter(condition).build()
    expected = {"name": {"operator": "like", "value": "%Ali%"}}
    assert options.expression == expected

    # contains operator
    qb_contains = QueryBuilder(model=User)
    condition = qb_contains.description.contains("test")
    options = qb_contains.filter(condition).build()
    expected = {"description": {"operator": "contains", "value": "test"}}
    assert options.expression == expected

    # startswith operator
    qb_starts = QueryBuilder(model=User)
    condition = qb_starts.username.startswith("admin")
    options = qb_starts.filter(condition).build()
    expected = {"username": {"operator": "startswith", "value": "admin"}}
    assert options.expression == expected

    # endswith operator
    qb_ends = QueryBuilder(model=User)
    condition = qb_ends.username.endswith("user")
    options = qb_ends.filter(condition).build()
    expected = {"username": {"operator": "endswith", "value": "user"}}
    assert options.expression == expected


def test_exists_and_regex_operators():
    # exists operator with its own QueryBuilder
    qb_exists = QueryBuilder(model=User)
    condition = qb_exists.email.exists(True)
    options = qb_exists.filter(condition).build()
    expected = {"email": {"operator": "exists", "value": True}}
    assert options.expression == expected

    # regex operator with its own QueryBuilder
    qb_regex = QueryBuilder(model=User)
    condition = qb_regex.code.regex(r"^\d{3}-\d{3}$")
    options = qb_regex.filter(condition).build()
    expected = {"code": {"operator": "regex", "value": r"^\d{3}-\d{3}$"}}
    assert options.expression == expected


def test_in_and_nin_operators():
    # in operator with its own QueryBuilder
    qb_in = QueryBuilder(model=User)
    condition = qb_in.status.in_(["active", "pending"])
    options = qb_in.filter(condition).build()
    expected = {"status": {"operator": "in", "value": ["active", "pending"]}}
    assert options.expression == expected

    # nin operator with its own QueryBuilder
    qb_nin = QueryBuilder(model=User)
    condition = qb_nin.category.nin(["obsolete", "deprecated"])
    options = qb_nin.filter(condition).build()
    expected = {"category": {"operator": "nin", "value": ["obsolete", "deprecated"]}}
    assert options.expression == expected


def test_sort_limit_offset_timeout():
    qb_instance = QueryBuilder(model=User)
    qb_instance.sort_by("name", descending=True)
    qb_instance.limit(50)
    qb_instance.offset(10)
    qb_instance.set_timeout(5.0)
    options = qb_instance.build()
    assert options.sort_by == "name"
    assert options.sort_desc is True
    assert options.limit == 50
    assert options.offset == 10
    assert options.timeout == 5.0


def test_random_order(qb):
    # Verify that the random_order flag is set when calling random_order().
    qb.random_order()
    options = qb.build()
    assert options.random_order is True


def test_invalid_field():
    qb_instance = QueryBuilder(model=User)
    with pytest.raises(AttributeError):
        _ = qb_instance.nonexistent_field
