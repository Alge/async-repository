# tests/database_implementations/test_overloading.py

import pytest
from pydantic import BaseModel
from typing import Optional, List

# Import internal expression classes and builder
from async_repository.base.query import (
    QueryBuilder, Field, Expression, FilterCondition, CombinedCondition, QueryOptions
)

# Define a simple model for builder tests if needed
class SampleModel(BaseModel):
    id: str
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []


# --- Tests for Field Operator Overloading ---

def test_field_equality_operators():
    """Test == and != operators on Field."""
    field = Field("name")
    eq_cond = field == "test"
    ne_cond = field != "other"

    assert isinstance(eq_cond, FilterCondition)
    assert eq_cond.field_path == "name"
    assert eq_cond.operator == "eq"
    assert eq_cond.value == "test"

    assert isinstance(ne_cond, FilterCondition)
    assert ne_cond.field_path == "name"
    assert ne_cond.operator == "ne"
    assert ne_cond.value == "other"


def test_field_comparison_operators():
    """Test >, <, >=, <= operators on Field."""
    field = Field("age")
    gt_cond = field > 18
    lt_cond = field < 65
    ge_cond = field >= 21
    le_cond = field <= 100

    assert isinstance(gt_cond, FilterCondition) and gt_cond.operator == "gt" and gt_cond.value == 18
    assert isinstance(lt_cond, FilterCondition) and lt_cond.operator == "lt" and lt_cond.value == 65
    assert isinstance(ge_cond, FilterCondition) and ge_cond.operator == "ge" and ge_cond.value == 21
    assert isinstance(le_cond, FilterCondition) and le_cond.operator == "le" and le_cond.value == 100


def test_field_methods():
    """Test methods like .in_(), .nin(), .like(), etc. on Field."""
    field = Field("tags")
    field_name = Field("name") # For string ops

    in_cond = field.in_(["tag1", "tag2"])
    nin_cond = field.nin(["tag3"])
    like_cond = field_name.like("tag%")
    contains_cond = field.contains("tag") # Can apply to list or string
    startswith_cond = field_name.startswith("t")
    endswith_cond = field_name.endswith("g")
    exists_cond = field.exists()
    not_exists_cond = field.exists(False)

    assert isinstance(in_cond, FilterCondition) and in_cond.operator == "in" and in_cond.value == ["tag1", "tag2"]
    assert isinstance(nin_cond, FilterCondition) and nin_cond.operator == "nin" and nin_cond.value == ["tag3"]
    assert isinstance(like_cond, FilterCondition) and like_cond.operator == "like" and like_cond.value == "tag%"
    assert isinstance(contains_cond, FilterCondition) and contains_cond.operator == "contains" and contains_cond.value == "tag"
    assert isinstance(startswith_cond, FilterCondition) and startswith_cond.operator == "startswith" and startswith_cond.value == "t"
    assert isinstance(endswith_cond, FilterCondition) and endswith_cond.operator == "endswith" and endswith_cond.value == "g"
    assert isinstance(exists_cond, FilterCondition) and exists_cond.operator == "exists" and exists_cond.value is True
    assert isinstance(not_exists_cond, FilterCondition) and not_exists_cond.operator == "exists" and not_exists_cond.value is False

    # Test type errors for string methods
    with pytest.raises(TypeError): field_name.like(123)
    with pytest.raises(TypeError): field_name.startswith(True)
    with pytest.raises(TypeError): field_name.endswith(None)
    # Test type errors for collection methods
    with pytest.raises(TypeError): field.in_(123)
    with pytest.raises(TypeError): field.nin("abc")
    # Test type errors for exists
    with pytest.raises(TypeError): field.exists("yes") # type: ignore


def test_logical_operators():
    """Test logical & (AND) and | (OR) operators between Expressions."""
    name_cond = Field("name") == "Alice"
    age_cond = Field("age") > 30

    # Test AND
    and_cond = name_cond & age_cond
    assert isinstance(and_cond, CombinedCondition)
    assert and_cond.logical_operator == "and"
    assert and_cond.left is name_cond
    assert and_cond.right is age_cond

    # Test OR
    or_cond = name_cond | age_cond
    assert isinstance(or_cond, CombinedCondition)
    assert or_cond.logical_operator == "or"
    assert or_cond.left is name_cond
    assert or_cond.right is age_cond

    # Test chaining
    active_cond = Field("active") == True
    chained_and = name_cond & age_cond & active_cond
    assert isinstance(chained_and, CombinedCondition)
    assert chained_and.logical_operator == "and"
    assert isinstance(chained_and.left, CombinedCondition) # Should be nested
    assert chained_and.right is active_cond


# --- Tests for QueryBuilder ---

def test_query_builder_field_access():
    """Test accessing fields via the builder's 'fields' proxy."""
    qb_model = QueryBuilder(SampleModel)
    qb_generic = QueryBuilder() # Model-less

    # Model-based
    assert isinstance(qb_model.fields.name, Field)
    assert qb_model.fields.name.path == "name"
    assert isinstance(qb_model.fields.tags, Field)
    assert qb_model.fields.tags.path == "tags"
    with pytest.raises(AttributeError): _ = qb_model.fields.invalid_field

    # Generic
    assert isinstance(qb_generic.fields.any_field, Field)
    assert qb_generic.fields.any_field.path == "any_field"
    assert isinstance(qb_generic.fields.nested.path, Field)
    assert qb_generic.fields.nested.path.path == "nested.path"


def test_query_builder_filter_method_builds_expression():
    """Test adding filters creates the internal expression structure."""
    qb = QueryBuilder()
    name_cond = qb.fields.name == "Alice"
    age_cond = qb.fields.age > 30

    qb.filter(name_cond)
    assert qb._expression is name_cond # First filter sets the expression

    qb.filter(age_cond)
    assert isinstance(qb._expression, CombinedCondition) # Second filter combines with AND
    assert qb._expression.logical_operator == "and"
    assert qb._expression.left is name_cond
    assert qb._expression.right is age_cond


def test_query_builder_build_options():
    """Test building QueryOptions with various settings."""
    qb = QueryBuilder(SampleModel)
    options = (
        qb.filter(qb.fields.name == "Test")
        .filter(qb.fields.age < 100)
        .sort_by(qb.fields.age, descending=True)
        .limit(50)
        .offset(10)
        .timeout(15.5)
        .build()
    )

    assert isinstance(options, QueryOptions)
    assert isinstance(options.expression, QueryLogical) # Should be QueryLogical after translation
    assert options.expression.operator == "and"
    # We don't check the exact translated expression structure here,
    # that's covered by the implementation tests (test_dsl_query.py)
    assert options.sort_by == "age"
    assert options.sort_desc is True
    assert options.limit == 50
    assert options.offset == 10
    assert options.timeout == 15.5
    assert options.random_order is False


def test_query_builder_random_order_overrides_sort():
    """Test that random_order clears sort settings."""
    qb = QueryBuilder()
    options = (
        qb.sort_by(qb.fields.name)
        .random_order()
        .build()
    )
    assert options.random_order is True
    assert options.sort_by is None
    assert options.sort_desc is False


def test_query_builder_sort_validation_with_model():
    """Test validation of sort field when a model is provided."""
    qb = QueryBuilder(SampleModel)
    qb.sort_by(qb.fields.name) # Valid
    with pytest.raises(AttributeError, match="Invalid sort field path"):
        qb.sort_by(Field("invalid_field")) # Use Field directly to bypass proxy check


def test_query_builder_repr_methods():
    """Test __repr__ methods for builder components."""
    qb = QueryBuilder(SampleModel)
    field = qb.fields.name
    filter_cond = field == "Test"
    combined = filter_cond & (qb.fields.age > 18)
    options = qb.filter(combined).limit(10).build()

    assert repr(field) == "Field(path='name')"
    assert repr(filter_cond) == "FilterCondition('name', 'eq', 'Test')"
    assert "CombinedCondition('and', FilterCondition('name', 'eq', 'Test'), FilterCondition('age', 'gt', 18))" in repr(combined)
    # Check QueryOptions repr contains key parts
    options_repr = repr(options)
    assert "QueryOptions(" in options_repr
    assert "expression=QueryLogical(operator='and'," in options_repr # Checks translated output
    assert "limit=10" in options_repr