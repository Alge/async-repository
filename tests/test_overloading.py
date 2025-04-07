import pytest
from pydantic import BaseModel
from typing import Optional, List

from async_repository.base.query import (
    QueryBuilder,
    QueryOptions,
    Field,
    Expression,
    FilterCondition,
    CombinedCondition,
)


# Define a Pydantic model for testing field validation
class SampleModel(BaseModel):
    id: str
    name: str
    age: int
    email: Optional[str] = None
    tags: List[str] = []


class TestQueryDSL:
    """Tests for query DSL and QueryBuilder functionality."""

    def test_field_equality_operator(self):
        """Test the overloaded equality operator on Field."""
        field = Field("name")
        condition = field == "test"
        assert isinstance(condition, FilterCondition)
        assert condition.field == "name"
        assert condition.operator == "eq"
        assert condition.value == "test"
        assert condition.to_dict() == {"name": "test"}

    def test_field_comparison_operators(self):
        """Test all comparison operators on Field."""
        field = Field("age")

        gt_condition = field > 18
        assert gt_condition.operator == "gt"
        assert gt_condition.value == 18
        assert gt_condition.to_dict() == {"age": {"operator": "gt", "value": 18}}

        lt_condition = field < 65
        assert lt_condition.operator == "lt"
        assert lt_condition.value == 65
        assert lt_condition.to_dict() == {"age": {"operator": "lt", "value": 65}}

        ge_condition = field >= 21
        assert ge_condition.operator == "ge"
        assert ge_condition.value == 21
        assert ge_condition.to_dict() == {"age": {"operator": "ge", "value": 21}}

        le_condition = field <= 100
        assert le_condition.operator == "le"
        assert le_condition.value == 100
        assert le_condition.to_dict() == {"age": {"operator": "le", "value": 100}}

        ne_condition = field.ne(0)
        assert ne_condition.operator == "ne"
        assert ne_condition.value == 0
        assert ne_condition.to_dict() == {"age": {"operator": "ne", "value": 0}}

    def test_field_methods(self):
        """Test various Field methods for different operators."""
        field = Field("tags")

        in_condition = field.in_(["tag1", "tag2"])
        assert in_condition.operator == "in"
        assert in_condition.value == ["tag1", "tag2"]
        assert in_condition.to_dict() == {
            "tags": {"operator": "in", "value": ["tag1", "tag2"]}
        }

        nin_condition = field.nin(["tag3"])
        assert nin_condition.operator == "nin"
        assert nin_condition.value == ["tag3"]
        assert nin_condition.to_dict() == {
            "tags": {"operator": "nin", "value": ["tag3"]}
        }

        like_condition = field.like("tag%")
        assert like_condition.operator == "like"
        assert like_condition.value == "tag%"
        assert like_condition.to_dict() == {
            "tags": {"operator": "like", "value": "tag%"}
        }

        contains_condition = field.contains("tag")
        assert contains_condition.operator == "contains"
        assert contains_condition.value == "tag"
        assert contains_condition.to_dict() == {
            "tags": {"operator": "contains", "value": "tag"}
        }

        startswith_condition = field.startswith("t")
        assert startswith_condition.operator == "startswith"
        assert startswith_condition.value == "t"
        assert startswith_condition.to_dict() == {
            "tags": {"operator": "startswith", "value": "t"}
        }

        endswith_condition = field.endswith("g")
        assert endswith_condition.operator == "endswith"
        assert endswith_condition.value == "g"
        assert endswith_condition.to_dict() == {
            "tags": {"operator": "endswith", "value": "g"}
        }

        exists_condition = field.exists()
        assert exists_condition.operator == "exists"
        assert exists_condition.value is True
        assert exists_condition.to_dict() == {
            "tags": {"operator": "exists", "value": True}
        }

        not_exists_condition = field.exists(False)
        assert not_exists_condition.operator == "exists"
        assert not_exists_condition.value is False
        assert not_exists_condition.to_dict() == {
            "tags": {"operator": "exists", "value": False}
        }

        regex_condition = field.regex("^tag[0-9]$")
        assert regex_condition.operator == "regex"
        assert regex_condition.value == "^tag[0-9]$"
        assert regex_condition.to_dict() == {
            "tags": {"operator": "regex", "value": "^tag[0-9]$"}
        }

    def test_logical_operators(self):
        """Test logical AND and OR operators."""
        name_field = Field("name")
        age_field = Field("age")

        name_condition = name_field == "Alice"
        age_condition = age_field > 30

        # Test AND operator
        and_condition = name_condition & age_condition
        assert isinstance(and_condition, CombinedCondition)
        assert and_condition.logical_operator == "and"
        assert and_condition.left is name_condition
        assert and_condition.right is age_condition

        expected_and_dict = {
            "and": [{"name": "Alice"}, {"age": {"operator": "gt", "value": 30}}]
        }
        assert and_condition.to_dict() == expected_and_dict

        # Test OR operator
        or_condition = name_condition | age_condition
        assert isinstance(or_condition, CombinedCondition)
        assert or_condition.logical_operator == "or"
        assert or_condition.left is name_condition
        assert or_condition.right is age_condition

        expected_or_dict = {
            "or": [{"name": "Alice"}, {"age": {"operator": "gt", "value": 30}}]
        }
        assert or_condition.to_dict() == expected_or_dict

    def test_combined_condition_invalid_operator(self):
        """Test that CombinedCondition raises ValueError for invalid operators."""
        name_field = Field("name")
        age_field = Field("age")

        name_condition = name_field == "Alice"
        age_condition = age_field > 30

        with pytest.raises(ValueError, match="logical_operator must be 'and' or 'or'"):
            CombinedCondition("invalid", name_condition, age_condition)

    def test_complex_expressions(self):
        """Test building complex expressions with multiple conditions and nested logic."""
        qb = QueryBuilder()

        # (name == "Alice" OR name == "Bob") AND (age >= 18 AND age <= 65)
        name_alice = qb.name == "Alice"
        name_bob = qb.name == "Bob"
        age_min = qb.age >= 18
        age_max = qb.age <= 65

        name_condition = name_alice | name_bob
        age_condition = age_min & age_max
        combined = name_condition & age_condition

        expected = {
            "and": [
                {"or": [{"name": "Alice"}, {"name": "Bob"}]},
                {
                    "and": [
                        {"age": {"operator": "ge", "value": 18}},
                        {"age": {"operator": "le", "value": 65}},
                    ]
                },
            ]
        }

        assert combined.to_dict() == expected

    def test_query_builder_with_model(self):
        """Test QueryBuilder with model validation."""
        qb = QueryBuilder(SampleModel)

        # Valid fields should work
        qb.id == "123"
        qb.name == "Test"
        qb.age > 18

        # Invalid field should raise AttributeError
        with pytest.raises(AttributeError):
            qb.invalid_field == "test"

    def test_query_builder_filter_method(self):
        """Test adding filters to the QueryBuilder."""
        qb = QueryBuilder()

        # Add a single filter
        qb.filter(qb.name == "Alice")

        # Add another filter (should be combined with AND)
        qb.filter(qb.age > 30)

        options = qb.build()
        expected = {
            "and": [{"name": "Alice"}, {"age": {"operator": "gt", "value": 30}}]
        }

        assert options.expression == expected

    def test_query_builder_chaining(self):
        """Test chaining methods on QueryBuilder."""
        qb = QueryBuilder(SampleModel)

        result = (
            qb.filter(qb.name == "Alice")
            .filter(qb.age > 30)
            .sort_by("name")
            .limit(10)
            .offset(5)
            .set_timeout(5.0)
            .build()
        )

        assert isinstance(result, QueryOptions)
        assert "and" in result.expression
        assert result.sort_by == "name"
        assert result.sort_desc is False
        assert result.limit == 10
        assert result.offset == 5
        assert result.timeout == 5.0

    def test_query_builder_sort_validation(self):
        """Test validation of sort field against model."""
        qb = QueryBuilder(SampleModel)

        # Valid sort field
        qb.sort_by("name")

        # Invalid sort field
        with pytest.raises(AttributeError):
            qb.sort_by("invalid_field")

    def test_random_order(self):
        """Test setting random order flag."""
        qb = QueryBuilder()

        # Default should be False
        assert qb.build().random_order is False

        # Set to True
        qb.random_order()
        assert qb.build().random_order is True

    def test_sort_by_descending(self):
        """Test sort_by with descending flag."""
        qb = QueryBuilder()
        qb.sort_by("name", descending=True)
        options = qb.build()

        assert options.sort_by == "name"
        assert options.sort_desc is True

    def test_nested_logical_combinations(self):
        """Test nesting multiple logical combinations."""
        name_field = Field("name")
        age_field = Field("age")
        email_field = Field("email")

        # (name == "Alice" OR name == "Bob") AND
        # (age > 20 OR (age < 18 AND email.exists()))
        cond1 = name_field == "Alice"
        cond2 = name_field == "Bob"
        cond3 = age_field > 20
        cond4 = age_field < 18
        cond5 = email_field.exists()

        expr1 = cond1 | cond2
        expr2 = cond4 & cond5
        expr3 = cond3 | expr2
        final = expr1 & expr3

        expected = {
            "and": [
                {"or": [{"name": "Alice"}, {"name": "Bob"}]},
                {
                    "or": [
                        {"age": {"operator": "gt", "value": 20}},
                        {
                            "and": [
                                {"age": {"operator": "lt", "value": 18}},
                                {"email": {"operator": "exists", "value": True}},
                            ]
                        },
                    ]
                },
            ]
        }

        assert final.to_dict() == expected

    def test_query_builder_without_expression(self):
        """Test QueryBuilder.build() when no expressions have been added."""
        qb = QueryBuilder()
        options = qb.build()

        assert options.expression == {}

    def test_query_options_repr(self):
        """Test the __repr__ method of QueryOptions."""
        options = QueryOptions(
            expression={"name": "Alice"},
            sort_by="age",
            sort_desc=True,
            limit=25,
            offset=10,
            random_order=True,
            timeout=30.0,
        )

        repr_str = repr(options)
        assert "expression={'name': 'Alice'}" in repr_str
        assert "sort_by=age" in repr_str
        assert "sort_desc=True" in repr_str
        assert "limit=25" in repr_str
        assert "offset=10" in repr_str
        assert "timeout=30.0" in repr_str
        assert "random_order=True" in repr_str

    def test_field_repr(self):
        """Test the __repr__ method of Field."""
        field = Field("name")
        assert repr(field) == "Field('name')"

    def test_filter_condition_repr(self):
        """Test the __repr__ method of FilterCondition."""
        condition = FilterCondition("name", "eq", "Alice")
        assert repr(condition) == "FilterCondition('name', 'eq', 'Alice')"

    def test_combined_condition_repr(self):
        """Test the __repr__ method of CombinedCondition."""
        cond1 = FilterCondition("name", "eq", "Alice")
        cond2 = FilterCondition("age", "gt", 30)
        combined = CombinedCondition("and", cond1, cond2)

        assert "CombinedCondition('and'" in repr(combined)
        assert "FilterCondition('name', 'eq', 'Alice')" in repr(combined)
        assert "FilterCondition('age', 'gt', 30)" in repr(combined)

    def test_query_builder_field_caching(self):
        """Test that QueryBuilder caches Field instances."""
        qb = QueryBuilder()
        field1 = qb.name
        field2 = qb.name

        # Should be the same instance
        assert field1 is field2

    def test_empty_expression_to_dict(self):
        """Test that Expression.to_dict() is properly implemented in subclasses."""

        class TestExpression(Expression):
            pass

        with pytest.raises(
            NotImplementedError, match="Subclasses must implement to_dict()"
        ):
            TestExpression().to_dict()

    def test_query_options_defaults(self):
        """Test that QueryOptions uses sensible defaults."""
        options = QueryOptions()

        assert options.expression == {}
        assert options.sort_by is None
        assert options.sort_desc is False
        assert options.limit == 100
        assert options.offset == 0
        assert options.timeout is None
        assert options.random_order is False
