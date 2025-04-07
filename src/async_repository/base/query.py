import asyncio
import logging
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel


###############################################################################
# Query Options
###############################################################################
class QueryOptions:
    """
    Options for querying repositories, including nested DSL expressions,
    sorting, pagination, and additional parameters.
    """

    def __init__(
        self,
        expression: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_desc: bool = False,
        limit: int = 100,
        offset: int = 0,
        random_order=False,
        timeout: Optional[float] = None,
    ):
        self.expression = expression or {}
        self.sort_by = sort_by
        self.sort_desc = sort_desc
        self.limit = limit
        self.offset = offset
        self.timeout = timeout
        self.random_order = random_order

    def __repr__(self):
        return (
            f"QueryOptions(expression={self.expression}, sort_by={self.sort_by}, "
            f"sort_desc={self.sort_desc}, limit={self.limit}, offset={self.offset}, "
            f"timeout={self.timeout}), random_order={self.random_order}"
        )


###############################################################################
# DSL Implementation
###############################################################################
class Expression:
    """
    Base class for query expressions.
    """

    def __and__(self, other: "Expression") -> "Expression":
        return CombinedCondition("and", self, other)

    def __or__(self, other: "Expression") -> "Expression":
        return CombinedCondition("or", self, other)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict()")


class FilterCondition(Expression):
    """
    Represents a simple filter condition.
    """

    def __init__(self, field: str, operator: str, value: Any):
        self.field = field
        self.operator = operator
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        if self.operator == "eq":
            return {self.field: self.value}
        else:
            return {self.field: {"operator": self.operator, "value": self.value}}

    def __repr__(self) -> str:
        return f"FilterCondition({self.field!r}, {self.operator!r}, {self.value!r})"


class CombinedCondition(Expression):
    """
    Represents a combined filter condition using logical 'and' or 'or'.
    """

    def __init__(self, logical_operator: str, left: Expression, right: Expression):
        if logical_operator not in ("and", "or"):
            raise ValueError("logical_operator must be 'and' or 'or'")
        self.logical_operator = logical_operator
        self.left = left
        self.right = right

    def to_dict(self) -> Dict[str, Any]:
        return {self.logical_operator: [self.left.to_dict(), self.right.to_dict()]}

    def __repr__(self) -> str:
        return f"CombinedCondition({self.logical_operator!r}, {self.left!r}, {self.right!r})"


class Field:
    """
    Represents a model field for building query conditions.
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "eq", other)

    def ne(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "ne", other)

    def __gt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "gt", other)

    def __lt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "lt", other)

    def __ge__(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "ge", other)

    def __le__(self, other: Any) -> FilterCondition:
        return FilterCondition(self.name, "le", other)

    def in_(self, collection: Any) -> FilterCondition:
        return FilterCondition(self.name, "in", collection)

    def nin(self, collection: Any) -> FilterCondition:
        return FilterCondition(self.name, "nin", collection)

    def like(self, pattern: str) -> FilterCondition:
        return FilterCondition(self.name, "like", pattern)

    def contains(self, value: Any) -> FilterCondition:
        """
        Checks if the field (as a list/array) contains the specified value.
        For list/array fields, this checks if the value is an element of the list.
        """
        return FilterCondition(self.name, "contains", value)

    def startswith(self, substring: str) -> FilterCondition:
        return FilterCondition(self.name, "startswith", substring)

    def endswith(self, substring: str) -> FilterCondition:
        return FilterCondition(self.name, "endswith", substring)

    def exists(self, exists: bool = True) -> FilterCondition:
        return FilterCondition(self.name, "exists", exists)

    def regex(self, pattern: str) -> FilterCondition:
        return FilterCondition(self.name, "regex", pattern)

    def __repr__(self) -> str:
        return f"Field({self.name!r})"


###############################################################################
# Query Builder
###############################################################################
T = TypeVar("T", bound=BaseModel)


class QueryBuilder:
    """
    Fluent query builder that dynamically generates Field instances, validates
    them against a Pydantic model (if provided), and supports chaining additional
    query parameters like sorting and pagination.
    """

    def __init__(self, model: Optional[Type[T]] = None):
        """
        :param model: Optional Pydantic model class for field validation.
        """
        self._model = model
        self._field_cache: Dict[str, Field] = {}
        self._expression: Optional[Expression] = None
        self._options = QueryOptions()
        self._logger = logging.getLogger(__name__)

    def __getattr__(self, name: str) -> Field:
        # Only validate if _model has a __fields__ attribute.
        if (
            self._model
            and hasattr(self._model, "__fields__")
            and name not in self._model.__fields__
        ):
            raise AttributeError(f"'{self._model.__name__}' has no field '{name}'")
        if name not in self._field_cache:
            self._field_cache[name] = Field(name)
        return self._field_cache[name]

    def filter(self, expr: Expression) -> "QueryBuilder":
        """
        Adds a filter expression.
        """
        if self._expression is None:
            self._expression = expr
        else:
            self._expression = self._expression & expr
        return self

    def sort_by(self, field: str, descending: bool = False) -> "QueryBuilder":
        """
        Sets the sort field and order.
        """
        if self._model and field not in self._model.__fields__:
            raise AttributeError(f"'{self._model.__name__}' has no field '{field}'")
        self._options.sort_by = field
        self._options.sort_desc = descending
        return self

    def random_order(self) -> "QueryBuilder":
        """
        Enables random ordering of the results.
        """
        self._options.random_order = True
        return self

    def limit(self, limit: int) -> "QueryBuilder":
        """
        Sets the limit for pagination.
        """
        self._options.limit = limit
        return self

    def offset(self, offset: int) -> "QueryBuilder":
        """
        Sets the offset for pagination.
        """
        self._options.offset = offset
        return self

    def set_timeout(self, timeout: float) -> "QueryBuilder":
        """
        Sets the query timeout.
        """
        self._options.timeout = timeout
        return self

    def build(self) -> QueryOptions:
        """
        Builds and returns a QueryOptions object based on the accumulated filters
        and parameters.
        """
        if self._expression:
            self._options.expression = self._expression.to_dict()
        self._logger.debug("Built QueryOptions: %s", self._options)
        return self._options
