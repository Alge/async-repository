# src/query_builder_static.py

import logging
import operator
from typing import (
    Any, Dict, Optional, Type, TypeVar, List, Tuple, Set, Union,
    Generic, overload, TypeAlias, Protocol, ClassVar, cast
)
from typing import dataclass_transform # Key ingredient
from types import SimpleNamespace # For the fields proxy

# Assume ModelValidator exists and works with types
# from async_repository.base.model_validator import ModelValidator, InvalidPathError, ValueTypeError
# Mock validator for demonstration if needed:
class MockModelValidator(Generic[TypeVar("_M")]):
    def __init__(self, model_cls: Type[_M]): self.model_cls = model_cls
    def get_field_type(self, path: str) -> Type[Any]: print(f"Runtime check: {path}"); return Any # Simulate check
    def validate_value(self, value: Any, expected_type: Type[Any], context: str): pass # Simulate check

ModelValidator = MockModelValidator # Use the mock for now

# --- Generic Type Variables ---
T = TypeVar("T")  # Field data type (e.g., str, int, bool)
M = TypeVar("M")  # Model class type (e.g., User, Address)

# --- Query Expression Classes ---
# (Similar to previous versions, but FilterCondition can be enhanced)

class Expression:
    """Base class for query expressions."""
    def __and__(self, other: "Expression") -> "CombinedCondition":
        return CombinedCondition("and", self, other)
    def __or__(self, other: "Expression") -> "CombinedCondition":
        return CombinedCondition("or", self, other)
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict()")

class FilterCondition(Expression, Generic[T]):
    """Represents a filter condition (field OP value). Generic on field type T."""
    field_path: str
    operator: str
    value: Any # The value being compared against (might not be T)

    def __init__(self, field_path: str, operator: str, value: Any):
        self.field_path = field_path
        self.operator = operator
        self.value = value
        # TODO: Optional: Add runtime check here that 'value' is compatible
        # with 'operator' and potentially the expected type T, if T is known
        # precisely (which is hard without passing Field[T]'s T here).

    def to_dict(self) -> Dict[str, Any]:
        return {self.field_path: {"operator": self.operator, "value": self.value}}

    def __repr__(self) -> str:
        return f"FilterCondition({self.field_path!r}, {self.operator!r}, {self.value!r})"

class CombinedCondition(Expression):
    """Combines two expressions with AND or OR."""
    logical_operator: str
    left: Expression
    right: Expression

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

# --- Field Representation ---
class Field(Generic[T]):
    """
    Represents a queryable field path with type T.
    Provides methods for building FilterCondition expressions.
    """
    _path: str
    # Note: We store the path, not the actual data type T at runtime directly,
    # but the Generic T helps type checkers with comparisons.

    def __init__(self, path: str):
        # Use object.__setattr__ to initialize internal attribute
        object.__setattr__(self, "_path", path)

    @property
    def path(self) -> str:
        """The full dot-notation path to the field."""
        return self._path

    # --- Comparison Operators ---
    # These methods now return FilterCondition[T] which *could* be used
    # for more advanced type checking if the value type could be constrained.
    # For now, we keep 'other: Any' for broad compatibility.

    # Helper to create FilterCondition, reducing repetition
    def _op(self, op_name: str, other: Any) -> FilterCondition[T]:
        return FilterCondition(self._path, op_name, other)

    def __eq__(self, other: Any) -> FilterCondition[T]:
         # Ideally, 'other' should be compatible with T. Type checkers might infer this.
         # e.g., if field is Field[str], comparing field == 123 might raise a warning.
        return self._op("eq", other)

    def __ne__(self, other: Any) -> FilterCondition[T]:
        return self._op("ne", other)

    def __gt__(self, other: Any) -> FilterCondition[T]: # Requires T to be orderable
        return self._op("gt", other)

    def __lt__(self, other: Any) -> FilterCondition[T]: # Requires T to be orderable
        return self._op("lt", other)

    def __ge__(self, other: Any) -> FilterCondition[T]: # Requires T to be orderable
        return self._op("ge", other)

    def __le__(self, other: Any) -> FilterCondition[T]: # Requires T to be orderable
        return self._op("le", other)

    # --- Collection/String Operators ---
    # These might have more specific type hints depending on T

    # For Field[List[ItemT]] or Field[Set[ItemT]] etc.
    def contains(self, item: Any) -> FilterCondition[T]:
        # If T is List[ItemT], item should ideally be ItemT
        return self._op("contains", item)

    # For Field[str]
    def like(self, pattern: str) -> FilterCondition[T]:
        # Requires T to be compatible with string operations (enforced by checker)
        if not isinstance(pattern, str): raise TypeError("like requires a string pattern")
        return self._op("like", pattern)

    def startswith(self, prefix: str) -> FilterCondition[T]:
        if not isinstance(prefix, str): raise TypeError("startswith requires a string prefix")
        return self._op("startswith", prefix)

    def endswith(self, suffix: str) -> FilterCondition[T]:
        if not isinstance(suffix, str): raise TypeError("endswith requires a string suffix")
        return self._op("endswith", suffix)

    # --- General Operators ---
    def in_(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        # If T is ItemT, collection elements should ideally be ItemT
        if not isinstance(collection, (list, set, tuple)):
             raise TypeError("in_ operator requires a list, set, or tuple")
        return self._op("in", list(collection))

    def nin(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        if not isinstance(collection, (list, set, tuple)):
            raise TypeError("nin operator requires a list, set, or tuple")
        return self._op("nin", list(collection))

    def exists(self, exists_value: bool = True) -> FilterCondition[T]:
        if not isinstance(exists_value, bool):
             raise TypeError("exists operator requires a boolean value")
        return self._op("exists", exists_value)

    # --- Accessing Nested Fields (Runtime via __getattr__) ---
    # This provides the runtime mechanism for qb.fields.address.city
    # Static analysis relies on the proxy generation understanding nesting.
    def __getattr__(self, name: str) -> 'Field[Any]': # Return Field[Any] for nested unknown type
        """Allows chaining for nested fields, e.g., Field('address').city."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Return a *new* Field instance representing the deeper path.
        # We lose the specific type T here, fallback to Any.
        return Field(f"{self._path}.{name}")

    def __repr__(self) -> str:
        # Try to get the generic type for repr if possible, otherwise just use path
        type_repr = getattr(self, '__orig_class__', type(self)).__name__
        return f"{type_repr}({self._path!r})"

    # Prevent setting attributes on Field instances
    def __setattr__(self, name: str, value: Any):
        if name == '_path': # Allow internal initialization
             object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}' on Field object. Use comparison methods.")


# --- Fields Proxy Generation ---

_PROXY_CACHE: Dict[Type, Any] = {} # Cache generated proxy objects/types if needed

def _generate_fields_proxy(model_cls: Type[M], base_path: str = "") -> Any:
    """
    Introspects a model class and creates a proxy object where attributes
    are Field instances. Handles nesting.
    """
    # Use SimpleNamespace for easy attribute setting
    proxy = SimpleNamespace()
    logger = logging.getLogger(__name__)
    prefix = f"{base_path}." if base_path else ""

    # --- Introspection Logic ---
    # Needs to be robust: handle __annotations__, dataclasses, Pydantic, etc.
    try:
        # Example: Use annotations primarily
        annotations = getattr(model_cls, '__annotations__', {})
        logger.debug(f"Introspecting fields for {model_cls.__name__} (prefix='{prefix}'): {annotations.keys()}")

        for name, type_hint in annotations.items():
            if name.startswith('_'): # Skip private/protected
                continue

            full_path = f"{prefix}{name}"

            # Check for nested models (recursive step)
            # Determine the 'origin' type (e.g., list for List[Address])
            origin_type = getattr(type_hint, '__origin__', None)
            field_type_args = getattr(type_hint, '__args__', None)

            nested_model_cls = None
            if origin_type in (list, List, set, Set) and field_type_args:
                # Handle List[NestedModel], Set[NestedModel]
                # We typically query the field itself (e.g. tags.contains),
                # but maybe allow querying *into* elements? (e.g., addresses.city - less common)
                # For now, treat lists/sets as simple fields unless elements are queryable models.
                # Let's assume for now we only nest direct attributes.
                 pass # Treat as simple field for now

            elif isinstance(type_hint, type) and hasattr(type_hint, '__annotations__'):
                 # Potential nested model (check if it's intended to be queryable)
                 # This check could be more sophisticated (e.g., check for BaseModel, dataclass)
                 logger.debug(f"Field '{name}' type '{type_hint.__name__}' might be a nested model.")
                 # Recursively generate proxy for the nested model
                 nested_proxy = _generate_fields_proxy(type_hint, base_path=full_path)
                 setattr(proxy, name, nested_proxy)
                 logger.debug(f"Added nested proxy for '{full_path}'")
                 continue # Skip creating a simple Field for the container

            # Default: Create a simple Field instance
            # Use the type_hint to create Field[SpecificType]
            # This assumes Field() doesn't *need* T at runtime, only for hints
            setattr(proxy, name, Field[type_hint](full_path)) # Generic Field creation

    except Exception as e:
        logger.error(f"Failed during field proxy generation for {model_cls.__name__}: {e}", exc_info=True)
        # Decide: raise error or return empty proxy? Raising is safer.
        raise TypeError(f"Could not generate query fields proxy for {model_cls.__name__}") from e

    return proxy


# --- Query Builder ---
# Use dataclass_transform to hint the structure of 'fields' based on 'model_cls'
@dataclass_transform()
class QueryBuilder(Generic[M]):
    """
    Builds database queries in a statically analyzable way.

    Use `qb.fields.fieldname` to access queryable fields.
    Requires type checker support for PEP 681 (dataclass_transform).
    """
    model_cls: Type[M]
    fields: Any # Type checkers supporting dataclass_transform should infer this
    _validator: ModelValidator[M]
    _expression: Optional[Expression]
    _options: Dict[str, Any] # Store options like sort, limit etc.
    _logger: logging.Logger

    def __init__(self, model_cls: Type[M]):
        """
        Initializes the QueryBuilder for a specific model class.

        Args:
            model_cls: The data model class (e.g., User, Product).
        """
        self.model_cls = model_cls
        self._logger = logging.getLogger(__name__)
        self._expression = None
        self._options = {"limit": 100, "offset": 0, "sort_desc": False} # Default options

        try:
            # Generate the fields proxy object at runtime
            # Type checkers supporting dataclass_transform should use model_cls
            # passed here to infer the attributes available on self.fields
            self.fields = _generate_fields_proxy(model_cls)
            self._validator = ModelValidator(model_cls) # Initialize validator
            self._logger.debug(f"Initialized QueryBuilder for model: {model_cls.__name__}")
        except Exception as e:
            self._logger.error(f"Failed to initialize QueryBuilder for {model_cls.__name__}: {e}", exc_info=True)
            raise ValueError(f"Could not initialize QueryBuilder for {model_cls.__name__}") from e

    def filter(self, expr: Expression) -> "QueryBuilder[M]":
        """Adds a filter expression (combines with existing using AND)."""
        if not isinstance(expr, Expression):
            raise TypeError("filter() requires an Expression object "
                            "(e.g., qb.fields.name == 'value')")

        # --- Runtime Validation of the Expression ---
        # Although field paths *should* be valid if created via qb.fields,
        # this validates value types against operators.
        try:
             self._validate_expression_runtime(expr)
        except Exception as e:
             # Catch validation errors (ValueTypeError, etc.) or unexpected errors
             self._logger.warning(f"Filter validation failed: {e}")
             # Re-raise as ValueError for consistency? Or let specific errors propagate?
             raise ValueError(f"Invalid filter expression: {e}") from e

        if self._expression is None:
            self._expression = expr
        else:
            # Combine using the expression's __and__ method
            self._expression = self._expression & expr
        return self

    def _validate_expression_runtime(self, expr: Expression) -> None:
        """Performs runtime checks on expression values and operators."""
        if isinstance(expr, FilterCondition):
            # Use validator to check value compatibility with field type/operator
            # Field path is assumed correct if generated via self.fields
            field_type = self._validator.get_field_type(expr.field_path) # Get expected type
            # Example check:
            if expr.operator == "eq":
                 self._validator.validate_value(expr.value, field_type, expr.field_path)
            # ... add more checks for gt, lt, contains, etc. based on field_type ...
            self._logger.debug(f"Validated filter condition runtime: {expr}")

        elif isinstance(expr, CombinedCondition):
            self._validate_expression_runtime(expr.left)
            self._validate_expression_runtime(expr.right)


    # Use Field object directly for sorting type safety
    def sort_by(self, field: Field[Any], descending: bool = False) -> "QueryBuilder[M]":
        """Sets the sort field and order."""
        if not isinstance(field, Field):
             raise TypeError("sort_by requires a Field object (e.g., qb.fields.name)")
        # Runtime check path validity (optional, should be guaranteed by Field origin)
        try:
             self._validator.get_field_type(field.path)
        except Exception as e: # Catch InvalidPathError etc.
             raise AttributeError(f"Invalid sort field path: {field.path}. {e}") from e

        self._options['sort_by'] = field.path
        self._options['sort_desc'] = descending
        self._options['random_order'] = False
        return self

    def random_order(self) -> "QueryBuilder[M]":
        """Sets random ordering."""
        self._options['random_order'] = True
        self._options['sort_by'] = None
        return self

    def limit(self, num: int) -> "QueryBuilder[M]":
        """Sets the query limit."""
        if not isinstance(num, int) or num < 0:
            raise ValueError("Limit must be a non-negative integer.")
        self._options['limit'] = num
        return self

    def offset(self, num: int) -> "QueryBuilder[M]":
        """Sets the query offset."""
        if not isinstance(num, int) or num < 0:
            raise ValueError("Offset must be a non-negative integer.")
        self._options['offset'] = num
        return self

    def build(self) -> Dict[str, Any]: # Return a dict representing the final query
        """Builds the final query structure."""
        query = {
            "model": self.model_cls.__name__,
            "filter": self._expression.to_dict() if self._expression else {},
            "options": self._options
        }
        self._logger.debug(f"Built query: {query}")
        return query


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # --- Define Models (No modification needed) ---
    class Address:
        street: Optional[str]
        city: str

    class User:
        id: int
        username: str
        email: Optional[str]
        age: int
        is_active: bool
        tags: List[str]
        address: Address # Nested model

    # --- Query Building ---
    qb = QueryBuilder(User)

    # Access fields via qb.fields - Autocompletion and static checks work here!
    # MyPy/Pyright should catch typos like qb.fields.user_name
    filter1 = qb.fields.username == "test"
    filter2 = qb.fields.age > 30

    # Nested access
    filter3 = qb.fields.address.city.startswith("Spring")

    # Correct type comparison (Field[int] > int)
    filter4 = qb.fields.age < 100

    # Potential type warning from checker (Field[str] == int)
    filter5 = qb.fields.username == 123 # Runtime validation might also catch this

    # Using operators
    filter6 = qb.fields.tags.contains("admin")
    filter7 = qb.fields.email.exists(True)
    filter8 = qb.fields.username.in_({"user1", "user2"})

    # Build the query
    query = qb.filter(filter1 & (filter2 | filter3))\
              .filter(filter6)\
              .filter(filter7)\
              .sort_by(qb.fields.age, descending=True)\
              .limit(50)\
              .offset(10)\
              .build()

    print("\nFinal Query Structure:")
    import json
    print(json.dumps(query, indent=2))

    # --- Static Error Examples (should be caught by MyPy/Pyright) ---
    print("\nExamples of static errors (won't run if checker catches them):")
    try:
        err_field = qb.fields.non_existent_field == "oops"
        print(f"ERROR: Should have failed statically on non_existent_field: {err_field}")
    except AttributeError: # Will likely happen at runtime if checker misses it
        print("Caught expected AttributeError for non_existent_field at runtime")

    try:
        err_nested = qb.fields.address.non_existent_nested == "oops"
        print(f"ERROR: Should have failed statically on non_existent_nested: {err_nested}")
    except AttributeError:
        print("Caught expected AttributeError for non_existent_nested at runtime")

    try:
        # Comparison type mismatch (e.g., Field[Address] > int)
        # Depending on checker strictness for operators on complex types
        err_type = qb.fields.address > 10
        print(f"ERROR: Should have failed statically on address > 10: {err_type}")
    except TypeError: # Or other error depending on how Field implements __gt__
        print("Caught expected TypeError for address > 10 at runtime/statically")