# src/async_repository/base/query.py
import logging
from typing import Any, Dict, Optional, Type, TypeVar, List, Tuple, Set, Union

# Import ModelValidator and related exceptions/helpers
from async_repository.base.model_validator import (
    ModelValidator,
    InvalidPathError,
    ValueTypeError,
    Any, # Ensure Any is imported from the right place if needed, though not strictly required here
    get_origin,
    get_args,
)


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
            f"timeout={self.timeout}, random_order={self.random_order})"
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
        # Consistently return the structured format
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
    Holds only the field name; validation happens in the QueryBuilder.

    Supports dot notation via __getattr__ for nested field access.
    """
    # Using internal _name to avoid potential clashes with actual model field names
    _name: str

    def __init__(self, name: str):
        # Use object.__setattr__ to set internal _name during initialization
        object.__setattr__(self, "_name", name)

    @property
    def name(self) -> str:
        """Public property to access the field name/path."""
        return self._name

    def __getattr__(self, name: str) -> 'Field':
        """Enable nested field access using dot notation (e.g., user.address.city)."""
        # Prevent recursion on special attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Return a *new* Field instance representing the deeper path
        return Field(f"{self._name}.{name}")

    # Prevent setting arbitrary attributes on Field instances
    def __setattr__(self, name: str, value: Any):
        if name == '_name':
             object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Cannot set attribute '{name}' on Field object")

    def __eq__(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "eq", other)

    def ne(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "ne", other)

    def __gt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "gt", other)

    def __lt__(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "lt", other)

    def __ge__(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "ge", other)

    def __le__(self, other: Any) -> FilterCondition:
        return FilterCondition(self._name, "le", other)

    def in_(self, collection: Union[List, Set, Tuple]) -> FilterCondition:
        # Basic type check here; detailed validation in QueryBuilder
        if not isinstance(collection, (list, set, tuple)):
            raise TypeError(f"in_ operator requires a list, set, or tuple, got {type(collection)}")
        # Convert set/tuple to list for consistent handling downstream if needed
        return FilterCondition(self._name, "in", list(collection))

    def nin(self, collection: Union[List, Set, Tuple]) -> FilterCondition:
        # Basic type check here; detailed validation in QueryBuilder
        if not isinstance(collection, (list, set, tuple)):
            raise TypeError(f"nin operator requires a list, set, or tuple, got {type(collection)}")
        # Convert set/tuple to list for consistent handling downstream if needed
        return FilterCondition(self._name, "nin", list(collection))

    def like(self, pattern: str) -> FilterCondition:
        if not isinstance(pattern, str):
            raise TypeError(f"like operator requires a string pattern, got {type(pattern)}")
        return FilterCondition(self._name, "like", pattern)

    def contains(self, value: Any) -> FilterCondition:
        """
        Checks if the field (as a list/array) contains the specified value.
        Validation that the field is actually a list happens in QueryBuilder.
        """
        return FilterCondition(self._name, "contains", value)

    def startswith(self, substring: str) -> FilterCondition:
        if not isinstance(substring, str):
            raise TypeError(f"startswith operator requires a string substring, got {type(substring)}")
        return FilterCondition(self._name, "startswith", substring)

    def endswith(self, substring: str) -> FilterCondition:
        if not isinstance(substring, str):
            raise TypeError(f"endswith operator requires a string substring, got {type(substring)}")
        return FilterCondition(self._name, "endswith", substring)

    def exists(self, exists: bool = True) -> FilterCondition:
        # Basic type check here; detailed validation in QueryBuilder
        if not isinstance(exists, bool):
            raise TypeError(f"exists operator requires a boolean value, got {type(exists)}")
        return FilterCondition(self._name, "exists", exists)

    def regex(self, pattern: str) -> FilterCondition:
        if not isinstance(pattern, str):
            raise TypeError(f"regex operator requires a string pattern, got {type(pattern)}")
        return FilterCondition(self._name, "regex", pattern)

    def __repr__(self) -> str:
        # Use internal _name for representation
        return f"Field({self._name!r})"


###############################################################################
# Query Builder
###############################################################################
T = TypeVar("T")  # Generic type for the model


class QueryBuilder:
    """
    Fluent query builder that dynamically generates Field instances, validates
    them against a model type (if provided) using ModelValidator, and supports
    chaining additional query parameters like sorting and pagination.
    """

    def __init__(self, model: Optional[Type[T]] = None):
        """
        Initializes the QueryBuilder.

        Args:
            model: Optional class definition (e.g., Pydantic model, dataclass)
                   for field and value validation during query building.
        """
        self._model = model
        self._validator: Optional[ModelValidator] = None
        self._field_cache: Dict[str, Field] = {} # Cache for base fields
        self._expression: Optional[Expression] = None
        self._options = QueryOptions()
        self._logger = logging.getLogger(__name__)

        if model:
            try:
                self._validator = ModelValidator(model)
            except TypeError as e:
                # Log a warning or handle cases where validation can't be set up
                self._logger.warning(
                    f"Could not initialize ModelValidator for type {model}: {e}. "
                    "Query validation will be disabled."
                )
                self._model = None  # Ensure validation is skipped later

    # --- REVERTED __getattr__ ---
    def __getattr__(self, name: str) -> Field:
        """
        Dynamically creates Field instances for attribute access.
        Validates the *base* field name exists if a validator is configured.
        Relies on Field.__getattr__ for nested paths.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Validate only the first part of the path if validator exists
        base_field_name = name.split(".")[0]
        if self._validator:
            try:
                # Check if the first part of the path exists in the model.
                self._validator.get_field_type(base_field_name)
            except InvalidPathError as e:
                model_name = self._model.__name__ if self._model else "model"
                raise AttributeError(
                    # Error message reflects base field access failure
                    f"'{model_name}' has no attribute '{base_field_name}'. Original error: {e}"
                ) from e
            except ValueError as e: # Catch potential errors like empty path ''
                 raise AttributeError(f"Invalid field name requested: '{name}'. Error: {e}") from e


        # Cache and return the Field object for the potentially full path.
        # Field.__getattr__ will handle nested access from here.
        # Cache only the base Field and let Field.__getattr__ create nested ones dynamically.
        if base_field_name not in self._field_cache:
             self._field_cache[base_field_name] = Field(base_field_name)

        # Start with the cached base field
        field_obj = self._field_cache[base_field_name]
        # If the requested name has dots, use Field.__getattr__ to get the nested field
        path_parts = name.split(".")[1:]
        for part in path_parts:
             # This will call Field.__getattr__ recursively
             # No validation here; validation happens in filter/sort_by
             field_obj = getattr(field_obj, part)

        return field_obj
    # --- END REVERT ---

    def filter(self, expr: Expression) -> "QueryBuilder":
        """
        Adds a filter expression, validating it against the model if a validator is configured.

        Args:
            expr: The Expression object (FilterCondition or CombinedCondition) to add.

        Raises:
            ValueError: If the expression is invalid according to the ModelValidator.
        """
        if self._validator:
            try:
                self._validate_expression(expr)
            # Catch specific validation errors first
            except (InvalidPathError, ValueTypeError) as e:
                # Re-raise validation errors with context from ModelValidator, wrapped in ValueError
                raise ValueError(f"Invalid filter condition: {e}") from e
            # Catch potential TypeErrors from Field basic checks or validator internals
            except TypeError as e:
                 raise ValueError(f"Type error during validation setup: {e}") from e
            # Catch other ValueErrors (e.g., re-raised from _validate_expression)
            except ValueError as e:
                 raise e # Propagate the specific error

        if self._expression is None:
            self._expression = expr
        else:
            # Combine using the expression's __and__ method
            self._expression = self._expression & expr
        return self

    def _validate_expression(self, expr: Expression) -> None:
        """
        Recursively validates an Expression using the ModelValidator.
        Relies on ModelValidator for type checks against the model schema.
        """
        if not self._validator: # Should only happen if model initialization failed
            return

        if isinstance(expr, FilterCondition):
            field_path = expr.field
            operator = expr.operator
            value = expr.value

            # --- Path Validation (Always perform this first) ---
            try:
                expected_field_type = self._validator.get_field_type(field_path)
            except InvalidPathError as e:
                # Re-raise as ValueError consistent with filter's error wrapping
                raise ValueError(f"Invalid field path used in filter: {e}") from e

            # --- Operator-Specific Validation ---

            # 1. 'exists': Only path validation is needed. Value type checked in Field.exists.
            if operator == "exists":
                # Path validation already done above.
                return # Skip further value validation

            # 2. Numeric operators: Check field is numeric, check value is numeric.
            if operator in ("gt", "lt", "ge", "le"):
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError( # Use ValueTypeError consistent with ModelValidator
                        f"Operator '{operator}' requires a numeric field, but '{field_path}' is not (type: {expected_field_type})."
                    )
                # Check the value itself is suitable for numeric comparison (int/float, not bool)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueTypeError(
                        f"Operator '{operator}' requires a numeric value (int/float) for comparison, got {type(value).__name__}."
                    )
                # No need for validate_value(value, expected_field_type) here for comparisons
                return # Validation done

            # 3. 'in'/'nin': Validate items in the collection against the field's item type.
            elif operator in ("in", "nin"):
                 # Basic check for list/set/tuple happens in Field.in_/nin

                 # Determine the expected type of items *within* the field
                 field_origin = get_origin(expected_field_type)
                 field_args = get_args(expected_field_type)
                 expected_item_type = Any # Default

                 if field_origin in (list, set) and field_args:
                     expected_item_type = field_args[0] # List[T] or Set[T]
                 elif field_origin is tuple and field_args:
                     if len(field_args) == 2 and field_args[1] == ...:
                         expected_item_type = field_args[0] # Tuple[T, ...]
                     else:
                         # Fixed tuple Tuple[A, B]. 'in' checks against elements. Use Union.
                         expected_item_type = Union[field_args]
                 elif expected_field_type is Any:
                      expected_item_type = Any
                 else:
                      # Allow 'in' check against non-collection field types (e.g., age.in_([20, 30]))
                      # The items must match the field's type.
                      expected_item_type = expected_field_type

                 # Validate each item in the collection against the determined expected item type
                 if expected_item_type is not Any:
                    for i, item in enumerate(value):
                        # Use validate_value for each item
                        self._validator.validate_value(
                            item, expected_item_type, f"'{field_path}' item {i} for '{operator}'"
                        )
                 return # Item validation done

            # 4. 'contains': Field must be List, value validated against list item type.
            elif operator == "contains":
                is_list, item_type = self._validator.get_list_item_type(field_path)
                if not is_list:
                    # Raise InvalidPathError as the operator isn't applicable
                    raise InvalidPathError(
                        f"Operator 'contains' requires a List field, but '{field_path}' is not."
                    )
                if item_type is not Any:
                    # Use validate_value for the item being searched for
                    self._validator.validate_value(
                        value, item_type, f"Value for '{field_path}' contains"
                    )
                return # Validation done

            # 5. String operators: Validate value is a string.
            elif operator in ("like", "startswith", "endswith", "regex"):
                 # Basic type check done in Field methods.
                 # Use validate_value here for consistency checks if needed,
                 # but primarily rely on Field's TypeError for non-strings.
                 # If Field check passed, assume value is a string.
                 # Optional: Could still check if expected_field_type is compatible.
                 return # Value type check done in Field

            # 6. Default ('eq', 'ne'): Validate value against the field's full type.
            # This is the fallback for operators not handled above.
            else:
                # Perform the standard validation using the ModelValidator
                self._validator.validate_value(value, expected_field_type, field_path)


        elif isinstance(expr, CombinedCondition):
            # Recursively validate left and right sides
            self._validate_expression(expr.left)
            self._validate_expression(expr.right)
        else:
            # Should not happen with current structure
            raise TypeError(f"Unsupported expression type for validation: {type(expr)}")


    def sort_by(self, field: str, descending: bool = False) -> "QueryBuilder":
        """
        Sets the sort field and order, validating the field if a model validator is present.

        Args:
            field: The field name (dot notation supported) to sort by.
            descending: Whether to sort in descending order.

        Raises:
            AttributeError: If the field does not exist in the model.
        """
        if self._validator:
            try:
                # Validate that the field path exists in the model
                self._validator.get_field_type(field)
            except InvalidPathError as e:
                model_name = self._model.__name__ if self._model else "model"
                # Raise AttributeError for sort field errors, include original error
                raise AttributeError(
                    f"'{model_name}' has no sortable attribute or valid path '{field}'. Validation failed: {e}"
                ) from e
            except ValueError as e: # Catch errors like empty path
                 raise AttributeError(f"Invalid field path for sorting: '{field}'. Error: {e}") from e

        self._options.sort_by = field
        self._options.sort_desc = descending
        return self

    def random_order(self) -> "QueryBuilder":
        """
        Enables random ordering of the results. Overrides any sort_by setting.
        """
        self._options.random_order = True
        self._options.sort_by = None # Explicitly clear sort field
        return self

    def limit(self, limit: int) -> "QueryBuilder":
        """
        Sets the limit for pagination.
        """
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Limit must be a non-negative integer.")
        self._options.limit = limit
        return self

    def offset(self, offset: int) -> "QueryBuilder":
        """
        Sets the offset for pagination.
        """
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        self._options.offset = offset
        return self

    def set_timeout(self, timeout: float) -> "QueryBuilder":
        """
        Sets the query timeout.
        """
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("Timeout must be a positive number.")
        self._options.timeout = timeout
        return self

    def build(self) -> QueryOptions:
        """
        Builds and returns a QueryOptions object based on the accumulated filters
        and parameters.
        """
        if self._expression:
            # Convert the final expression tree to the dictionary format
            self._options.expression = self._expression.to_dict()
        else:
            # Ensure expression is an empty dict if no filters were added
            self._options.expression = {}

        self._logger.debug("Built QueryOptions: %s", self._options)
        # Return a new instance to prevent modification after build
        return QueryOptions(
             expression=self._options.expression,
             sort_by=self._options.sort_by,
             sort_desc=self._options.sort_desc,
             limit=self._options.limit,
             offset=self._options.offset,
             random_order=self._options.random_order,
             timeout=self._options.timeout,
        )