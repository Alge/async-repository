# src/async_repository/base/query.py
import copy
import logging
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from inspect import isclass
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

# Use @dataclass_transform for better IDE support with model-specific fields
from typing import dataclass_transform

# Import from local modules
from .model_validator import (
    ModelValidator,
    InvalidPathError,
    ValueTypeError,
    _is_none_type,  # Import helper from model_validator
)
from .utils import prepare_for_storage

# --- Setup Logging ---
log = logging.getLogger(__name__)

# --- Generic Type Variables ---
T = TypeVar("T")
M = TypeVar("M")


# --- Helper Functions (Kept local for clarity, though _is_none_type is also in validator) ---
# def _is_none_type(t: Optional[Type]) -> bool: # Now imported from model_validator
#     """Checks if a type is NoneType."""
#     return t is type(None)


def _origin_to_class(origin: Optional[Type]) -> Optional[Type]:
    """Maps typing origins (List, Dict) to their runtime classes (list, dict)."""
    if origin is None:
        return None
    map_ = {
        list: list,
        List: list,
        dict: dict,
        Dict: dict,
        set: set,
        Set: set,
        tuple: tuple,
        Tuple: tuple,
        Mapping: dict,
    }
    mapped = map_.get(origin, origin)
    return mapped if isclass(mapped) else origin


# --- Query Operator Enum ---
class QueryOperator(Enum):
    """Enumeration of valid query filter operators."""

    # Comparison
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"
    GTE = "ge"
    LTE = "le"
    # Membership
    IN = "in"
    NIN = "nin"
    # String/Collection Specific
    CONTAINS = "contains"
    LIKE = "like"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"
    # Existence
    EXISTS = "exists"


# --- Structured Query Expression Classes ---
@dataclass
class QueryExpression:
    """Base class for structured query filter expressions."""

    pass


@dataclass
class QueryFilter(QueryExpression):
    """Represents a single filter condition (field_path <operator> value)."""

    field_path: str
    operator: QueryOperator
    value: Any


@dataclass
class QueryLogical(QueryExpression):
    """Represents a logical combination (AND/OR) of expressions."""

    operator: Literal["and", "or"]
    conditions: List[QueryExpression] = field(default_factory=list)


# --- Query Options ---
@dataclass
class QueryOptions:
    """Options for querying repositories, including a structured expression."""

    expression: Optional[QueryExpression] = None
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: int = 100
    offset: int = 0
    random_order: bool = False
    timeout: Optional[float] = None

    def __repr__(self) -> str:
        parts = []
        if self.expression:
            parts.append(f"expression={self.expression!r}")
        if self.sort_by:
            parts.append(f"sort_by={self.sort_by!r}")
            parts.append(f"sort_desc={self.sort_desc!r}")
        parts.append(f"limit={self.limit!r}")
        parts.append(f"offset={self.offset!r}")
        if self.random_order:
            parts.append(f"random_order={self.random_order!r}")
        if self.timeout is not None:
            parts.append(f"timeout={self.timeout!r}")
        return f"QueryOptions({', '.join(parts)})"

    def copy(self) -> "QueryOptions":
        """Creates a shallow copy of the QueryOptions."""
        # Use asdict for simple shallow copy of dataclass fields
        return copy.copy(self)


# --- Internal Expression Classes (Used by Builder API) ---
class Expression:
    """Base class for internal query expressions (used by builder)."""

    def __and__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with AND: {self!r} & {other!r}")
        return CombinedCondition("and", self, other)

    def __or__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with OR: {self!r} | {other!r}")
        return CombinedCondition("or", self, other)


class FilterCondition(Expression, Generic[T]):
    """Represents an internal filter condition (field OP value)."""

    field_path: str
    operator: str  # Keep internal operator as string
    value: Any

    def __init__(self, field_path: str, operator: str, value: Any):
        self.field_path = field_path
        self.operator = operator
        self.value = value

    def __repr__(self) -> str:
        return (
            f"FilterCondition({self.field_path!r}, {self.operator!r}, "
            f"{self.value!r})"
        )


class CombinedCondition(Expression):
    """Combines two internal expressions with AND or OR."""

    logical_operator: str
    left: Expression
    right: Expression

    def __init__(self, logical_operator: str, left: Expression, right: Expression):
        if logical_operator not in ("and", "or"):
            raise ValueError("logical_operator must be 'and' or 'or'")
        self.logical_operator = logical_operator
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return (
            f"CombinedCondition({self.logical_operator!r}, {self.left!r}, "
            f"{self.right!r})"
        )


# --- Field Representation ---
class Field(Generic[T]):
    """Represents a queryable field path."""

    _path: str

    def __init__(self, path: str):
        log.debug(f"Creating Field instance for path: '{path}'")
        object.__setattr__(self, "_path", path)

    @property
    def path(self) -> str:
        return self._path

    def _op(self, op_name: str, other: Any) -> FilterCondition[T]:
        """Helper to create FilterCondition."""
        log.debug(f"Creating filter: Field('{self._path}') {op_name} {repr(other)}")
        if op_name in ("like", "startswith", "endswith") and not isinstance(other, str):
            raise TypeError(f"Operator '{op_name}' requires a string value")
        if op_name in ("in", "nin") and not isinstance(other, (list, set, tuple)):
            raise TypeError(f"Operator '{op_name}' requires a list/set/tuple")
        if op_name == "exists" and not isinstance(other, bool):
            raise TypeError(f"Operator '{op_name}' requires a boolean value")

        value_to_use = other
        if isinstance(other, (set, tuple)) and op_name in ("in", "nin"):
            value_to_use = list(other)
            log.debug(f"Converted {type(other).__name__} to list for '{op_name}'")

        return FilterCondition(self._path, op_name, value_to_use)

    # Comparison operators
    def __eq__(self, other: Any) -> FilterCondition[T]:
        return self._op("eq", other)

    def __ne__(self, other: Any) -> FilterCondition[T]:
        return self._op("ne", other)

    def __gt__(self, other: Any) -> FilterCondition[T]:
        return self._op("gt", other)

    def __lt__(self, other: Any) -> FilterCondition[T]:
        return self._op("lt", other)

    def __ge__(self, other: Any) -> FilterCondition[T]:
        return self._op("ge", other)

    def __le__(self, other: Any) -> FilterCondition[T]:
        return self._op("le", other)

    # Other operators
    def contains(self, item: Any) -> FilterCondition[T]:
        return self._op("contains", item)

    def like(self, pattern: str) -> FilterCondition[T]:
        return self._op("like", pattern)

    def startswith(self, prefix: str) -> FilterCondition[T]:
        return self._op("startswith", prefix)

    def endswith(self, suffix: str) -> FilterCondition[T]:
        return self._op("endswith", suffix)

    def in_(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        return self._op("in", collection)

    def nin(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        return self._op("nin", collection)

    def exists(self, exists_value: bool = True) -> FilterCondition[T]:
        return self._op("exists", exists_value)

    def __getitem__(self, key: Any) -> "Field[Any]":
        """
        Handles indexed access (e.g., field[0] or field['key']) for list/dictionary elements.
        - Integer keys construct a path like 'parent_path.index' for list indexing
        - String keys construct a path like 'parent_path.key' for dictionary access
        """
        # Handle the test cases first - removed as specific test logic should be in tests
        # if isinstance(key, str) and key == "key":
        #     raise TypeError("Field index must be an integer, got str")
        # if isinstance(key, int) and key < 0:
        #     raise IndexError(
        #         "Negative indexing is not currently supported for query fields")

        # Normal handling for production use
        if isinstance(key, int):
            if key < 0:
                raise IndexError(
                    "Negative indexing is not currently supported for query fields"
                )
            # Construct the path for index access, consistent with how the validator expects it.
            new_path = f"{self._path}.{key}"  # The validator handles numeric string parts as indices.
            log.debug(
                f"Accessing indexed field via getitem: key={key} -> new Field path '{new_path}'"
            )
            # Return a new Field object representing the indexed path.
            return Field(new_path)
        elif isinstance(key, str):
            # For string keys, use the same pattern as attribute access
            new_path = f"{self._path}.{key}"
            log.debug(
                f"Accessing field with string key: key='{key}' -> new Field path '{new_path}'"
            )
            return Field(new_path)
        else:
            # For any other type of key, raise an error
            raise TypeError(
                f"Field index must be an integer or string, got {type(key).__name__}"
            )

    def __getattr__(self, name: str) -> "Field[Any]":
        """Dynamically create nested Field objects."""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        new_path = f"{self._path}.{name}"
        log.debug(f"Accessing nested field: '{name}' -> new Field path '{new_path}'")
        return Field(new_path)

    def __repr__(self) -> str:
        return f"Field(path={self._path!r})"

    def __setattr__(self, name: str, value: Any):
        """Prevent modification after initialization."""
        if name == "_path" and hasattr(self, name):
            raise AttributeError("Cannot modify Field attributes after initialization.")
        elif name != "_path":
            raise AttributeError(f"Cannot set attribute '{name}' on Field object.")
        else:
            object.__setattr__(self, name, value)


# --- Fields Proxy Generation ---
_PROXY_CACHE: Dict[Type, SimpleNamespace] = {}


def _generate_fields_proxy(model_cls: Type[M], base_path: str = "") -> SimpleNamespace:
    """Introspects model and creates a SimpleNamespace with Field attributes."""
    # Cache key should include the base path if relevant, but for top-level it's simpler
    cache_key = model_cls if not base_path else (model_cls, base_path)

    if cache_key in _PROXY_CACHE:
        log.debug(
            f"Returning cached fields proxy object for {model_cls.__name__} (base_path='{base_path}')"
        )
        return _PROXY_CACHE[cache_key]

    log.debug(
        f"Generating fields proxy object for {model_cls.__name__} "
        f"(base_path='{base_path}')"
    )
    proxy_obj = SimpleNamespace()
    prefix = f"{base_path}." if base_path else ""
    try:
        # Attempt to get type hints, handling potential errors
        try:
            # Use localns/globalns carefully if forward refs are needed across modules
            annotations = get_type_hints(model_cls, include_extras=True)
        except (
            TypeError,
            NameError,
        ) as e:  # Catch common errors during hint resolution
            log.warning(
                f"get_type_hints failed for {model_cls.__name__} (likely forward ref or complex type): {e}. Falling back to __annotations__."
            )
            annotations = getattr(model_cls, "__annotations__", {})
        except Exception as e:  # Catch other unexpected errors
            log.error(
                f"Unexpected error in get_type_hints for {model_cls.__name__}: {e}",
                exc_info=True,
            )
            annotations = getattr(model_cls, "__annotations__", {})

        log.debug(
            f"Introspecting fields for {model_cls.__name__} (prefix='{prefix}'): "
            f"{list(annotations.keys())}"
        )
        for name, type_hint in annotations.items():
            if name.startswith("_"):
                continue
            full_path = f"{prefix}{name}"
            # Recursively generate proxy for nested dataclasses/models if needed?
            # Potentially complex, for now just create Field.
            # origin = get_origin(type_hint)
            # args = get_args(type_hint)
            # unwrapped_type = args[0] if origin is Optional and args else type_hint
            # if is_dataclass(unwrapped_type) or hasattr(unwrapped_type, 'model_fields'):
            #    nested_proxy = _generate_fields_proxy(unwrapped_type, full_path)
            #    setattr(proxy_obj, name, nested_proxy) # Or merge? Needs careful design.
            # else:
            setattr(proxy_obj, name, Field[type_hint](full_path))
            log.debug(f"Added Field for '{name}' to proxy object.")

    except Exception as e:
        log.error(
            f"Failed field proxy generation for {model_cls.__name__}", exc_info=True
        )
        raise TypeError(
            f"Could not generate query fields proxy for {model_cls.__name__}"
        ) from e

    # Only cache top-level proxies
    if not base_path:
        _PROXY_CACHE[model_cls] = proxy_obj
    log.debug(f"Finished generating fields proxy object for {model_cls.__name__}")
    return proxy_obj


# --- Generic Fields Proxy ---
class GenericFieldsProxy:
    """Creates Field instances dynamically for any attribute access."""

    __slots__ = ()  # Optimizes memory usage

    def __getattr__(self, name: str) -> Field[Any]:
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")
        log.debug(f"GenericFieldsProxy: Creating Field for attribute '{name}'")
        return Field(name)

    def __dir__(self) -> List[str]:
        # Avoid suggesting attributes that don't exist
        return []


# --- Query Builder ---
@dataclass_transform()
class QueryBuilder(Generic[M]):
    """
    Builds database query options using a fluent API. The final `build()`
    method returns a `QueryOptions` object containing a structured,
    database-agnostic `QueryExpression` for filtering.
    """

    model_cls: Optional[Type[M]]
    fields: Any  # SimpleNamespace or GenericFieldsProxy
    _validator: Optional[ModelValidator[M]]
    _expression: Optional[Expression]
    _options: Dict[str, Any]
    _logger: logging.Logger

    # Map internal string operators to QueryOperator enum members
    _OPERATOR_MAP = {op.value: op for op in QueryOperator}

    def __init__(self, model_cls: Optional[Type[M]] = None):
        self._logger = log
        self.model_cls = model_cls
        self._expression = None
        self._options = {
            "limit": 100,
            "offset": 0,
            "sort_desc": False,
            "sort_by": None,
            "random_order": False,
            "timeout": None,
        }
        self._validator = None
        self.fields = None

        if model_cls:
            self._logger.info(
                f"Initializing QueryBuilder WITH validation for: {model_cls.__name__}"
            )
            try:
                self.fields = _generate_fields_proxy(model_cls)
                self._validator = ModelValidator(model_cls)
                self._logger.info(f"QueryBuilder for {model_cls.__name__} initialized.")
            except Exception as e:
                self._logger.error(
                    f"Failed QueryBuilder init for {model_cls.__name__}",
                    exc_info=True,
                )
                # Fallback to generic proxy if model-specific fails
                self.fields = GenericFieldsProxy()
                self._validator = None  # Ensure validator is None if proxy failed
                raise ValueError(
                    f"Could not initialize QueryBuilder for {model_cls.__name__}"
                ) from e
        else:
            self._logger.info("Initializing QueryBuilder WITHOUT model validation.")
            self.fields = GenericFieldsProxy()
            self._validator = None
        self._logger.debug(f"Default query options set: {self._options}")

    def filter(self, expr: Expression) -> "QueryBuilder[M]":
        """Adds a filter expression (combined with AND if one exists)."""
        self._logger.debug(f"Adding filter expression: {expr!r}")
        if not isinstance(expr, Expression):
            raise TypeError(
                f"filter() requires an Expression object, got {type(expr).__name__}"
            )

        if self._validator:
            try:
                self._validate_expression_runtime(expr)
                self._logger.debug(f"Runtime validation successful for: {expr!r}")
            except (ValueTypeError, InvalidPathError) as e:
                # Wrap validation errors in a ValueError for a consistent public API
                raise ValueError(f"Invalid filter expression: {e}") from e
            except Exception as e:
                # Wrap unexpected errors in a RuntimeError
                self._logger.error(
                    f"Unexpected error during filter validation: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected error validating filter {expr!r}: {e}"
                ) from e
        else:
            self._logger.debug("Skipping filter validation (no model validator).")

        if self._expression is None:
            self._expression = expr
        else:
            # Combine with existing expression using AND
            self._expression = self._expression & expr
        self._logger.debug(f"Current internal expression is now: {self._expression!r}")
        return self

    # --- Refactored Validation Logic ---

    def _validate_expression_runtime(self, expr: Expression) -> None:
        """Internal: Validates expression if validator is present."""
        assert (
            self._validator is not None
        ), "_validate_expression_runtime called without validator"

        if isinstance(expr, CombinedCondition):
            # Recursively validate combined conditions
            self._validate_expression_runtime(expr.left)
            self._validate_expression_runtime(expr.right)
        elif isinstance(expr, FilterCondition):
            try:
                # Delegate the actual validation logic
                self._validate_filter_condition(expr)
            except (ValueTypeError, InvalidPathError) as e:
                # Let specific validation errors propagate directly
                raise e
            except Exception as e:
                # Catch unexpected errors during the validation process
                log.error(
                    f"Unexpected error during runtime validation of condition ({expr!r}): {e}",
                    exc_info=True,
                )
                # Wrap unexpected errors in a RuntimeError for clarity
                raise RuntimeError(
                    f"Unexpected error validating condition {expr!r}: {e}"
                ) from e
        else:
            # Handle unknown expression types
            raise TypeError(
                f"Cannot validate unknown expression type: {type(expr).__name__}"
            )

    def _validate_filter_condition(self, condition: FilterCondition) -> None:
        """Validates a single FilterCondition against the model."""
        field_path = condition.field_path
        operator = condition.operator
        value = condition.value  # Convenience

        log.debug(f"Runtime validating FilterCondition: {condition!r}")

        # 1. Get Field Type (This implicitly validates the path)
        field_type = self._validator.get_field_type(field_path)
        log.debug(f"  Field '{field_path}' expected type: {field_type!r}")

        # 2. Operator-Specific Logic & Validation (that might short-circuit)
        if operator == QueryOperator.EXISTS.value:
            if not isinstance(value, bool):  # Basic type check for the operator itself
                raise ValueTypeError(
                    f"Operator 'exists' requires a boolean value, got {type(value).__name__}"
                )
            return  # EXISTS validation complete

        if operator in (QueryOperator.IN.value, QueryOperator.NIN.value):
            self._validate_in_nin_op(condition, field_type)
            return  # IN/NIN handled value validation

        if operator == QueryOperator.CONTAINS.value:
            self._validate_contains_op(condition, field_type)
            # CONTAINS handles validation internally if applicable (list/set/str)
            # If it falls through (not list/set/str), general validation below applies
            if get_origin(field_type) in (list, List, set, Set) or field_type is str:
                return

        # --- If execution reaches here, it's not EXISTS, IN, NIN, or handled CONTAINS ---

        # 3. General Value Validation (Check if the provided value fits the field's type)
        # This is crucial for operators like EQ, NE, GT, LT, etc., and fallback for CONTAINS
        self._validate_basic_value(
            condition, field_type
        )  # Raises ValueTypeError if value is wrong type for field

        # 4. Operator/Field Type Compatibility Checks
        # (Only run if the basic value validation passed)
        numeric_compare_ops = (
            QueryOperator.GT.value,
            QueryOperator.LT.value,
            QueryOperator.GTE.value,
            QueryOperator.LTE.value,
        )
        string_compare_ops = (
            QueryOperator.LIKE.value,
            QueryOperator.STARTSWITH.value,
            QueryOperator.ENDSWITH.value,
        )

        if operator in numeric_compare_ops:
            self._check_numeric_op_vs_type(condition, field_type)
        elif operator in string_compare_ops:
            self._check_string_op_vs_type(
                condition, field_type
            )  # Raises ValueTypeError if incompatible

    def _validate_in_nin_op(self, condition: FilterCondition, field_type: Type):
        """Validates items for IN/NIN operators."""
        field_path = condition.field_path
        operator = condition.operator
        value = condition.value
        assert operator in (QueryOperator.IN.value, QueryOperator.NIN.value)
        assert self._validator is not None

        is_list_field, item_type_list = self._validator.get_list_item_type(field_path)
        is_set_field = False
        item_type_set = Any
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Check for Set type explicitly if not List
        if not is_list_field:
            current_origin = origin
            current_args = args
            # Handle Optional[Set[T]]
            if current_origin is Union and any(_is_none_type(a) for a in current_args):
                non_none = tuple(t for t in current_args if not _is_none_type(t))
                if len(non_none) == 1:
                    inner_type = non_none[0]
                    current_origin = get_origin(inner_type)
                    current_args = get_args(inner_type)
            # Check if the (potentially unwrapped) type is a Set
            if current_origin in (set, Set) and current_args:
                is_set_field = True
                item_type_set = current_args[0]

        # Determine the type against which items in the input list should be validated
        expected_item_type = Any
        if is_list_field:
            expected_item_type = item_type_list
        elif is_set_field:
            expected_item_type = item_type_set
        else:
            # Field is not List or Set, maybe validate against the field type itself?
            # This might be needed for DBs supporting IN on non-array fields.
            expected_item_type = field_type
            log.warning(
                f"Applying '{operator}' to non-list/set field '{field_path}' ({field_type!r}). Validating items against field type."
            )

        log.debug(
            f"  Validating '{operator}' items against type: {expected_item_type!r}"
        )

        if not isinstance(value, (list, set, tuple)):
            # The Field._op check should catch this, but double-check here.
            raise TypeError(
                f"Operator '{operator}' requires a list/set/tuple value, got {type(value).__name__}"
            )

        if value:  # value is the list/set/tuple provided to IN/NIN
            for i, item in enumerate(value):
                try:
                    self._validator.validate_value(
                        item,
                        expected_item_type,
                        f"{field_path} ({operator} item {i})",
                    )
                except ValueTypeError as e:
                    # Re-raise with context - NO 'path=' kwarg
                    raise ValueTypeError(
                        f"Invalid item in '{operator}' list for field '{field_path}'. {e}"
                    ) from e

    def _validate_contains_op(self, condition: FilterCondition, field_type: Type):
        """Validates the value for the CONTAINS operator."""
        field_path = condition.field_path
        value = condition.value
        assert condition.operator == QueryOperator.CONTAINS.value
        assert self._validator is not None

        # Check if field is List or Set and get item type
        is_list, item_type_list = self._validator.get_list_item_type(field_path)
        is_set = False
        item_type_set = Any
        actual_field_type = field_type  # Start with the original type

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional[List/Set/str] by unwrapping
        if origin is Union and any(_is_none_type(a) for a in args):
            non_none = tuple(t for t in args if not _is_none_type(t))
            if len(non_none) == 1:
                actual_field_type = non_none[0]
                origin = get_origin(actual_field_type)
                args = get_args(actual_field_type)
                # Re-check if the unwrapped type is list/set
                if origin in (list, List) and args:
                    is_list = True
                    item_type_list = args[0]
                elif origin in (set, Set) and args:
                    is_set = True
                    item_type_set = args[0]

        # Logic if field is List or Set
        if is_list or is_set:
            item_type = item_type_list if is_list else item_type_set
            coll_name = "list" if is_list else "set"
            log.debug(
                f"  Validating 'contains' value against item type {item_type!r} for {coll_name} field."
            )
            try:
                # Validate the provided value against the *item type* of the list/set
                self._validator.validate_value(
                    value, item_type, f"{field_path} (contains value)"
                )
            except ValueTypeError as e:
                # Re-raise with context - NO 'path=' kwarg
                raise ValueTypeError(
                    f"Operator 'contains' on field '{field_path}' requires value compatible with item type {self._validator._get_type_name(item_type)}, got {type(value).__name__}. Original error: {e}"
                ) from e
            # Validation done for list/set, return to skip general validation
            return

        # Logic if field is string
        if actual_field_type is str:
            if not isinstance(value, str):
                # Error if value provided to 'contains' on string field is not a string
                raise ValueTypeError(
                    f"Operator 'contains' on string field '{field_path}' requires a string value, got {type(value).__name__}",
                )
            # Validation done for string, return to skip general validation
            return

        # If field is not list/set/string, standard validation applies (value vs field type)
        log.warning(
            f"Runtime check for 'contains' on non-list/set/string field '{field_path}' ({field_type!r}). Falling back to general value validation."
        )
        # Let execution fall through to _validate_basic_value

    def _validate_basic_value(self, condition: FilterCondition, field_type: Type):
        """Performs general value validation against the field type."""
        assert self._validator is not None
        self._validator.validate_value(
            condition.value, field_type, condition.field_path
        )

    def _check_numeric_op_vs_type(self, condition: FilterCondition, field_type: Type):
        """Checks compatibility of numeric comparison operators with the field type."""
        assert self._validator is not None
        operator = condition.operator
        field_path = condition.field_path

        # Unwrap Optional for check
        _origin = get_origin(field_type)
        _args = get_args(field_type)
        _actual_type = field_type
        if _origin is Union and any(_is_none_type(a) for a in _args):
            _non_none = tuple(t for t in _args if not _is_none_type(t))
            if len(_non_none) == 1:
                _actual_type = _non_none[0]

        # Check if the effective type is numeric or string (some DBs allow comparison)
        if (
            not self._validator._is_single_type_numeric(_actual_type)
            and _actual_type is not str
        ):
            log.warning(
                f"Operator '{operator}' used on non-numeric, non-string field '{field_path}' ({field_type!r}). Comparison support depends on backend."
            )

    def _check_string_op_vs_type(self, condition: FilterCondition, field_type: Type):
        """Checks compatibility of string operators with the field type."""
        assert self._validator is not None
        operator = condition.operator
        field_path = condition.field_path

        # Unwrap Optional for check
        _origin = get_origin(field_type)
        _args = get_args(field_type)
        _actual_type = field_type
        if _origin is Union and any(_is_none_type(a) for a in _args):
            _non_none = tuple(t for t in _args if not _is_none_type(t))
            if len(_non_none) == 1:
                _actual_type = _non_none[0]

        if _actual_type is not str:
            raise ValueTypeError(
                f"Operator '{operator}' typically requires a string field, but field '{field_path}' has type {self._validator._get_type_name(field_type)}",
            )

    # --- End of Refactored Validation Logic ---

    def sort_by(self, field: Field[Any], descending: bool = False) -> "QueryBuilder[M]":
        """Sets the sort field and order."""
        self._logger.debug(
            f"Setting sort order: field={field!r}, descending={descending}"
        )
        if not isinstance(field, Field):
            raise TypeError("sort_by requires a Field object")

        if self._validator:
            try:
                # Validate path exists, type doesn't strictly matter for sorting usually
                self._validator.get_field_type(field.path)
            except InvalidPathError as e:
                raise AttributeError(
                    f"Invalid sort field path: {field.path}. Error: {e}"
                ) from e
            except Exception as e:
                # Catch unexpected validation errors
                self._logger.error(
                    f"Unexpected error validating sort field: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected error validating sort field {field.path}: {e}"
                ) from e
        else:
            self._logger.debug("Skipping sort_by path validation.")

        self._options["sort_by"] = field.path
        self._options["sort_desc"] = descending
        # Disable random order if sort_by is set
        if self._options.get("random_order"):
            self._options["random_order"] = False
            self._logger.debug("Disabled random_order due to sort_by.")
        self._logger.info(
            f"Sort order set: field='{field.path}', descending={descending}"
        )
        return self

    def random_order(self) -> "QueryBuilder[M]":
        """Sets random ordering, clearing previous sort."""
        self._options["random_order"] = True
        self._options["sort_by"] = None
        self._options["sort_desc"] = False
        self._logger.info("Query order set to random.")
        return self

    def limit(self, num: int) -> "QueryBuilder[M]":
        """Sets the query limit."""
        if not isinstance(num, int) or num < 0:
            raise ValueError("Limit must be a non-negative integer.")
        self._options["limit"] = num
        self._logger.info(f"Query limit set to: {num}")
        return self

    def offset(self, num: int) -> "QueryBuilder[M]":
        """Sets the query offset."""
        if not isinstance(num, int) or num < 0:
            raise ValueError("Offset must be a non-negative integer.")
        self._options["offset"] = num
        self._logger.info(f"Query offset set to: {num}")
        return self

    def timeout(self, seconds: Optional[float]) -> "QueryBuilder[M]":
        """Sets the query timeout in seconds."""
        if seconds is not None and (
            not isinstance(seconds, (int, float)) or seconds < 0
        ):
            raise ValueError("Timeout must be a non-negative number or None.")
        self._options["timeout"] = seconds
        self._logger.info(f"Query timeout set to: {seconds}")
        return self

    def _translate_expression(
        self, internal_expr: Optional[Expression]
    ) -> Optional[QueryExpression]:
        """Recursively translates internal Expression tree to public QueryExpression tree."""
        if internal_expr is None:
            return None

        if isinstance(internal_expr, FilterCondition):
            operator_enum = self._OPERATOR_MAP.get(internal_expr.operator)
            if operator_enum is None:
                # Should not happen if validation worked, but handle defensively
                self._logger.warning(
                    f"Unknown internal operator string '{internal_expr.operator}' "
                    f"during translation. Skipping filter."
                )
                return None

            # Serialize value for storage/backend
            serialized_value = prepare_for_storage(internal_expr.value)
            return QueryFilter(
                field_path=internal_expr.field_path,
                operator=operator_enum,
                value=serialized_value,
            )
        elif isinstance(internal_expr, CombinedCondition):
            # Recursively translate children
            translated_left = self._translate_expression(internal_expr.left)
            translated_right = self._translate_expression(internal_expr.right)

            # Filter out None results (e.g., from skipped unknown operators)
            valid_conditions = [
                cond for cond in [translated_left, translated_right] if cond is not None
            ]

            # Handle cases where one or both children became None
            if not valid_conditions:
                return None
            if len(valid_conditions) == 1:
                return valid_conditions[0]

            # Combine valid children into a QueryLogical node
            # TODO: Consider flattening consecutive identical operators (e.g., AND(AND(a,b), c) -> AND(a,b,c))
            return QueryLogical(
                operator=internal_expr.logical_operator,  # "and" or "or"
                conditions=valid_conditions,
            )
        else:
            # This should not happen with the defined Expression types
            self._logger.error(
                f"Unsupported internal expression type during translation: "
                f"{type(internal_expr)}"
            )
            raise TypeError(
                f"Unsupported internal expression type: {type(internal_expr)}"
            )

    def build(self) -> QueryOptions:
        """Builds the final QueryOptions object with a structured QueryExpression."""
        self._logger.debug("Building QueryOptions...")
        public_expression = self._translate_expression(self._expression)

        options = QueryOptions(
            expression=public_expression,
            sort_by=self._options.get("sort_by"),
            sort_desc=self._options.get("sort_desc", False),
            limit=self._options.get("limit", 100),
            offset=self._options.get("offset", 0),
            random_order=self._options.get("random_order", False),
            timeout=self._options.get("timeout"),
        )
        model_name = self.model_cls.__name__ if self.model_cls else "Generic"
        self._logger.info(f"Built query options for {model_name} model: {options!r}")
        return options
