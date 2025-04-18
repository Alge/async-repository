# src/query.py

import logging
import operator
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    TypeVar,
    List,
    Tuple,
    Set,
    Union,
    Generic,
    overload,
    TypeAlias,
    Protocol,
    ClassVar,
    cast,
)
from typing import dataclass_transform  # Key ingredient
from types import SimpleNamespace  # For the fields proxy

# --- Required imports for ModelValidator ---
import traceback
from dataclasses import MISSING, is_dataclass
from inspect import isclass
from typing import (
    Mapping,
    get_args,
    get_origin,
    get_type_hints,
)

# --- Setup Logging ---
# Configure logging at the application entry point or use basicConfig for standalone.
# Using __name__ ensures logs are specific to this module.
log = logging.getLogger(__name__)


# --- Generic Type Variables ---
T = TypeVar("T")  # Field data type (e.g., str, int, bool)
M = TypeVar("M")  # Model class type (e.g., User, Address)
K = TypeVar(
    "K"
)  # Key type for ModelValidator (unused here directly, but part of validator)
V = TypeVar("V")  # Value type for ModelValidator (unused here directly)


class ValidationError(TypeError):
    """Base class for validation errors."""
    def __init__(self, message: str, *, path: Optional[str] = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.path = path
        # Store any extra kwargs if needed, though often just message/path are useful
        self.details = kwargs

    def __str__(self) -> str:
        # Optionally include path in default string representation
        if self.path:
            # Check if path is already included reasonably in the message
            # This avoids duplication like "Path 'x': Path 'x': Expected..."
            path_prefix = f"Path '{self.path}':"
            if self.message.startswith(path_prefix):
                return self.message
            else:
                 # Add path context if not already present
                return f"Path '{self.path}': {self.message}"
        return self.message

class InvalidPathError(ValidationError, AttributeError):
    """Error raised when a field path does not exist or is invalid."""
    # Inherits __init__ from ValidationError


class ValueTypeError(ValidationError, TypeError):
    """Error raised when a value's type is incompatible."""
     # Inherits __init__ from ValidationError


# --- Helper Functions (copied from validator section for standalone use) ---
def _is_none_type(t: Optional[Type]) -> bool:
    return t is type(None)


def _is_typevar(t: Any) -> bool:
    return isinstance(t, TypeVar)


def _origin_to_class(origin: Optional[Type]) -> Optional[Type]:
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


# --- Model Validator (Generic - Copied and assumed available) ---
# NOTE: The full ModelValidator class is assumed to be present below this point.
# For brevity in this response, only the *interface* is shown here,
# assuming the implementation provided previously is included in the final file.
class ModelValidator(Generic[M]):
    """
    Performs validation of field paths and values against model M.
    (Implementation assumed to be included below - see original prompt)
    """

    model_type: Type[M]
    _type_hints_cache: Dict[Type, Dict[str, Type]]
    _field_aliases_cache: Dict[Type, Dict[str, str]]
    _generic_type_cache: Dict[str, Dict[TypeVar, Type]]

    def __init__(self, model_type: Type[M]):
        log.debug(f"Initializing ModelValidator for type: {model_type!r}")
        if (
            not isclass(model_type)
            and not get_origin(model_type)
            and not hasattr(model_type, "__pydantic_generic_metadata__")
        ):
            log.error(f"Invalid model_type for ModelValidator: {type(model_type)}")
            raise TypeError(
                f"model_type must be a class or generic alias, received {type(model_type)}."
            )
        self.model_type = model_type
        self._type_hints_cache = {}
        self._field_aliases_cache = {}
        self._generic_type_cache = {}
        log.debug(f"ModelValidator initialized successfully for {model_type!r}")

    def _get_type_name(self, type_obj: Any) -> str:
        # ... (full implementation assumed present) ...
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__
        return str(type_obj)

    def _get_cached_type_hints(self, cls: Type) -> Dict[str, Type]:
        # ... (full implementation assumed present) ...
        if cls in self._type_hints_cache:
            log.debug(f"Cache hit for type hints of {cls.__name__}")
            return self._type_hints_cache[cls]

        log.debug(f"Cache miss for type hints of {cls.__name__}. Fetching...")
        try:
            hints = get_type_hints(cls, include_extras=True)
            self._type_hints_cache[cls] = hints
            log.debug(
                f"Successfully fetched and cached hints for {cls.__name__}: {list(hints.keys())}"
            )
            return hints
        except Exception as e:
            log.error(
                f"Failed getting hints for {cls.__name__}: {e}",
                exc_info=log.level <= logging.DEBUG,
            )
            self._type_hints_cache[cls] = {}  # Cache empty on error
            return {}

    def _get_field_aliases(self, cls: Type) -> Dict[str, str]:
        # ... (full implementation assumed present) ...
        # Add logging if this method becomes complex
        return {}  # Simplified

    def _resolve_generic_type_args(self, cls: Type) -> Dict[TypeVar, Type]:
        # ... (full implementation assumed present) ...
        # Add logging if this method becomes complex
        return {}  # Simplified

    def _format_error_message(
        self, error_context_path: str, expected_type: Type, value: Any
    ) -> str:
        # ... (full implementation assumed present) ...
        type_name = self._get_type_name(expected_type)
        value_type_name = type(value).__name__
        msg = f"Path '{error_context_path}': expected type {type_name}, got {repr(value)} ({value_type_name})."
        log.debug(f"Formatted validation error message: {msg}")
        return msg

    def _traverse_path(self, field_path: str) -> Type:
        log.debug(f"Traversing path '{field_path}' starting from {self.model_type!r}")
        parts = field_path.split(".")
        current_type = self.model_type
        current_path_context = []  # Keep track for error messages

        for i, part in enumerate(parts):
            current_path_context.append(part)
            error_context_path = ".".join(
                current_path_context
            )  # Path up to the current part
            log.debug(
                f"  Traversing part '{part}' (current type: {current_type!r}, path context: '{error_context_path}')"
            )

            # Ensure current_type is something we can introspect
            actual_type = current_type
            origin = get_origin(actual_type)
            args = get_args(actual_type)

            # Handle Optional[NextType] or Union[NextType, None] before trying getattr/getitem
            if origin is Union and len(args) >= 2 and _is_none_type(args[-1]):
                log.debug(
                    f"    Path part '{part}' is applied to Optional/Union, using inner type(s) for lookup: {args[:-1]!r}"
                )
                # If only one non-None type, use it directly. If multiple, keep as Union for now.
                non_none_args = tuple(t for t in args if not _is_none_type(t))
                if len(non_none_args) == 1:
                    actual_type = non_none_args[0]
                else:
                    # Cannot traverse into a plain Union like Union[ModelA, ModelB] directly yet
                    # Might need specific backend logic or disallow
                    log.error(
                        f"Path traversal into non-Optional Union {current_type!r} at '{error_context_path}' is not directly supported."
                    )
                    raise InvalidPathError(
                        f"Path traversal into non-Optional Union {current_type!r} at '{error_context_path}' is not directly supported."
                    )
                origin = get_origin(
                    actual_type
                )  # Update origin/args based on unwrapped type
                args = get_args(actual_type)
                log.debug(f"    Actual type for lookup is now: {actual_type!r}")

            # --- Resolve 'part' based on 'actual_type' ---
            resolved_part_type = None
            # Priority 1: get_type_hints (handles forward refs, inheritance better)
            if isclass(actual_type):  # get_type_hints needs a class
                try:
                    hints = self._get_cached_type_hints(actual_type)
                    if part in hints:
                        resolved_part_type = hints[part]
                        log.debug(
                            f"    Found '{part}' in hints for {actual_type.__name__}, next type: {resolved_part_type!r}"
                        )
                    # Fallback: Check __annotations__ directly if get_type_hints fails/misses? Less robust.
                    # elif hasattr(actual_type, "__annotations__") and part in actual_type.__annotations__:
                    #     resolved_part_type = actual_type.__annotations__[part]
                    #     log.debug(f"    Found '{part}' in __annotations__ for {actual_type.__name__}, next type: {resolved_part_type!r}")

                except Exception as e:
                    log.warning(
                        f"    Error getting hints for {actual_type.__name__} while resolving '{part}': {e}. Skipping hint check."
                    )

            # Placeholder for other resolution strategies if needed (e.g., Pydantic fields, dataclass fields)

            # --- FIX: Raise error if part was not resolved ---
            if resolved_part_type is None:
                log.error(
                    f"Cannot resolve part '{part}' in path '{field_path}' for type {actual_type!r} (at path context '{error_context_path}')"
                )
                # Try to provide a better error message by listing available fields
                available_fields = []
                if isclass(actual_type):
                    try:
                        hints = self._get_cached_type_hints(actual_type)
                        available_fields = list(hints.keys())
                    except Exception:
                        if hasattr(actual_type, "__annotations__"):
                            available_fields = list(
                                getattr(actual_type, "__annotations__", {}).keys()
                            )

                error_msg = f"Cannot resolve part '{part}' in path '{field_path}' for type {self._get_type_name(actual_type)}."
                if available_fields:
                    error_msg += f" Available fields: {available_fields}."

                raise InvalidPathError(error_msg)
            # --- END FIX ---

            # Update current_type for the next iteration
            current_type = resolved_part_type

            # Check if we need to continue traversing
            if i < len(parts) - 1:
                # Before the next part, if the current resolved type is Optional, unwrap it
                next_origin = get_origin(current_type)
                next_args = get_args(current_type)
                if (
                    next_origin is Union
                    and len(next_args) == 2
                    and _is_none_type(next_args[-1])
                ):
                    current_type = next_args[
                        0
                    ]  # Use the non-none type for next traversal step
                    log.debug(
                        f"    Unwrapped Optional for next segment, continuing with: {current_type!r}"
                    )

                # The unwrapped type must be traversable for the *next* part
                if not (isclass(current_type) or get_origin(current_type)):
                    log.error(
                        f"Path segment '{error_context_path}' resolved to non-traversable type {current_type!r} before end of path '{field_path}'"
                    )
                    raise InvalidPathError(
                        f"Path segment '{error_context_path}' does not lead to a nested model or structure (got {current_type!r})."
                    )

        log.debug(
            f"Successfully traversed path '{field_path}', final type: {current_type!r}"
        )
        return current_type

    def get_field_type(self, field_path: str) -> Type:
        log.debug(f"Getting field type for path: '{field_path}'")
        try:
            field_type = self._traverse_path(field_path)
            log.debug(f"Resolved type for '{field_path}': {field_type!r}")
            return field_type
        except InvalidPathError as e:
            log.warning(f"Invalid path encountered for '{field_path}': {e}")
            raise  # Re-raise specific error
        except Exception as e:
            log.error(
                f"Unexpected error getting type for path '{field_path}': {e}",
                exc_info=log.level <= logging.DEBUG,
            )
            raise InvalidPathError(
                f"Failed to get type for path '{field_path}': {e}"
            ) from e

    def validate_value(
        self, value: Any, expected_type: Type, path: str = "value"
    ) -> None:
        log.debug(
            f"Validating value at path '{path}': {repr(value)} against expected type {expected_type!r}"
        )
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_opt = origin is Union and type(None) in args

        if is_opt:
            if value is None:
                log.debug(
                    f"Value at '{path}' is None, which is valid for Optional type {expected_type!r}."
                )
                return
            # Get the non-None type(s) from Optional[T] or Union[T, None]
            non_none = tuple(t for t in args if not _is_none_type(t))
            expected_type = non_none[0] if len(non_none) == 1 else Union[non_none]  # type: ignore
            origin = get_origin(expected_type)  # Update origin for the inner type
            log.debug(
                f"Value at '{path}' is not None. Validating against inner type: {expected_type!r}"
            )

        base_expected = (
            _origin_to_class(origin) or expected_type
        )  # Get base class (e.g., list for List[int])
        if base_expected is Any:
            log.debug(
                f"Expected type at '{path}' is Any. Skipping validation for value {repr(value)}."
            )
            return  # Allow anything for Any

        # --- More robust check needed for full validator ---
        # Simplified check for focus: basic type check
        if not isinstance(base_expected, type):
            # Cannot easily check complex types like Union[int, str] without full logic
            log.warning(
                f"Cannot perform robust instance check for complex/non-class type {expected_type!r} at path '{path}'. Skipping strict check for value {repr(value)}. Full validator needed for unions, generics etc."
            )
            # Full validator would handle Unions, Generics (List[int]), etc. here
            return

        if not isinstance(value, base_expected):
            # Handle bool/int case specifically if needed
            if base_expected is int and isinstance(value, bool):
                err_msg = (
                    self._format_error_message(path, expected_type, value)
                    + " (bool is not int)"
                )
                log.warning(f"Validation failed: {err_msg}")
                raise ValueTypeError(err_msg)

            # Basic check failed
            err_msg = self._format_error_message(path, expected_type, value)
            log.warning(f"Validation failed: {err_msg}")
            raise ValueTypeError(err_msg)

        # Add further checks for generics if base_expected is list, dict etc.
        # e.g., if expected_type is List[int], check if value is a list and all items are ints.
        # This requires the full ModelValidator logic.
        log.debug(
            f"Basic type validation passed for path '{path}': value {repr(value)} matches base type {base_expected!r}."
        )

    def validate_value_for_path(self, field_path: str, value: Any) -> None:
        log.debug(f"Validating value for path '{field_path}': {repr(value)}")
        try:
            expected_type = self.get_field_type(field_path)
            self.validate_value(value, expected_type, field_path)
            log.debug(
                f"Validation successful for path '{field_path}' with value {repr(value)}"
            )
        except (InvalidPathError, ValueTypeError) as e:
            log.warning(f"Validation failed for path '{field_path}': {e}")
            raise  # Re-raise specific validation errors
        except Exception as e:
            log.error(
                f"Unexpected error during validation for path '{field_path}': {e}",
                exc_info=log.level <= logging.DEBUG,
            )
            # Wrap unexpected errors
            raise RuntimeError(
                f"Unexpected error during validation for path '{field_path}': {e}"
            ) from e

    def _is_single_type_numeric(self, t: Type) -> bool:
        # ... (full implementation assumed present) ...
        if not isinstance(t, type):
            return False
        is_num = issubclass(t, (int, float)) and t is not bool  # Exclude bool
        # log.debug(f"Checking if type {t!r} is numeric: {is_num}") # Can be noisy
        return is_num

    def is_field_numeric(self, field_path: str) -> bool:
        log.debug(f"Checking if field '{field_path}' is numeric.")
        try:
            t = self.get_field_type(field_path)
        except InvalidPathError:
            log.warning(
                f"Field path '{field_path}' not found while checking if numeric."
            )
            return False
        except Exception as e:
            log.error(
                f"Error getting type for '{field_path}' while checking if numeric: {e}",
                exc_info=log.level <= logging.DEBUG,
            )
            return False  # Treat error as non-numeric

        origin = get_origin(t)
        args = get_args(t)
        is_numeric = False
        if origin is Union:
            # Check if any non-None type in the Union is numeric
            is_numeric = any(
                self._is_single_type_numeric(a) for a in args if not _is_none_type(a)
            )
            log.debug(
                f"Field '{field_path}' is Union type {t!r}. Numeric check result: {is_numeric}"
            )
        else:
            is_numeric = self._is_single_type_numeric(t)
            log.debug(
                f"Field '{field_path}' is type {t!r}. Numeric check result: {is_numeric}"
            )
        return is_numeric

    def get_list_item_type(self, field_path: str) -> Tuple[bool, Type]:
        log.debug(f"Getting list item type for field '{field_path}'.")
        try:
            t = self.get_field_type(field_path)
            origin = get_origin(t)
            args = get_args(t)

            # Check for List[T] or list
            if origin in (list, List) or t is list:
                item_type = args[0] if args else Any
                log.debug(
                    f"Field '{field_path}' is List type. Item type: {item_type!r}"
                )
                return True, item_type

            # Check for Optional[List[T]] or Union[List[T], None]
            if origin is Union:
                for arg in args:
                    if _is_none_type(arg):
                        continue
                    arg_origin = get_origin(arg)
                    arg_args = get_args(arg)
                    if arg_origin in (list, List) or arg is list:
                        item_type = arg_args[0] if arg_args else Any
                        log.debug(
                            f"Field '{field_path}' is Optional/Union containing List. Item type: {item_type!r}"
                        )
                        return True, item_type  # Found the List type within the Union

            log.debug(
                f"Field '{field_path}' (type {t!r}) is not a List or Optional[List]."
            )
            return False, Any  # Not a list type we can determine item type for easily
        except InvalidPathError:
            log.warning(
                f"Field path '{field_path}' not found while getting list item type."
            )
            return False, Any
        except Exception as e:
            log.error(
                f"Error getting list item type for '{field_path}': {e}",
                exc_info=log.level <= logging.DEBUG,
            )
            return False, Any


###############################################################################
# Query Options
###############################################################################
class QueryOptions:
    """
    Options for querying repositories, including nested DSL expressions,
    sorting, pagination, and additional parameters.
    """

    # No complex logic, logging likely not needed here unless debugging its creation specifically.
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
        # log.debug(f"QueryOptions created: {self!r}") # Optional: log creation

    def __repr__(self):
        return (
            f"QueryOptions(expression={self.expression}, sort_by={self.sort_by}, "
            f"sort_desc={self.sort_desc}, limit={self.limit}, offset={self.offset}, "
            f"timeout={self.timeout}, random_order={self.random_order})"
        )


# --- Query Expression Classes ---


class Expression:
    """Base class for query expressions."""

    # Base class, no direct logging needed unless debugging inheritance issues.
    def __and__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with AND: {self!r} & {other!r}")
        return CombinedCondition("and", self, other)

    def __or__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with OR: {self!r} | {other!r}")
        return CombinedCondition("or", self, other)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict()")


class FilterCondition(Expression, Generic[T]):
    """
    Represents a filter condition (field OP value). Generic on field type T.
    T represents the expected type of the *field*, not necessarily the value.
    """

    field_path: str
    operator: str
    value: Any  # The value being compared against (might not be T)

    def __init__(self, field_path: str, operator: str, value: Any):
        # Logging here can be very noisy, often better to log when filter is *applied*
        # log.debug(f"Creating FilterCondition: path={field_path!r}, op={operator!r}, value={value!r}")
        self.field_path = field_path
        self.operator = operator
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        d = {self.field_path: {"operator": self.operator, "value": self.value}}
        # log.debug(f"FilterCondition {self!r} converted to dict: {d}") # Optional
        return d

    def __repr__(self) -> str:
        return (
            f"FilterCondition({self.field_path!r}, {self.operator!r}, {self.value!r})"
        )


class CombinedCondition(Expression):
    """Combines two expressions with AND or OR."""

    # Logging handled by __and__ / __or__ in Expression class
    logical_operator: str
    left: Expression
    right: Expression

    def __init__(self, logical_operator: str, left: Expression, right: Expression):
        if logical_operator not in ("and", "or"):
            log.error(
                f"Invalid logical_operator '{logical_operator}' for CombinedCondition."
            )
            raise ValueError("logical_operator must be 'and' or 'or'")
        self.logical_operator = logical_operator
        self.left = left
        self.right = right
        # log.debug(f"Creating CombinedCondition: {self!r}") # Optional

    def to_dict(self) -> Dict[str, Any]:
        d = {self.logical_operator: [self.left.to_dict(), self.right.to_dict()]}
        # log.debug(f"CombinedCondition {self!r} converted to dict: {d}") # Optional
        return d

    def __repr__(self) -> str:
        return f"CombinedCondition({self.logical_operator!r}, {self.left!r}, {self.right!r})"


# --- Field Representation ---
class Field(Generic[T]):
    """
    Represents a queryable field path with expected Python type T.
    Provides methods for building FilterCondition expressions.
    The type T helps static analysis but is not strictly enforced at runtime here.
    """

    _path: str

    def __init__(self, path: str):
        log.debug(f"Creating Field instance for path: '{path}'")
        object.__setattr__(self, "_path", path)

    @property
    def path(self) -> str:
        """The full dot-notation path to the field."""
        return self._path

    # Helper to create FilterCondition, reducing repetition
    def _op(self, op_name: str, other: Any) -> FilterCondition[T]:
        log.debug(
            f"Creating filter condition: Field('{self._path}') {op_name} {repr(other)}"
        )
        # Basic runtime checks for value types specific to operators
        # These checks are good but validation against the *field type* happens later
        # in QueryBuilder._validate_expression_runtime
        if op_name in ("like", "startswith", "endswith") and not isinstance(other, str):
            log.warning(
                f"Type mismatch for '{op_name}': Expected str value, got {type(other).__name__} for field '{self._path}'"
            )
            # Raise error early for obviously wrong value types for the operator itself
            raise TypeError(
                f"Operator '{op_name}' requires a string value, got {type(other).__name__}"
            )
        if op_name in ("in", "nin") and not isinstance(other, (list, set, tuple)):
            log.warning(
                f"Type mismatch for '{op_name}': Expected list/set/tuple value, got {type(other).__name__} for field '{self._path}'"
            )
            raise TypeError(
                f"Operator '{op_name}' requires a list, set, or tuple value, got {type(other).__name__}"
            )
        if op_name == "exists" and not isinstance(other, bool):
            log.warning(
                f"Type mismatch for '{op_name}': Expected bool value, got {type(other).__name__} for field '{self._path}'"
            )
            raise TypeError(
                f"Operator '{op_name}' requires a boolean value, got {type(other).__name__}"
            )

        # Convert sets/tuples for 'in'/'nin' to lists for consistent serialization
        value_to_use = (
            list(other)
            if isinstance(other, (set, tuple)) and op_name in ("in", "nin")
            else other
        )
        if value_to_use is not other:
            log.debug(
                f"Converted {type(other).__name__} to list for '{op_name}' operator on field '{self._path}'."
            )

        condition = FilterCondition(self._path, op_name, value_to_use)
        log.debug(f"Created condition: {condition!r}")
        return condition

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

    # --- Collection/String Operators ---
    def contains(self, item: Any) -> FilterCondition[T]:
        return self._op("contains", item)

    def like(self, pattern: str) -> FilterCondition[T]:
        return self._op("like", pattern)

    def startswith(self, prefix: str) -> FilterCondition[T]:
        return self._op("startswith", prefix)

    def endswith(self, suffix: str) -> FilterCondition[T]:
        return self._op("endswith", suffix)

    # --- General Operators ---
    def in_(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        return self._op("in", collection)

    def nin(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]:
        return self._op("nin", collection)

    def exists(self, exists_value: bool = True) -> FilterCondition[T]:
        return self._op("exists", exists_value)

    # --- Accessing Nested Fields (Runtime via __getattr__) ---
    def __getattr__(
        self, name: str
    ) -> "Field[Any]":  # Return Field[Any] for nested unknown type
        if name.startswith("_"):  # Avoid proxying special methods
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        new_path = f"{self._path}.{name}"
        log.debug(f"Accessing nested field: '{name}' -> new Field path '{new_path}'")
        return Field(new_path)

    def __repr__(self) -> str:
        # Try to get Generic alias name if possible (e.g., Field[str])
        try:
            # Use __orig_class__ for instances of Generic types
            base = getattr(self, "__orig_class__", type(self))
            type_repr = base.__name__
            generic_args = getattr(self, "__args__", None)
            if generic_args:
                arg_repr = ", ".join(
                    getattr(a, "__name__", repr(a)) for a in generic_args
                )
                type_repr = f"{type_repr}[{arg_repr}]"
            return f"{type_repr}(path={self._path!r})"
        except Exception:  # Fallback if introspection fails
            return f"Field(path={self._path!r})"

    # Prevent setting attributes on Field instances
    def __setattr__(self, name: str, value: Any):
        if name == "_path":  # Allow internal initialization
            object.__setattr__(self, name, value)
        else:
            log.error(
                f"Attempted to set attribute '{name}' on Field object '{self._path}'. Field objects are immutable."
            )
            raise AttributeError(
                f"Cannot set attribute '{name}' on Field object. Use comparison methods."
            )


# --- Fields Proxy Generation ---

# ** Improvement: Added caching **

_PROXY_CACHE: Dict[Type, Any] = {}  # Cache generated proxy objects


def _generate_fields_proxy(model_cls: Type[M], base_path: str = "") -> Any:
    """
    Introspects a model class and creates a proxy object where attributes
    are Field instances. Handles nesting via Field.__getattr__.
    Caches results for top-level models.
    """
    # ** Improvement: Cache Check **
    if not base_path and model_cls in _PROXY_CACHE:
        log.debug(f"Returning cached fields proxy for {model_cls.__name__}")
        return _PROXY_CACHE[model_cls]

    log.debug(
        f"Generating fields proxy for {model_cls.__name__} (base_path='{base_path}')"
    )
    proxy = SimpleNamespace()
    prefix = f"{base_path}." if base_path else ""

    # --- Introspection Logic ---
    try:
        # Use get_type_hints for better resolution, including forward refs if possible
        try:
            # Local and global namespaces might be needed for complex cases,
            # but often work without them for standard class definitions.
            annotations = get_type_hints(
                model_cls, include_extras=True
            )  # , globalns=globals(), localns=locals())
            log.debug(
                f"Successfully got type hints for {model_cls.__name__}: {list(annotations.keys())}"
            )
        except Exception as e:
            log.warning(
                f"get_type_hints failed for {model_cls.__name__}: {e}. Falling back to __annotations__."
            )
            annotations = getattr(model_cls, "__annotations__", {})
            if not annotations:
                log.error(
                    f"Could not find annotations for {model_cls.__name__}. Proxy generation will likely fail or be incomplete."
                )

        log.debug(
            f"Introspecting fields for {model_cls.__name__} (prefix='{prefix}'): {list(annotations.keys())}"
        )

        for name, type_hint in annotations.items():
            if name.startswith("_"):
                log.debug(
                    f"Skipping private attribute '{name}' in {model_cls.__name__}"
                )
                continue

            full_path = f"{prefix}{name}"
            log.debug(
                f"Processing field '{name}' (path: {full_path}), type hint: {type_hint!r}"
            )

            # --- CHANGE: Always create a Field object ---
            # Regardless of whether the type_hint points to a nested model or a simple type,
            # create a Field object. Nested access like qb.fields.address.street will
            # be handled seamlessly by Field.__getattr__, which creates the next Field object.
            log.debug(
                f"Creating Field for '{name}' with path '{full_path}' and type hint {type_hint!r}"
            )
            # We instantiate Field and let its __class_getitem__ handle the type hint if needed by static analysis.
            # For runtime, the path and the ModelValidator are key.
            setattr(proxy, name, Field[type_hint](full_path))
            log.debug(f"Added Field for '{name}' to proxy.")
            # --- END CHANGE ---

            # --- REMOVED OLD NESTED PROXY LOGIC ---
            # The previous logic that checked for nested models and recursively called
            # _generate_fields_proxy to create a SimpleNamespace is no longer needed.
            # --- END REMOVED OLD NESTED PROXY LOGIC ---

    except Exception as e:
        log.error(
            f"Failed during field proxy generation for {model_cls.__name__} at path '{prefix}': {e}",
            exc_info=log.level <= logging.DEBUG,
        )
        # It's crucial to raise here, as a partially generated proxy is problematic.
        raise TypeError(
            f"Could not generate query fields proxy for {model_cls.__name__}"
        ) from e

    # ** Improvement: Cache Store **
    if not base_path:
        log.debug(
            f"Caching generated fields proxy for top-level model {model_cls.__name__}"
        )
        _PROXY_CACHE[model_cls] = proxy

    log.debug(
        f"Finished generating fields proxy for {model_cls.__name__} (base_path='{base_path}')"
    )
    return proxy


# --- Query Builder ---
@dataclass_transform()
class QueryBuilder(Generic[M]):
    """
    Builds database queries in a statically analyzable way for model `M`.

    Use `qb.fields.fieldname` to access queryable fields. Autocompletion and
    static type checking are supported via PEP 681 (@dataclass_transform).

    Args:
        model_cls (Type[M]): The data model class (e.g., User, Product) being queried.
    """

    model_cls: Type[M]
    fields: Any  # Type checkers supporting dataclass_transform should infer this based on model_cls
    _validator: ModelValidator[M]
    _expression: Optional[Expression]
    _options: Dict[str, Any]
    _logger: logging.Logger  # Use instance logger based on shared config

    def __init__(self, model_cls: Type[M]):
        """
        Initializes the QueryBuilder for a specific model class `M`.
        """
        # Use the module-level logger configured earlier
        self._logger = log
        self._logger.info(f"Initializing QueryBuilder for model: {model_cls.__name__}")

        self.model_cls = model_cls
        self._expression = None
        self._options = {
            "limit": 100,
            "offset": 0,
            "sort_desc": False,
            "sort_by": None,
            "random_order": False,
            "timeout": None,
        }  # Default options
        self._logger.debug(f"Default query options set: {self._options}")

        try:
            # Generate the fields proxy FIRST, as it's often the most complex part
            self._logger.debug(f"Generating fields proxy for {model_cls.__name__}...")
            self.fields = _generate_fields_proxy(model_cls)
            self._logger.debug(
                f"Fields proxy generated successfully for {model_cls.__name__}."
            )

            # Initialize the validator AFTER proxy generation (though order doesn't strictly matter here)
            self._logger.debug(
                f"Initializing ModelValidator for {model_cls.__name__}..."
            )
            self._validator = ModelValidator(model_cls)
            self._logger.debug(
                f"ModelValidator initialized successfully for {model_cls.__name__}."
            )

            self._logger.info(
                f"QueryBuilder for model {model_cls.__name__} initialized successfully."
            )
        except Exception as e:
            self._logger.error(
                f"Failed to initialize QueryBuilder for {model_cls.__name__}: {e}",
                exc_info=self._logger.level <= logging.DEBUG,
            )
            # Ensure fields is initialized even on error to prevent downstream AttributeErrors
            self.fields = SimpleNamespace()  # Assign empty namespace on failure
            # Re-raise a more specific error
            raise ValueError(
                f"Could not initialize QueryBuilder for {model_cls.__name__}. "
                f"Failed generating fields proxy or initializing validator. Original error: {e}"
            ) from e

    def filter(self, expr: Expression) -> "QueryBuilder[M]":
        """
        Adds a filter expression to the query.
        If a filter already exists, it's combined with the new one using AND.

        Args:
            expr (Expression): A FilterCondition or CombinedCondition created using
                               qb.fields (e.g., `qb.fields.name == 'value'`).

        Returns:
            QueryBuilder[M]: The QueryBuilder instance for chaining.

        Raises:
            TypeError: If `expr` is not a valid Expression object.
            ValueError: If the expression fails runtime validation (e.g., type mismatch).
        """
        self._logger.debug(f"Adding filter expression: {expr!r}")
        if not isinstance(expr, Expression):
            self._logger.error(
                f"Invalid type passed to filter(): expected Expression, got {type(expr).__name__}"
            )
            raise TypeError(
                "filter() requires an Expression object (e.g., qb.fields.name == 'value'). "
                f"Got type: {type(expr).__name__}"
            )

        # ** Improvement: Perform runtime validation **
        try:
            self._logger.debug(
                f"Performing runtime validation for expression: {expr!r}"
            )
            self._validate_expression_runtime(expr)
            self._logger.debug(f"Runtime validation successful for: {expr!r}")
        except (ValueTypeError, InvalidPathError) as e:
            # Catch specific validation errors
            self._logger.warning(f"Filter validation failed for {expr!r}: {e}")
            # Re-raise as ValueError for a consistent API error type from filter()
            raise ValueError(f"Invalid filter expression: {e}") from e
        except Exception as e:
            # Catch unexpected errors during validation
            self._logger.error(
                f"Unexpected error during filter validation for {expr!r}: {e}",
                exc_info=self._logger.level <= logging.DEBUG,
            )
            raise ValueError(f"Unexpected error validating filter: {e}") from e

        if self._expression is None:
            self._logger.debug("No existing expression, setting new expression.")
            self._expression = expr
        else:
            self._logger.debug("Existing expression found, combining with AND.")
            self._expression = self._expression & expr  # Combine using AND
            self._logger.debug(f"Combined expression is now: {self._expression!r}")
        return self

    #################################
    #################################
    #################################

    # src/query.py -> class QueryBuilder

    def _validate_expression_runtime(self, expr: Expression) -> None:
        """
        Performs runtime checks on an expression's values and operators
        against the model schema using the ModelValidator.

        Args:
            expr (Expression): The expression to validate.

        Raises:
            InvalidPathError: If a field path in the expression is invalid.
            ValueTypeError: If a value type is incompatible with its field or operator.
        """
        if isinstance(expr, FilterCondition):
            field_path = expr.field_path
            operator = expr.operator
            value = expr.value
            self._logger.debug(
                f"Runtime validating FilterCondition: path='{field_path}', op='{operator}', value={repr(value)}"
            )

            try:
                # 1. Check path validity and get expected type (also caches hints)
                self._logger.debug(
                    f"  Getting expected type for path '{field_path}'..."
                )
                field_type = self._validator.get_field_type(field_path)
                self._logger.debug(
                    f"  Field '{field_path}' expected type (from validator): {field_type!r}"
                )

                # 2. Basic Value Type Check (most operators)
                if operator == "exists":
                    # Value type (bool) checked in Field._op, no further check needed against field_type
                    self._logger.debug(
                        f"  Operator is 'exists', skipping value-vs-fieldtype validation."
                    )
                    return

                if operator in ("in", "nin"):
                    # --- This block (for in/nin) should already be correct from previous fix ---
                    # ... (keep the existing logic for in/nin) ...
                    self._logger.debug(
                        f"  Operator is '{operator}', validating collection items."
                    )
                    if not isinstance(
                        value, (list, tuple)
                    ):  # Field._op converts sets/tuples to list
                        self._logger.error(
                            f"  Internal error: value for '{operator}' is not a list/tuple: {type(value)}"
                        )
                        raise ValueTypeError(
                            f"Internal error: value for '{operator}' should be a list or tuple",
                            path=field_path,
                        )

                    # Determine the expected type for ITEMS in the collection value...
                    expected_item_type = Any  # Default if we can't determine
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                    actual_field_type = field_type
                    if origin is Union and len(args) >= 2 and _is_none_type(args[-1]):
                        self._logger.debug(
                            f"  Field '{field_path}' is Optional/Union, unwrapping to check inner type: {args[:-1]!r}"
                        )
                        non_none_args = tuple(t for t in args if not _is_none_type(t))
                        actual_field_type = non_none_args[0] if len(non_none_args) == 1 else Union[non_none_args]  # type: ignore
                        origin = get_origin(actual_field_type)
                        args = get_args(actual_field_type)

                    # Check if the field is List[T], Set[T], or Tuple[T, ...]
                    if origin in (list, List, set, Set) and args:
                        expected_item_type = args[0]  # Get T from List[T] or Set[T]
                        self._logger.debug(
                            f"  Field is {origin.__name__ if origin else actual_field_type}[{expected_item_type!r}]. Expected item type: {expected_item_type!r}"
                        )
                    elif (
                        origin in (tuple, Tuple)
                        and args
                        and len(args) == 2
                        and args[1] is Ellipsis
                    ):
                        expected_item_type = args[0]  # Get T from Tuple[T, ...]
                        self._logger.debug(
                            f"  Field is Tuple[{expected_item_type!r}, ...]. Expected item type: {expected_item_type!r}"
                        )
                    elif actual_field_type in (list, set, tuple):
                        expected_item_type = Any
                        self._logger.debug(
                            f"  Field is raw {actual_field_type}, using Any for item type."
                        )
                    elif origin in (tuple, Tuple) and args:
                        expected_item_type = Union[args]  # type: ignore
                        self._logger.debug(
                            f"  Field is fixed Tuple {actual_field_type!r}, using {expected_item_type!r} for item type."
                        )
                    else:
                        expected_item_type = actual_field_type
                        self._logger.debug(
                            f"  Field '{field_path}' ({actual_field_type!r}) is not a standard collection. Items must match field type: {expected_item_type!r}"
                        )

                    self._logger.debug(
                        f"  Validating items in {repr(value)} against expected item type: {expected_item_type!r}"
                    )

                    if not value:
                        self._logger.debug(
                            f"  Collection for '{operator}' is empty, validation skipped."
                        )
                        return

                    for i, item in enumerate(value):
                        try:
                            self._validator.validate_value(
                                item,
                                expected_item_type,
                                f"{field_path} ({operator} item {i})",
                            )
                        except ValueTypeError as e:
                            # Add context about which operator failed
                            raise ValueTypeError(
                                f"Invalid item in '{operator}' list for field '{field_path}'. {e}",
                                path=field_path,
                            ) from e
                    self._logger.debug(
                        f"  All items in collection for '{operator}' validated successfully."
                    )
                    return  # Validation for 'in'/'nin' done

                if operator == "contains":
                    self._logger.debug(
                        f"  Operator is 'contains'. Checking field type {field_type!r}..."
                    )
                    # --- START FIX for contains ---
                    origin = get_origin(field_type)
                    args = get_args(field_type)
                    actual_field_type = field_type
                    item_type = Any

                    # Handle Optional[Collection[T]]
                    if origin is Union and len(args) >= 2 and _is_none_type(args[-1]):
                        self._logger.debug(
                            f"  Field '{field_path}' is Optional/Union, unwrapping to check inner type for 'contains': {args[:-1]!r}"
                        )
                        non_none_args = tuple(t for t in args if not _is_none_type(t))
                        actual_field_type = non_none_args[0] if len(non_none_args) == 1 else Union[non_none_args]  # type: ignore
                        origin = get_origin(actual_field_type)  # Update origin/args
                        args = get_args(actual_field_type)
                        self._logger.debug(
                            f"  Actual type for 'contains' check: {actual_field_type!r}"
                        )

                    # Check if the field is List[T] or Set[T] (or raw list/set)
                    is_collection_field = False
                    collection_name = "collection"  # Default
                    if origin in (list, List, set, Set) and args:
                        item_type = args[0]
                        is_collection_field = True
                        collection_name = origin.__name__ if origin else "collection"
                        self._logger.debug(
                            f"  Field '{field_path}' is {collection_name}[{item_type!r}]. Expected item type: {item_type!r}"
                        )
                    elif actual_field_type in (list, set):
                        item_type = Any
                        is_collection_field = True
                        collection_name = actual_field_type.__name__
                        self._logger.debug(
                            f"  Field '{field_path}' is raw {collection_name}. Expected item type: Any"
                        )

                    if is_collection_field:
                        self._logger.debug(
                            f"  Validating 'contains' value {repr(value)} against item type {item_type!r} for {collection_name} field '{field_path}'."
                        )
                        try:
                            # Value for 'contains' on a list/set field must match the collection's item type
                            self._validator.validate_value(
                                value, item_type, f"{field_path} (contains value)"
                            )
                            self._logger.debug(
                                f"  'contains' value validation successful for {collection_name} field."
                            )
                        except (ValueTypeError, TypeError) as item_e:
                            # Make error message more specific
                            raise ValueTypeError(
                                f"Operator 'contains' on field '{field_path}' requires value compatible with item type {self._validator._get_type_name(item_type)}, got {type(value).__name__}. Original error: {item_e}",
                                path=field_path,
                            ) from item_e
                        return  # List/Set contains validation done

                    # If not a list/set field, assume it's a string field for 'contains' (common case)
                    is_str_field = actual_field_type is str

                    if is_str_field:
                        self._logger.debug(
                            f"  Field '{field_path}' is string-like. Validating 'contains' value {repr(value)} is a string."
                        )
                        if not isinstance(value, str):
                            raise ValueTypeError(
                                f"Operator 'contains' on string field '{field_path}' requires a string value, got {type(value).__name__}",
                                path=field_path,
                            )
                        self._logger.debug(
                            f"  'contains' value validation successful for string field."
                        )
                        return  # String contains validation done
                    else:
                        # Contains might be backend-specific for other types (e.g., JSONB, maybe Tuple?)
                        # We will fall through to general validation, but warn.
                        self._logger.warning(
                            f"Runtime validation for 'contains' on non-list, non-set, non-string field '{field_path}' "
                            f"(type {actual_field_type!r}) is limited. Performing basic value-vs-fieldtype check."
                        )
                        # Fall through to general validation below
                    # --- END FIX for contains ---

                # General validation for eq, ne, gt, lt, ge, le, and potentially contains(non-list/set/str)
                self._logger.debug(
                    f"  Performing general value validation for operator '{operator}': value {repr(value)} vs field type {field_type!r}"
                )
                # Use the original field_type here for Optional checks etc.
                self._validator.validate_value(value, field_type, field_path)
                self._logger.debug(f"  General value validation successful.")

                # 3. Operator-Specific Value Constraints (if applicable) after basic type check
                if operator in ("gt", "lt", "ge", "le"):
                    # --- This block (for gt/lt/ge/le) should be okay ---
                    self._logger.debug(
                        f"  Performing orderability check for operator '{operator}' on field '{field_path}'."
                    )
                    is_numeric_field = self._validator.is_field_numeric(field_path)
                    # Need to check based on actual (unwrapped) type
                    _origin = get_origin(field_type)
                    _args = get_args(field_type)
                    _actual_type = field_type
                    if (
                        _origin is Union
                        and len(_args) >= 2
                        and _is_none_type(_args[-1])
                    ):
                        _non_none = tuple(t for t in _args if not _is_none_type(t))
                        if len(_non_none) == 1:
                            _actual_type = _non_none[0]

                    is_str_field = _actual_type is str

                    is_numeric_value = isinstance(
                        value, (int, float)
                    ) and not isinstance(value, bool)
                    is_str_value = isinstance(value, str)

                    # Allow comparison if field and value types are compatible (numeric-numeric or str-str)
                    if is_numeric_field and not is_numeric_value:
                        # Allow comparing int value to float field and vice-versa? Yes, typically.
                        if not (
                            isinstance(value, int) and _actual_type is float
                        ) and not (isinstance(value, float) and _actual_type is int):
                            raise ValueTypeError(
                                f"Operator '{operator}' on numeric field '{field_path}' requires a numeric value (int/float), got {type(value).__name__}",
                                path=field_path,
                            )
                    if is_str_field and not is_str_value:
                        raise ValueTypeError(
                            f"Operator '{operator}' on string field '{field_path}' requires a string value, got {type(value).__name__}",
                            path=field_path,
                        )
                    # Allow comparison if value is directly instance of field type (e.g. datetime)
                    if (
                        not is_numeric_field
                        and not is_str_field
                        and not isinstance(value, _actual_type)
                    ):
                        # If not clearly numeric or string comparison, and value doesn't match field type, issue warning/error
                        # Warning is likely safer as backend might support it (e.g., dates).
                        self._logger.warning(
                            f"Runtime check: Operator '{operator}' used on non-numeric, non-string field "
                            f"'{field_path}' (type {field_type!r}) with value {repr(value)} of type {type(value).__name__}. "
                            f"Comparison support depends on backend and types involved."
                        )
                        # Raise stricter error if needed:
                        # raise ValueTypeError(
                        #    f"Operator '{operator}' requires a value type compatible with the field type '{field_type!r}' for comparison, got {type(value).__name__}",
                        #    path=field_path,
                        # )

                    self._logger.debug(
                        f"  Orderability check passed (or warned) for '{operator}'."
                    )

                elif operator in ("like", "startswith", "endswith"):
                    # --- This block (for like/startswith/endswith) should be okay ---
                    self._logger.debug(
                        f"  Performing string compatibility check for operator '{operator}' on field '{field_path}'."
                    )
                    _origin = get_origin(field_type)
                    _args = get_args(field_type)
                    _actual_type = field_type
                    if (
                        _origin is Union
                        and len(_args) >= 2
                        and _is_none_type(_args[-1])
                    ):
                        _non_none = tuple(t for t in _args if not _is_none_type(t))
                        if len(_non_none) == 1:
                            _actual_type = _non_none[0]

                    is_str_field = _actual_type is str

                    if not is_str_field:
                        raise ValueTypeError(
                            f"Operator '{operator}' is typically used with string fields, but field '{field_path}' has type {field_type!r}",
                            path=field_path,
                        )
                    self._logger.debug(
                        f"  String compatibility check passed for '{operator}'."
                    )

            except (InvalidPathError, ValueTypeError) as e:
                # Log already happened in validator methods or here
                self._logger.warning(f"Validation failed for condition ({expr!r}): {e}")
                raise  # Re-raise the specific validation error
            except Exception as e:
                # Capture unexpected errors during validation more gracefully
                self._logger.error(
                    f"Unexpected error during runtime validation of condition ({expr!r}): {e}",
                    exc_info=self._logger.level <= logging.DEBUG,
                )
                # Wrap unexpected errors into a RuntimeError or specific internal error
                raise RuntimeError(
                    f"Unexpected error validating condition {expr!r}: {e}"
                ) from e

        elif isinstance(expr, CombinedCondition):
            # --- This block (for CombinedCondition) should be okay ---
            self._logger.debug(
                f"Runtime validating CombinedCondition (op='{expr.logical_operator}'): Left side..."
            )
            self._validate_expression_runtime(expr.left)
            self._logger.debug(
                f"Runtime validating CombinedCondition (op='{expr.logical_operator}'): Right side..."
            )
            self._validate_expression_runtime(expr.right)
            self._logger.debug(
                f"Runtime validation successful for CombinedCondition: {expr!r}"
            )
        else:
            # Should not happen if filter() checks input type
            self._logger.error(
                f"Encountered unknown expression type during validation: {type(expr)}"
            )
            raise TypeError(
                f"Cannot validate unknown expression type: {type(expr).__name__}"
            )

    #################################
    #################################
    #################################
    #################################

    def sort_by(self, field: Field[Any], descending: bool = False) -> "QueryBuilder[M]":
        """
        Sets the sort field and order.

        Args:
            field (Field): The field to sort by (e.g., `qb.fields.name`).
            descending (bool): Sort in descending order if True.

        Returns:
            QueryBuilder[M]: The QueryBuilder instance for chaining.

        Raises:
            TypeError: If `field` is not a Field object.
            AttributeError: If the field path is invalid for the model.
        """
        self._logger.debug(
            f"Setting sort order: field={field!r}, descending={descending}"
        )
        if not isinstance(field, Field):
            self._logger.error(
                f"Invalid type passed to sort_by(): expected Field, got {type(field).__name__}"
            )
            raise TypeError("sort_by requires a Field object (e.g., qb.fields.name)")

        # Runtime check path validity using the validator
        try:
            # This implicitly uses _traverse_path and raises InvalidPathError if needed
            self._logger.debug(f"Validating sort field path: '{field.path}'")
            self._validator.get_field_type(field.path)
            self._logger.debug(
                f"Sort field path '{field.path}' validated successfully."
            )
        except InvalidPathError as e:
            self._logger.warning(f"Invalid sort field path '{field.path}': {e}")
            # Raise AttributeError for consistency with accessing invalid fields via proxy
            raise AttributeError(
                f"Invalid sort field path: {field.path}. Error: {e}"
            ) from e
        except Exception as e:  # Catch other potential errors from get_field_type
            self._logger.error(
                f"Unexpected error validating sort field '{field.path}': {e}",
                exc_info=self._logger.level <= logging.DEBUG,
            )
            raise RuntimeError(
                f"Unexpected error validating sort field {field.path}: {e}"
            ) from e

        self._options["sort_by"] = field.path
        self._options["sort_desc"] = descending
        # Sorting overrides random order
        if self._options.get("random_order"):
            self._logger.debug("Clearing random_order because sort_by was called.")
            self._options["random_order"] = False
        self._logger.info(
            f"Sort order set to: field='{field.path}', descending={descending}"
        )
        return self

    def random_order(self) -> "QueryBuilder[M]":
        """Sets random ordering (clears any previous sort_by)."""
        self._logger.debug("Setting random order.")
        self._options["random_order"] = True
        if self._options.get("sort_by") is not None:
            self._logger.debug(
                "Clearing sort_by/sort_desc because random_order was called."
            )
            self._options["sort_by"] = None
            self._options["sort_desc"] = False
        self._logger.info("Query order set to random.")
        return self

    def limit(self, num: int) -> "QueryBuilder[M]":
        """Sets the query limit."""
        self._logger.debug(f"Setting limit to: {num}")
        if not isinstance(num, int) or num < 0:
            self._logger.error(
                f"Invalid limit value: {num}. Must be non-negative integer."
            )
            raise ValueError("Limit must be a non-negative integer.")
        self._options["limit"] = num
        self._logger.info(f"Query limit set to: {num}")
        return self

    def offset(self, num: int) -> "QueryBuilder[M]":
        """Sets the query offset."""
        self._logger.debug(f"Setting offset to: {num}")
        if not isinstance(num, int) or num < 0:
            self._logger.error(
                f"Invalid offset value: {num}. Must be non-negative integer."
            )
            raise ValueError("Offset must be a non-negative integer.")
        self._options["offset"] = num
        self._logger.info(f"Query offset set to: {num}")
        return self

    def timeout(self, seconds: Optional[float]) -> "QueryBuilder[M]":
        """Sets the query timeout in seconds (implementation specific)."""
        self._logger.debug(f"Setting timeout to: {seconds}")
        if seconds is not None and (
            not isinstance(seconds, (int, float)) or seconds < 0
        ):
            self._logger.error(
                f"Invalid timeout value: {seconds}. Must be non-negative number or None."
            )
            raise ValueError("Timeout must be a non-negative number or None.")
        self._options["timeout"] = seconds
        self._logger.info(f"Query timeout set to: {seconds}")
        return self

    def build(self) -> QueryOptions:
        """
        Builds the final QueryOptions object representing the query.

        Returns:
            QueryOptions: An object containing the filter expression dictionary
                          and other query options (sort, limit, offset, etc.).
        """
        self._logger.debug("Building QueryOptions...")
        expression_dict = {}
        if self._expression:
            self._logger.debug(
                f"Converting final expression to dict: {self._expression!r}"
            )
            expression_dict = self._expression.to_dict()
            self._logger.debug(f"Expression dict: {expression_dict}")
        else:
            self._logger.debug("No filter expression was set.")

        options = QueryOptions(
            expression=expression_dict,
            sort_by=self._options.get("sort_by"),
            sort_desc=self._options.get("sort_desc", False),
            limit=self._options.get("limit", 100),
            offset=self._options.get("offset", 0),
            random_order=self._options.get("random_order", False),
            timeout=self._options.get("timeout"),
        )
        self._logger.info(
            f"Built query options for {self.model_cls.__name__}: {options!r}"
        )
        return options
