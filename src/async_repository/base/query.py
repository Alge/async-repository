# src/query.py

import logging
import operator
import traceback
from dataclasses import is_dataclass
from inspect import isclass
from types import SimpleNamespace  # For underlying fields proxy storage
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
    get_args,
    get_origin,
    get_type_hints,
)

# Use @dataclass_transform for better IDE support with model-specific fields
from typing import dataclass_transform

# --- Setup Logging ---
log = logging.getLogger(__name__)

# --- Generic Type Variables ---
T = TypeVar("T")  # Field data type (e.g., str, int, bool)
M = TypeVar("M")  # Model class type (e.g., User, Address)


# --- Validation Exceptions ---
class ValidationError(TypeError):
    """Base class for validation errors."""

    def __init__(self, message: str, *, path: Optional[str] = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.path = path
        # Store any extra kwargs if needed
        self.details = kwargs

    def __str__(self) -> str:
        """Optionally include path context in the error message."""
        if self.path:
            path_prefix = f"Path '{self.path}':"
            # Avoid duplicating path if message already includes it
            if self.message.startswith(path_prefix):
                return self.message
            else:
                return f"Path '{self.path}': {self.message}"
        return self.message


class InvalidPathError(ValidationError, AttributeError):
    """Error raised when a field path does not exist or is invalid."""


class ValueTypeError(ValidationError, TypeError):
    """Error raised when a value's type is incompatible."""


# --- Helper Functions ---
def _is_none_type(t: Optional[Type]) -> bool:
    """Checks if a type is NoneType."""
    return t is type(None)


def _is_typevar(t: Any) -> bool:
    """Checks if an object is a TypeVar."""
    return isinstance(t, TypeVar)


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
        Mapping: dict,  # Map Mapping to dict for isinstance checks
    }
    mapped = map_.get(origin, origin)
    # Return only if it's actually a class, otherwise return the original origin
    return mapped if isclass(mapped) else origin


# --- Model Validator ---
class ModelValidator(Generic[M]):
    """Performs validation of field paths and values against model M."""

    model_type: Type[M]
    _type_hints_cache: Dict[Type, Dict[str, Type]]
    _field_aliases_cache: Dict[Type, Dict[str, str]]  # Placeholder
    _generic_type_cache: Dict[str, Dict[TypeVar, Type]]  # Placeholder

    def __init__(self, model_type: Type[M]):
        log.debug(f"Initializing ModelValidator for type: {model_type!r}")
        if (
            not isclass(model_type)
            and not get_origin(model_type)
            and not hasattr(model_type, "__pydantic_generic_metadata__")
        ):  # Basic Pydantic check
            log.error(f"Invalid model_type for ModelValidator: {type(model_type)}")
            raise TypeError(
                "model_type must be a class or generic alias, received "
                f"{type(model_type)}."
            )
        self.model_type = model_type
        self._type_hints_cache = {}
        self._field_aliases_cache = {}
        self._generic_type_cache = {}
        log.debug(f"ModelValidator initialized successfully for {model_type!r}")

    def _get_type_name(self, type_obj: Any) -> str:
        """Helper to get a user-friendly name for a type."""
        if type_obj is Any:
            return "Any"
        if _is_none_type(type_obj):
            return "None"

        origin = get_origin(type_obj)
        args = get_args(type_obj)

        if origin is Union:
            non_none_args = [a for a in args if not _is_none_type(a)]
            if len(non_none_args) == 1 and len(args) == 2:
                # Format Union[X, None] as Optional[X]
                return f"Optional[{self._get_type_name(non_none_args[0])}]"
            else:  # General Union
                return f"Union[{', '.join(self._get_type_name(a) for a in args)}]"
        elif origin is not None:  # Generics like List[str]
            base_name = getattr(origin, "__name__", str(origin))
            arg_names = ", ".join(self._get_type_name(a) for a in args)
            return f"{base_name}[{arg_names}]"
        elif hasattr(type_obj, "__name__"):  # Simple types like int, str
            return type_obj.__name__
        else:  # Fallback
            return repr(type_obj)

    def _get_cached_type_hints(self, cls: Type) -> Dict[str, Type]:
        """Gets type hints for a class, using a cache."""
        if not isclass(cls):
            log.debug(f"Cannot get hints for non-class type: {cls!r}")
            return {}

        if cls in self._type_hints_cache:
            log.debug(f"Cache hit for type hints of {cls.__name__}")
            return self._type_hints_cache[cls]

        log.debug(f"Cache miss for type hints of {cls.__name__}. Fetching...")
        try:
            hints = get_type_hints(cls, include_extras=True)
            self._type_hints_cache[cls] = hints
            log.debug(
                f"Successfully fetched/cached hints for {cls.__name__}: "
                f"{list(hints.keys())}"
            )
            return hints
        except Exception as e:
            log.error(f"Failed getting hints for {cls.__name__}: {e}",
                      exc_info=log.level <= logging.DEBUG)
            # Fallback to __annotations__ if get_type_hints fails
            try:
                hints = getattr(cls, "__annotations__", {})
                self._type_hints_cache[cls] = hints
                log.warning(
                    f"Fell back to __annotations__ for {cls.__name__}: "
                    f"{list(hints.keys())}"
                )
                return hints
            except Exception as e_inner:
                log.error(
                    f"Failed getting __annotations__ fallback for "
                    f"{cls.__name__}: {e_inner}",
                    exc_info=log.level <= logging.DEBUG
                )
                self._type_hints_cache[cls] = {}  # Cache empty on error
                return {}

    def _get_field_aliases(self, cls: Type) -> Dict[str, str]:
        # Placeholder: Real implementation needed
        return {}

    def _resolve_generic_type_args(self, cls: Type) -> Dict[TypeVar, Type]:
        # Placeholder: Real implementation needed
        return {}

    def _format_error_message(
        self, error_context_path: str, expected_type: Type, value: Any
    ) -> str:
        """Creates a standardized validation error message."""
        expected_type_name = self._get_type_name(expected_type)
        value_type_name = type(value).__name__
        msg = (
            f"Path '{error_context_path}': expected type {expected_type_name}, "
            f"got {repr(value)} ({value_type_name})."
        )
        log.debug(f"Formatted validation error message: {msg}")
        return msg

    def _traverse_path(self, field_path: str) -> Type:
        """Recursively traverses a dot-notation path to find the target type."""
        log.debug(f"Traversing path '{field_path}' starting from {self.model_type!r}")
        parts = field_path.split(".")
        current_type = self.model_type
        current_path_context = []  # Track path for error messages

        for i, part in enumerate(parts):
            current_path_context.append(part)
            error_context_path = ".".join(current_path_context)
            log.debug(
                f"  Traversing part '{part}' (current type: "
                f"{current_type!r}, path context: '{error_context_path}')"
            )

            actual_type = current_type
            origin = get_origin(actual_type)
            args = get_args(actual_type)
            is_optional_like = origin is Union and any(_is_none_type(a) for a in args)

            # Handle Optional/Union before resolving the part on the inner type
            if is_optional_like:
                non_none_args = tuple(t for t in args if not _is_none_type(t))
                log.debug(
                    f"    Path part '{part}' applied to Optional/Union, "
                    f"using inner type(s) for lookup: {non_none_args!r}"
                )
                if len(non_none_args) == 1:
                    actual_type = non_none_args[0]
                elif len(non_none_args) > 1:
                    msg = (
                        f"Path traversal into non-Optional Union "
                        f"{current_type!r} at '{error_context_path}' is not "
                        f"directly supported."
                    )
                    log.error(msg)
                    raise InvalidPathError(msg, path=error_context_path)
                else: # Should not happen
                     msg = f"Invalid Optional/Union type: {current_type!r}"
                     log.error(f"Internal error: {msg}")
                     raise TypeError(msg)
                # Update origin/args based on unwrapped type
                origin = get_origin(actual_type)
                args = get_args(actual_type)
                log.debug(f"    Actual type for lookup is now: {actual_type!r}")

            # Resolve 'part' based on 'actual_type'
            resolved_part_type = None
            if isclass(actual_type):
                try:
                    hints = self._get_cached_type_hints(actual_type)
                    if part in hints:
                        resolved_part_type = hints[part]
                        log.debug(
                            f"    Found '{part}' in hints for "
                            f"{actual_type.__name__}, next type: "
                            f"{resolved_part_type!r}"
                        )
                except Exception as e:
                    log.warning(
                        f"    Error getting hints for {actual_type.__name__} "
                        f"while resolving '{part}': {e}. Skipping."
                    )

            # Raise error if part was not resolved
            if resolved_part_type is None:
                log.error(
                    f"Cannot resolve part '{part}' in path '{field_path}' "
                    f"for type {actual_type!r} (at path context "
                    f"'{error_context_path}')"
                )
                available_fields = []
                if isclass(actual_type):
                    try:
                        hints = self._get_cached_type_hints(actual_type)
                        available_fields = [f for f in hints if not f.startswith('_')]
                    except Exception: # nosec B110 - ignore errors during error reporting
                        pass # Ignore exceptions during error reporting enhancement
                error_msg = (
                    f"Cannot resolve part '{part}' in path '{field_path}' "
                    f"for type {self._get_type_name(actual_type)}."
                )
                if available_fields:
                    error_msg += f" Available fields: {available_fields}."
                raise InvalidPathError(error_msg, path=error_context_path)

            current_type = resolved_part_type

            # Check if we need to continue traversing
            if i < len(parts) - 1:
                next_origin = get_origin(current_type)
                next_args = get_args(current_type)
                unwrapped_type_for_next = current_type
                is_next_optional = (next_origin is Union and
                                    any(_is_none_type(a) for a in next_args))

                if is_next_optional:
                    non_none_next = tuple(t for t in next_args if not _is_none_type(t))
                    if len(non_none_next) == 1:
                        unwrapped_type_for_next = non_none_next[0]
                        log.debug(
                            f"    Unwrapped Optional for next segment, "
                            f"continuing with: {unwrapped_type_for_next!r}"
                        )
                    # else: cannot traverse into complex Union like Union[ModelA, ModelB]

                # The unwrapped type must be class-like to traverse further
                if not isclass(unwrapped_type_for_next):
                    msg = (
                        f"Path segment '{error_context_path}' resolved to "
                        f"non-traversable type {current_type!r} "
                        f"(unwrapped: {unwrapped_type_for_next!r}) before "
                        f"end of path '{field_path}'"
                    )
                    log.error(msg)
                    raise InvalidPathError(
                        f"Path segment '{error_context_path}' does not lead to "
                        f"a nested model or structure (got "
                        f"{self._get_type_name(current_type)}).",
                        path=error_context_path
                    )

        log.debug(
            f"Successfully traversed path '{field_path}', "
            f"final type: {current_type!r}"
        )
        return current_type

    def get_field_type(self, field_path: str) -> Type:
        """Gets the expected Python type of a field specified by its path."""
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
                exc_info=log.level <= logging.DEBUG
            )
            raise InvalidPathError(
                f"Failed to get type for path '{field_path}': {e}",
                path=field_path
            ) from e

    def validate_value(
        self, value: Any, expected_type: Type, path: str = "value"
    ) -> None:
        """Validates a value against an expected type, handling Optionals/Unions."""
        log.debug(
            f"Validating value at path '{path}': {repr(value)} "
            f"against expected type {expected_type!r}"
        )
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_opt = origin is Union and any(_is_none_type(a) for a in args)

        # Handle Optional[T] or Union[T, None, ...]
        if is_opt:
            if value is None:
                log.debug(
                    f"Value at '{path}' is None, which is valid for "
                    f"Optional type {expected_type!r}."
                )
                return
            non_none_args = tuple(t for t in args if not _is_none_type(t))
            if len(non_none_args) == 1:
                inner_expected_type = non_none_args[0]
            elif len(non_none_args) > 1:
                 inner_expected_type = Union[non_none_args] # type: ignore
                 log.warning(
                     f"Validating against Union {inner_expected_type!r} at path '{path}'"
                 )
            else:
                 raise TypeError(f"Invalid Optional/Union type: {expected_type!r}")
            log.debug(
                f"Value at '{path}' is not None. Validating against "
                f"inner type: {inner_expected_type!r}"
            )
            # Recursively validate against the non-None type(s)
            return self.validate_value(value, inner_expected_type, path=path)

        # --- Non-Optional Validation ---
        if expected_type is Any:
            log.debug(f"Expected type at '{path}' is Any. Skipping validation.")
            return

        # Handle simple Union types (e.g., Union[int, str])
        if origin is Union:
            matched = False
            for arg_type in args:
                 # Check against origin if it's a generic (like list), else the type itself
                 base_arg_type = _origin_to_class(get_origin(arg_type)) or arg_type
                 if isinstance(base_arg_type, type) and isinstance(value, base_arg_type):
                      # Basic match found, complex generic validation might be needed
                      matched = True
                      break
            if matched:
                 log.debug(
                     f"Value {repr(value)} matches one type in Union "
                     f"{expected_type!r} at path '{path}'."
                 )
                 return
            else:
                 err_msg = self._format_error_message(path, expected_type, value)
                 log.warning(f"Validation failed (Union): {err_msg}")
                 raise ValueTypeError(err_msg, path=path)

        # Basic type check using origin or the type itself
        base_expected = _origin_to_class(origin) or expected_type
        if not isinstance(base_expected, type):
            log.warning(
                f"Cannot perform robust instance check for non-class "
                f"type {expected_type!r} at path '{path}'. Skipping."
            )
            return

        # Strict instance check
        if not isinstance(value, base_expected):
            err_msg = self._format_error_message(path, expected_type, value)
            # Add specific hints for common issues
            if base_expected is int and isinstance(value, bool):
                err_msg += " (bool is not int)"
            elif base_expected is float and isinstance(value, bool):
                 err_msg += " (bool is not float)"
            log.warning(f"Validation failed (Type): {err_msg}")
            raise ValueTypeError(err_msg, path=path)

        # TODO: Add recursive validation for container types if needed
        log.debug(f"Basic type validation passed for path '{path}'.")

    def validate_value_for_path(self, field_path: str, value: Any) -> None:
        """Validates a value against the expected type of a field path."""
        log.debug(f"Validating value for path '{field_path}': {repr(value)}")
        try:
            expected_type = self.get_field_type(field_path)
            self.validate_value(value, expected_type, field_path)
            log.debug(f"Validation successful for path '{field_path}'.")
        except (InvalidPathError, ValueTypeError) as e:
             log.warning(f"Validation failed for path '{field_path}': {e}")
             raise # Re-raise specific validation errors
        except Exception as e:
             log.error(
                 f"Unexpected error during validation for path "
                 f"'{field_path}': {e}",
                 exc_info=log.level <= logging.DEBUG
             )
             raise RuntimeError(
                 f"Unexpected error during validation for path "
                 f"'{field_path}': {e}"
             ) from e

    def _is_single_type_numeric(self, t: Type) -> bool:
        """Checks if a single type is numeric (int/float, excluding bool)."""
        if not isinstance(t, type):
            return False
        return issubclass(t, (int, float)) and t is not bool

    def is_field_numeric(self, field_path: str) -> bool:
        """Checks if the field at the path is numeric (handles Optional)."""
        log.debug(f"Checking if field '{field_path}' is numeric.")
        try:
            t = self.get_field_type(field_path)
        except InvalidPathError:
            log.warning(f"Field path '{field_path}' not found checking numeric.")
            return False # Path not found
        except Exception as e:
            log.error(f"Error getting type for '{field_path}'",
                      exc_info=log.level <= logging.DEBUG)
            return False # Treat error as non-numeric

        origin = get_origin(t)
        args = get_args(t)
        is_numeric = False
        if origin is Union:
            is_numeric = any(self._is_single_type_numeric(a)
                             for a in args if not _is_none_type(a))
        else:
            is_numeric = self._is_single_type_numeric(t)

        log.debug(f"Field '{field_path}' type {t!r}. Numeric: {is_numeric}")
        return is_numeric

    def get_list_item_type(self, field_path: str) -> Tuple[bool, Type]:
        """Gets List item type if field is List/Optional[List], else (False, Any)."""
        log.debug(f"Getting list item type for field '{field_path}'.")
        try:
            t = self.get_field_type(field_path)
            origin = get_origin(t)
            args = get_args(t)

            # Check for List[T] or list
            if origin in (list, List) or t is list:
                item_type = args[0] if args else Any
                log.debug(f"Field '{field_path}' is List type. Item type: {item_type!r}")
                return True, item_type

            # Check for Optional[List[T]] or Union[List[T], None, ...]
            if origin is Union:
                for arg in args:
                    if _is_none_type(arg):
                        continue
                    arg_origin = get_origin(arg)
                    arg_args = get_args(arg)
                    if arg_origin in (list, List) or arg is list:
                        item_type = arg_args[0] if arg_args else Any
                        log.debug(
                            f"Field '{field_path}' is Optional/Union containing "
                            f"List. Item type: {item_type!r}"
                        )
                        return True, item_type

            log.debug(f"Field '{field_path}' (type {t!r}) is not a List or Optional[List].")
            return False, Any
        except InvalidPathError:
            log.warning(f"Field path '{field_path}' not found getting list item type.")
            return False, Any
        except Exception as e:
            log.error(
                f"Error getting list item type for '{field_path}'",
                exc_info=log.level <= logging.DEBUG
            )
            return False, Any


# --- Query Options & Expressions ---
class QueryOptions:
    """Options for querying repositories."""
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


class Expression:
    """Base class for query expressions."""
    def __and__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with AND: {self!r} & {other!r}")
        return CombinedCondition("and", self, other)

    def __or__(self, other: "Expression") -> "CombinedCondition":
        log.debug(f"Combining expressions with OR: {self!r} | {other!r}")
        return CombinedCondition("or", self, other)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement to_dict()")


class FilterCondition(Expression, Generic[T]):
    """Represents a filter condition (field OP value)."""
    field_path: str
    operator: str
    value: Any

    def __init__(self, field_path: str, operator: str, value: Any):
        self.field_path = field_path
        self.operator = operator
        self.value = value

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
    """Represents a queryable field path."""
    _path: str

    def __init__(self, path: str):
        log.debug(f"Creating Field instance for path: '{path}'")
        object.__setattr__(self, "_path", path)

    @property
    def path(self) -> str:
        return self._path

    def _op(self, op_name: str, other: Any) -> FilterCondition[T]:
        """Helper to create FilterCondition with basic operator value type checks."""
        log.debug(f"Creating filter: Field('{self._path}') {op_name} {repr(other)}")
        if op_name in ("like", "startswith", "endswith") and not isinstance(other, str):
            raise TypeError(f"Operator '{op_name}' requires a string value")
        if op_name in ("in", "nin") and not isinstance(other, (list, set, tuple)):
            raise TypeError(f"Operator '{op_name}' requires a list/set/tuple")
        if op_name == "exists" and not isinstance(other, bool):
            raise TypeError(f"Operator '{op_name}' requires a boolean value")

        # Convert sets/tuples for 'in'/'nin' to lists for consistent serialization
        value_to_use = list(other) if isinstance(other, (set, tuple)) and op_name in ("in", "nin") else other
        if value_to_use is not other:
            log.debug(f"Converted {type(other).__name__} to list for '{op_name}'")

        return FilterCondition(self._path, op_name, value_to_use)

    # Comparison operators
    def __eq__(self, other: Any) -> FilterCondition[T]: return self._op("eq", other)
    def __ne__(self, other: Any) -> FilterCondition[T]: return self._op("ne", other)
    def __gt__(self, other: Any) -> FilterCondition[T]: return self._op("gt", other)
    def __lt__(self, other: Any) -> FilterCondition[T]: return self._op("lt", other)
    def __ge__(self, other: Any) -> FilterCondition[T]: return self._op("ge", other)
    def __le__(self, other: Any) -> FilterCondition[T]: return self._op("le", other)

    # Other operators
    def contains(self, item: Any) -> FilterCondition[T]: return self._op("contains", item)
    def like(self, pattern: str) -> FilterCondition[T]: return self._op("like", pattern)
    def startswith(self, prefix: str) -> FilterCondition[T]: return self._op("startswith", prefix)
    def endswith(self, suffix: str) -> FilterCondition[T]: return self._op("endswith", suffix)
    def in_(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]: return self._op("in", collection)
    def nin(self, collection: Union[List, Set, Tuple]) -> FilterCondition[T]: return self._op("nin", collection)
    def exists(self, exists_value: bool = True) -> FilterCondition[T]: return self._op("exists", exists_value)

    def __getattr__(self, name: str) -> "Field[Any]":
        """Dynamically create nested Field objects."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        new_path = f"{self._path}.{name}"
        log.debug(f"Accessing nested field: '{name}' -> new Field path '{new_path}'")
        return Field(new_path) # Type validation happens later

    def __repr__(self) -> str:
        # Simple repr for clarity
        return f"Field(path={self._path!r})"

    def __setattr__(self, name: str, value: Any):
        """Prevent modification after initialization."""
        # Allow internal setting of '_path' via object.__setattr__ in __init__
        if name == "_path" and hasattr(self, name): # Check if already initialized
             raise AttributeError("Cannot modify Field attributes after initialization.")
        elif name != "_path": # Prevent setting any other attribute
            raise AttributeError(f"Cannot set attribute '{name}' on Field object.")
        else:
             # Allow the initial setting in __init__
             object.__setattr__(self, name, value)


# --- Fields Proxy Generation ---
_PROXY_CACHE: Dict[Type, SimpleNamespace] = {}

def _generate_fields_proxy(model_cls: Type[M], base_path: str = "") -> SimpleNamespace:
    """Introspects model and creates a SimpleNamespace with Field attributes."""
    if not base_path and model_cls in _PROXY_CACHE:
        log.debug(f"Returning cached fields proxy object for {model_cls.__name__}")
        return _PROXY_CACHE[model_cls]

    log.debug(f"Generating fields proxy object for {model_cls.__name__} (base_path='{base_path}')")
    proxy_obj = SimpleNamespace()
    prefix = f"{base_path}." if base_path else ""
    try:
        try:
            annotations = get_type_hints(model_cls, include_extras=True)
        except Exception:
            log.warning(f"get_type_hints failed for {model_cls.__name__}, falling back.")
            annotations = getattr(model_cls, "__annotations__", {})

        log.debug(f"Introspecting fields for {model_cls.__name__} (prefix='{prefix}'): {list(annotations.keys())}")
        for name, type_hint in annotations.items():
            if name.startswith("_"):
                continue
            full_path = f"{prefix}{name}"
            # Create Field[SpecificType] to potentially help type checkers
            setattr(proxy_obj, name, Field[type_hint](full_path))
            log.debug(f"Added Field for '{name}' to proxy object.")
    except Exception as e:
        log.error(f"Failed field proxy generation for {model_cls.__name__}", exc_info=True)
        raise TypeError(f"Could not generate query fields proxy for {model_cls.__name__}") from e

    if not base_path:
        _PROXY_CACHE[model_cls] = proxy_obj
    log.debug(f"Finished generating fields proxy object for {model_cls.__name__}")
    return proxy_obj


# --- Generic Fields Proxy (Model-less) ---
class GenericFieldsProxy:
    """Creates Field instances dynamically for any attribute access."""
    __slots__ = () # Optimize memory, prevent instance dict
    def __getattr__(self, name: str) -> Field[Any]:
        if name.startswith("_"):
            raise AttributeError(f"No attribute '{name}'")
        log.debug(f"GenericFieldsProxy: Creating Field for attribute '{name}'")
        return Field(name)
    def __dir__(self) -> List[str]:
        # Cannot reliably list fields without a model
        return []


# --- Query Builder ---
@dataclass_transform()
class QueryBuilder(Generic[M]):
    """
    Builds database queries, optionally validated against model `M`.

    If `model_cls` is provided, `fields` will be populated based on the model
    and validation will occur. If `model_cls` is None, `fields` allows access
    to any attribute, creating `Field` objects dynamically without validation.
    """
    model_cls: Optional[Type[M]]
    fields: Any  # Will be SimpleNamespace or GenericFieldsProxy
    _validator: Optional[ModelValidator[M]]
    _expression: Optional[Expression]
    _options: Dict[str, Any]
    _logger: logging.Logger

    def __init__(self, model_cls: Optional[Type[M]] = None):
        self._logger = log
        self.model_cls = model_cls
        self._expression = None
        self._options = {
            "limit": 100, "offset": 0, "sort_desc": False,
            "sort_by": None, "random_order": False, "timeout": None
        }
        self._validator = None
        self.fields = None # Assigned below

        if model_cls:
            self._logger.info(f"Initializing QueryBuilder WITH validation for: {model_cls.__name__}")
            try:
                self.fields = _generate_fields_proxy(model_cls)
                self._validator = ModelValidator(model_cls)
                self._logger.info(f"QueryBuilder for {model_cls.__name__} initialized.")
            except Exception as e:
                self._logger.error(f"Failed QueryBuilder init for {model_cls.__name__}", exc_info=True)
                self.fields = GenericFieldsProxy() # Fallback
                self._validator = None
                raise ValueError(f"Could not initialize QueryBuilder for {model_cls.__name__}") from e
        else:
            self._logger.info("Initializing QueryBuilder WITHOUT model validation.")
            self.fields = GenericFieldsProxy()
            self._validator = None

        self._logger.debug(f"Default query options set: {self._options}")


    def filter(self, expr: Expression) -> "QueryBuilder[M]":
        """Adds a filter expression (combined with AND if one exists)."""
        self._logger.debug(f"Adding filter expression: {expr!r}")
        if not isinstance(expr, Expression):
            raise TypeError(f"filter() requires an Expression object, got {type(expr).__name__}")

        # Conditionally validate
        if self._validator:
            try:
                self._validate_expression_runtime(expr)
                self._logger.debug(f"Runtime validation successful for: {expr!r}")
            except (ValueTypeError, InvalidPathError) as e:
                # Wrap specific validation errors in ValueError for consistent API
                raise ValueError(f"Invalid filter expression: {e}") from e
            except Exception as e:
                # Wrap unexpected errors
                raise RuntimeError(f"Unexpected error validating filter {expr!r}: {e}") from e
        else:
            self._logger.debug("Skipping filter validation (no model validator).")

        # Combine expressions using AND
        if self._expression is None:
            self._expression = expr
        else:
            self._expression = self._expression & expr

        self._logger.debug(f"Current expression is now: {self._expression!r}")
        return self


    def _validate_expression_runtime(self, expr: Expression) -> None:
        """Internal: Validates expression if validator is present."""
        assert self._validator is not None, "_validate_expression_runtime called without validator"

        if isinstance(expr, FilterCondition):
            field_path = expr.field_path
            operator = expr.operator
            value = expr.value
            log.debug(f"Runtime validating: {expr!r}")
            try:
                field_type = self._validator.get_field_type(field_path)
                log.debug(f"  Field '{field_path}' expected type: {field_type!r}")

                # --- Operator-Specific Validation ---
                if operator == "exists":
                    # Boolean value type already checked in Field._op
                    return # No further validation needed

                if operator in ("in", "nin"):
                    # Validate items in the input list against the field's expected item type
                    is_list_field, item_type_list = self._validator.get_list_item_type(field_path)
                    is_set_field = False
                    item_type_set = Any
                    if not is_list_field: # Check if it's a Set
                        origin = get_origin(field_type)
                        args = get_args(field_type)
                        # Handle Optional[Set[T]]
                        if origin is Union and any(_is_none_type(a) for a in args):
                             non_none = tuple(t for t in args if not _is_none_type(t))
                             if len(non_none) == 1:
                                 inner_type = non_none[0]
                                 origin = get_origin(inner_type)
                                 args = get_args(inner_type)
                        if origin in (set, Set) and args:
                            is_set_field = True
                            item_type_set = args[0]

                    # Determine expected type for items in the 'value' list
                    if is_list_field:
                        expected_item_type = item_type_list
                    elif is_set_field:
                         expected_item_type = item_type_set
                    else:
                        # If field is not list/set, items must match field type itself
                        expected_item_type = field_type

                    log.debug(f"  Validating '{operator}' items against type: {expected_item_type!r}")
                    if value: # value is already guaranteed to be a list here by Field._op
                        for i, item in enumerate(value):
                           try:
                               self._validator.validate_value(
                                   item, expected_item_type, f"{field_path} ({operator} item {i})"
                               )
                           except ValueTypeError as e:
                               # Re-raise with context
                               raise ValueTypeError(
                                   f"Invalid item in '{operator}' list for field "
                                   f"'{field_path}'. {e}", path=field_path
                               ) from e
                    return # Done with in/nin validation

                if operator == "contains":
                    # Validate 'contains' value against item type (if list/set) or field type (if str)
                    is_list, item_type_list = self._validator.get_list_item_type(field_path)
                    is_set = False
                    item_type_set = Any
                    actual_field_type = field_type # Start with original type
                    origin = get_origin(field_type)
                    args = get_args(field_type)

                    # Handle Optional field before checking if it's list/set/str
                    if origin is Union and any(_is_none_type(a) for a in args):
                        non_none = tuple(t for t in args if not _is_none_type(t))
                        if len(non_none) == 1:
                            actual_field_type = non_none[0]
                            origin = get_origin(actual_field_type)
                            args = get_args(actual_field_type)
                        # else: leave as Union for now, handled below

                    # Check if the (potentially unwrapped) type is List or Set
                    if origin in (list, List) and args:
                        is_list = True; item_type_list = args[0]
                    elif origin in (set, Set) and args:
                        is_set = True; item_type_set = args[0]

                    if is_list or is_set:
                        item_type = item_type_list if is_list else item_type_set
                        coll_name = "list" if is_list else "set"
                        log.debug(f"  Validating 'contains' value against item type "
                                  f"{item_type!r} for {coll_name} field.")
                        try:
                            self._validator.validate_value(value, item_type, f"{field_path} (contains value)")
                        except ValueTypeError as e:
                            raise ValueTypeError(
                                f"Operator 'contains' on field '{field_path}' requires value "
                                f"compatible with item type {self._validator._get_type_name(item_type)}, "
                                f"got {type(value).__name__}. Original error: {e}", path=field_path
                            ) from e
                        return # Done contains validation for list/set

                    # Check if string field (use actual_field_type after Optional unwrap)
                    if actual_field_type is str:
                        if not isinstance(value, str):
                            raise ValueTypeError(
                                f"Operator 'contains' on string field '{field_path}' "
                                f"requires a string value, got {type(value).__name__}", path=field_path
                            )
                        return # Done contains validation for str

                    # If not list/set/str, fall through after warning
                    log.warning(f"Runtime check for 'contains' on non-list/set/string "
                                f"field '{field_path}' ({field_type!r}) is limited.")

                # --- General Value Validation (Catches remaining basic type errors) ---
                # This runs for ops like eq, ne, gt, lt, ge, le and falls through
                # for contains on unsupported types. It also ensures value type matches
                # field type even for ops like like, startswith, endswith.
                self._validator.validate_value(value, field_type, field_path)

                # --- Operator vs Field Type Compatibility ---
                # Check if the operator is appropriate for the field's type *after*
                # ensuring the value itself is compatible with the field type.
                if operator in ("gt", "lt", "ge", "le"):
                    # Get the non-optional base type for check
                    _origin = get_origin(field_type); _args = get_args(field_type)
                    _actual_type = field_type
                    if _origin is Union and any(_is_none_type(a) for a in _args):
                        _non_none = tuple(t for t in _args if not _is_none_type(t))
                        if len(_non_none) == 1: _actual_type = _non_none[0]

                    # Check if the field type is orderable (numeric, string, date etc.)
                    if not self._validator._is_single_type_numeric(_actual_type) and \
                       not _actual_type is str:
                           # Add checks for other known comparable types like datetime here if needed
                           log.warning(f"Operator '{operator}' used on non-numeric, "
                                       f"non-string field '{field_path}' "
                                       f"({field_type!r}). Comparison support depends on backend.")

                elif operator in ("like", "startswith", "endswith"):
                    # Ensure field is string-like
                    _origin = get_origin(field_type); _args = get_args(field_type)
                    _actual_type = field_type
                    if _origin is Union and any(_is_none_type(a) for a in _args):
                         _non_none = tuple(t for t in _args if not _is_none_type(t))
                         if len(_non_none) == 1: _actual_type = _non_none[0]

                    if _actual_type is not str:
                         raise ValueTypeError(
                             f"Operator '{operator}' typically requires a string field, "
                             f"but field '{field_path}' has type "
                             f"{self._validator._get_type_name(field_type)}", path=field_path
                         )

            # --- Exception Handling for Validation ---
            except (InvalidPathError, ValueTypeError) as e:
                # Re-raise specific validation errors directly
                raise e
            except Exception as e:
                # Wrap unexpected errors
                log.error(f"Unexpected error during runtime validation of "
                          f"condition ({expr!r}): {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error validating condition {expr!r}: {e}") from e

        elif isinstance(expr, CombinedCondition):
            # Recursively validate combined expressions
            self._validate_expression_runtime(expr.left)
            self._validate_expression_runtime(expr.right)
        else:
            # Should not happen if filter() checks input type
            raise TypeError(f"Cannot validate unknown expression type: {type(expr).__name__}")


    def sort_by(self, field: Field[Any], descending: bool = False) -> "QueryBuilder[M]":
        """Sets the sort field and order."""
        self._logger.debug(f"Setting sort order: field={field!r}, descending={descending}")
        if not isinstance(field, Field):
            raise TypeError("sort_by requires a Field object")

        # Conditionally validate path
        if self._validator:
            try:
                self._validator.get_field_type(field.path) # Validate path exists
            except InvalidPathError as e:
                # Raise AttributeError for consistency with field access errors
                raise AttributeError(f"Invalid sort field path: {field.path}. Error: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error validating sort field {field.path}: {e}") from e
        else:
            self._logger.debug("Skipping sort_by path validation.")

        self._options["sort_by"] = field.path
        self._options["sort_desc"] = descending
        if self._options.get("random_order"):
            self._options["random_order"] = False # Sorting overrides random
        self._logger.info(f"Sort order set: field='{field.path}', descending={descending}")
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
        if seconds is not None and (not isinstance(seconds, (int, float)) or seconds < 0):
            raise ValueError("Timeout must be a non-negative number or None.")
        self._options["timeout"] = seconds
        self._logger.info(f"Query timeout set to: {seconds}")
        return self

    def build(self) -> QueryOptions:
        """Builds the final QueryOptions object."""
        self._logger.debug("Building QueryOptions...")
        expression_dict = self._expression.to_dict() if self._expression else {}
        options = QueryOptions(
            expression=expression_dict,
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