from inspect import isclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    TypeVar,  # Import TypeVar
    Generic,  # Import Generic
    get_args,
    get_origin,
    get_type_hints,
)

# --- Generic Type Variable ---
M = TypeVar("M")  # Represents the specific model class (e.g., User, Address)


# --- Validation Exceptions ---
# (Keep these exactly as they are)
class ValidationError(TypeError):
    """Base class for validation errors related to model types."""
    pass


class InvalidPathError(ValidationError, AttributeError):
    """Error raised when a field path does not exist or is invalid for the model."""
    pass


class ValueTypeError(ValidationError, TypeError):
    """Error raised when a value's type is incompatible with the expected field type."""
    pass


# --- Helper Function ---
# (Keep this exactly as it is)
def _is_none_type(t: Type) -> bool:
    """Checks if a type is NoneType."""
    return t is type(None)


# --- Model Validator (Now Generic) ---

# Make the class generic on the model type M
class ModelValidator(Generic[M]):
    """
    Performs validation of field paths and values against a given model type M.
    Handles nested structures, lists, dictionaries, Unions, Optionals, etc.
    Provides runtime validation support for query builders.
    """

    model_type: Type[M]  # Add type hint for the instance variable
    # Cache for type hints to avoid repeated introspection
    _type_hints_cache: Dict[Type, Dict[str, Type]]

    # Use Type[M] in the constructor signature
    def __init__(self, model_type: Type[M]):
        """
        Initializes the validator with a model type.

        Args:
            model_type: The class definition (e.g., Pydantic model, dataclass, standard class)
                        against which validation will be performed. Type M associated with this validator.

        Raises:
            TypeError: If model_type is not a class.
        """
        if not isclass(model_type):
            raise TypeError(f"model_type must be a class, received {type(model_type)}.")
        self.model_type = model_type
        self._type_hints_cache = {}  # Initialize cache here

    # --- Internal Methods (_get_cached_type_hints, _traverse_path) ---
    # These methods remain the same internally. They operate correctly
    # on the self.model_type which is now Type[M].

    def _get_cached_type_hints(self, cls: Type) -> Dict[str, Type]:
        """Gets type hints for a class, using a cache."""
        if cls not in self._type_hints_cache:
            try:
                # Resolve forward references and include extras for comprehensive hints
                # Use globalns=getattr(cls, '__dict__', None) ? Maybe not needed with modern typing.
                hints = get_type_hints(cls, include_extras=True)
                self._type_hints_cache[cls] = hints
            except Exception as e:
                # More specific error handling might be needed for complex cases
                raise TypeError(f"Could not get type hints for {cls.__name__}. Error: {e}") from e
        return self._type_hints_cache[cls]

    def _traverse_path(self, field_path: str) -> Type:
        """
        Internal logic to traverse a field path and return the final type.
        Raises InvalidPathError if the path is invalid.
        """
        parts = field_path.split(".")
        current_type = self.model_type
        current_path_parts = []

        for part in parts:
            current_path_parts.append(part)
            full_path_str = ".".join(current_path_parts)
            parent_path_str = ".".join(current_path_parts[:-1]) or "root"

            origin = get_origin(current_type)
            args = get_args(current_type)

            # Check for dictionary with non-string keys first (example check)
            if origin is dict and args and len(args) == 2:
                key_type, _ = args
                if key_type is not str and part.isdigit():  # Basic check, might need refinement
                    raise InvalidPathError(
                        f"Cannot traverse Dict path '{full_path_str}' with non-string key type {key_type}."
                    )

            # --- List/Tuple/Set Index Handling ---
            if part.isdigit():
                # ... (keep your original index handling logic for list, tuple, set, Optional[...]) ...
                index = int(part)
                actual_iterable_type = None
                item_type = Any
                is_tuple = False

                # Check direct type or origin
                if current_type is list or origin is list:
                    actual_iterable_type = current_type
                elif current_type is tuple or origin is tuple:
                    actual_iterable_type = current_type
                    is_tuple = True
                elif current_type is set or origin is set:
                    actual_iterable_type = current_type
                # Check Optional[Iterable]
                elif origin is Union:
                    potential_iterable = None
                    has_none = False
                    for arg in args:
                        if _is_none_type(arg):
                            has_none = True
                        elif get_origin(arg) in (list, tuple, set) or arg in (list, tuple, set):
                            potential_iterable = arg
                            is_tuple = get_origin(arg) is tuple or arg is tuple
                    if has_none and potential_iterable:
                        actual_iterable_type = potential_iterable

                if actual_iterable_type:
                    iterable_args = get_args(actual_iterable_type)
                    if is_tuple:  # Fixed-size tuple logic
                        if iterable_args and 0 <= index < len(iterable_args):
                            item_type = iterable_args[index]
                        # else: index out of bounds, item_type remains Any
                    else:  # List or Set logic
                        if iterable_args: item_type = iterable_args[0]
                        # else: plain list/set, item_type remains Any
                    current_type = item_type
                    continue  # Move to next part
                else:
                    raise InvalidPathError(
                        f"Cannot apply index '{part}' to non-list/tuple/set field "
                        f"'{parent_path_str}' (type: {current_type}) in path '{field_path}'."
                    )

            # --- Attribute/Dict Key Handling ---
            # Handle Optional[T] -> T for attribute access
            if origin is Union:
                non_none_args = [arg for arg in args if not _is_none_type(arg)]
                if len(non_none_args) == 1 and any(_is_none_type(arg) for arg in args):
                    current_type = non_none_args[0]
                    origin = get_origin(current_type)  # Update origin for checks below
                    args = get_args(current_type)
                else:
                    # Cannot traverse into non-Optional Union
                    raise InvalidPathError(
                        f"Cannot access nested field '{part}' on Union type field "
                        f"'{parent_path_str}' (type: {current_type}). Path: '{field_path}'."
                    )

            # Standard class attribute or Dict[str, V] key access
            if isclass(current_type) and hasattr(current_type, "__annotations__"):
                type_hints = self._get_cached_type_hints(current_type)
                if part not in type_hints:
                    raise InvalidPathError(
                        f"Field '{part}' does not exist in type {current_type.__name__} "
                        f"(path: '{full_path_str}')."
                    )
                current_type = type_hints[part]
            elif origin is dict and args and len(args) == 2:
                key_type, value_type = args
                if key_type is not str:
                    raise InvalidPathError(
                        f"Cannot traverse Dict path '{full_path_str}' via attribute access "
                        f"with non-string key type {key_type}."
                    )
                current_type = value_type  # Continue with value type
            elif origin is Any or current_type is Any:
                current_type = Any
            else:
                # Cannot traverse further (e.g., attribute on int)
                raise InvalidPathError(
                    f"Cannot access nested field '{part}'. Parent field '{parent_path_str}' "
                    f"is not a traversable class or Dict[str, ...] (type: {current_type}). Path: '{field_path}'."
                )

            if current_type is Any: break  # Stop if we hit Any

        return current_type

    # --- Public API Methods (get_field_type, validate_value, etc.) ---
    # These methods remain the same internally.

    def get_field_type(self, field_path: str) -> Type:
        """
        Gets the expected type hint for a given field path within the model M.
        """
        if not field_path:
            raise ValueError("field_path cannot be empty.")
        try:
            return self._traverse_path(field_path)
        except InvalidPathError as e:
            # Add model context to the error
            raise InvalidPathError(f"{e} in model {self.model_type.__name__}") from e
        except TypeError as e:  # Catch errors from _get_cached_type_hints
            raise TypeError(
                f"Error processing type hints for path '{field_path}' in model {self.model_type.__name__}: {e}") from e

    def validate_value(self, value: Any, expected_type: Type,
                       error_context_path: str = "value") -> None:
        """
        Recursively validates a value against an expected type hint.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle bool vs int distinction early
        if expected_type is int and isinstance(value, bool):
            raise ValueTypeError(f"Path '{error_context_path}': expected int, got bool.")

        # 1. Handle Optional[T]
        is_optional = origin is Union and args and _is_none_type(args[-1])
        if is_optional:
            if value is None: return  # None is valid
            non_none_types = tuple(t for t in args if not _is_none_type(t))
            if len(non_none_types) == 1:  # Optional[T] -> T
                expected_type = non_none_types[0]
                origin = get_origin(expected_type);
                args = get_args(expected_type)
            else:  # Optional[Union[A, B]] -> Union[A, B]
                expected_type = Union[non_none_types]
                origin = Union;
                args = non_none_types  # Update for Union check below

        # 2. Handle Any
        if expected_type is Any:
            if value is None and not is_optional:  # None requires Optional[Any]
                raise ValueTypeError(
                    f"Path '{error_context_path}': received None but expected Any.")
            return

        # 3. Handle None when not Optional
        if value is None:  # Already checked for Optional above
            raise ValueTypeError(
                f"Path '{error_context_path}': received None but expected {expected_type}.")

        # 4. Handle Union[A, B, ...]
        if origin is Union:
            # Handle bool special case for Union containing int but not str
            if isinstance(value, bool) and int in args and str not in args:
                raise ValueTypeError(
                    f"Path '{error_context_path}': bool value {repr(value)} not valid for Union {expected_type} containing int.")
            # Try matching against each type in the Union
            for possible_type in args:
                try:
                    self.validate_value(value, possible_type, error_context_path);
                    return
                except ValueTypeError:
                    continue
            raise ValueTypeError(
                f"Path '{error_context_path}': value {repr(value)} ({type(value).__name__}) does not match any type in {expected_type}.")

        # 5. Handle List, Set, Tuple
        if origin in (list, set, tuple):
            if not isinstance(value, origin): raise ValueTypeError(
                f"Path '{error_context_path}': expected {origin.__name__}, got {type(value).__name__}.")
            if args:  # Type arguments provided (e.g., List[int])
                if origin is tuple:  # Tuple[T1, T2] or Tuple[T, ...]
                    is_variadic = len(args) == 2 and args[1] == ...
                    if is_variadic:  # Tuple[T, ...]
                        item_type = args[0]
                        for i, item in enumerate(value): self.validate_value(item, item_type,
                                                                             f"{error_context_path}[{i}]")
                    else:  # Tuple[T1, T2]
                        if len(value) != len(args): raise ValueTypeError(
                            f"Path '{error_context_path}': expected tuple length {len(args)}, got {len(value)}.")
                        for i, (item, item_type) in enumerate(
                            zip(value, args)): self.validate_value(item, item_type,
                                                                   f"{error_context_path}[{i}]")
                else:  # List[T] or Set[T]
                    item_type = args[0]
                    for i, item in enumerate(value): self.validate_value(item, item_type,
                                                                         f"{error_context_path}[{i}]")
            return  # Validation done for collection itself and potentially items

        # 6. Handle Dict
        if origin is dict:
            if not isinstance(value, dict): raise ValueTypeError(
                f"Path '{error_context_path}': expected dict, got {type(value).__name__}.")
            if len(args) == 2:  # Dict[K, V]
                key_type, value_type = args
                if key_type is not Any or value_type is not Any:
                    for k, v in value.items():
                        if key_type is not Any: self.validate_value(k, key_type,
                                                                    f"{error_context_path} key '{k}'")
                        if value_type is not Any: self.validate_value(v, value_type,
                                                                      f"{error_context_path}['{k}']")
            return  # Validation done

        # 7. Handle Specific Classes (including compatible dicts)
        if isclass(expected_type):
            if isinstance(value, dict) and hasattr(expected_type, "__annotations__"):
                # Get the type hints for the expected class
                type_hints = self._get_cached_type_hints(expected_type)

                # Check for missing required fields in the dict
                for field_name, field_type in type_hints.items():
                    # Skip if field is present
                    if field_name in value:
                        continue

                    # Check if field is Optional or has a default value
                    field_origin = get_origin(field_type)
                    field_args = get_args(field_type)
                    is_field_optional = (field_origin is Union and
                                         any(_is_none_type(arg) for arg in field_args))

                    # If field is NOT optional and NOT present, raise error
                    if not is_field_optional:
                        raise ValueTypeError(
                            f"Path '{error_context_path}': missing required field '{field_name}'"
                        )

                # Validate the fields that are present
                for field_name, field_value in value.items():
                    if field_name in type_hints:
                        field_type = type_hints[field_name]
                        self.validate_value(field_value, field_type,
                                            f"{error_context_path}.{field_name}")
                    # Ignore extra fields for now - could add strict mode later

                return  # Dict validated against class
            elif isinstance(value, expected_type):
                return  # Instance is already correct type
            else:
                raise ValueTypeError(
                    f"Path '{error_context_path}': expected {expected_type.__name__} or compatible dict, got {repr(value)} ({type(value).__name__}).")

        # 8. Handle basic types / Fallback isinstance check
        if not isinstance(value, expected_type):
            # Check again for bool vs int specifically if needed
            if expected_type is int and isinstance(value,
                                                   bool):  # Should be caught earlier, but belt-and-suspenders
                raise ValueTypeError(f"Path '{error_context_path}': expected int, got bool.")
            raise ValueTypeError(
                f"Path '{error_context_path}': expected type {expected_type.__name__}, got {repr(value)} ({type(value).__name__}).")

    def validate_value_for_path(self, field_path: str, value: Any) -> None:
        """
        Validates that a value is compatible with the type expected at a specific field path.
        """
        expected_type = self.get_field_type(field_path)
        self.validate_value(value, expected_type, field_path)

    # --- Helper Methods (_is_single_type_numeric, is_field_numeric, get_list_item_type) ---
    # These methods remain the same internally.

    def _is_single_type_numeric(self, single_type: Type) -> bool:
        """Helper to check if a non-Union type is numeric (int/float, not bool/Any)."""
        if single_type is Any: return False
        return isclass(single_type) and issubclass(single_type,
                                                   (int, float)) and single_type is not bool

    def is_field_numeric(self, field_path: str) -> bool:
        """
        Checks if the field at the given path is a numeric type (int, float,
        or an Optional/Union containing them, excluding bool).
        """
        try:
            field_type = self.get_field_type(field_path)
        except InvalidPathError:
            return False
        origin = get_origin(field_type);
        args = get_args(field_type)
        if origin is Union:
            return any(self._is_single_type_numeric(arg) for arg in args if not _is_none_type(arg))
        return self._is_single_type_numeric(field_type)

    def get_list_item_type(self, field_path: str) -> Tuple[bool, Type]:
        """
        Checks if field is a List or Optional[List] and returns its item type.
        """
        field_type = self.get_field_type(field_path)  # Raises InvalidPathError if path bad
        origin = get_origin(field_type);
        args = get_args(field_type)
        is_list_type = False;
        item_type = Any;
        actual_list_type = None

        if origin is list or field_type is list:
            is_list_type = True;
            actual_list_type = field_type
        elif origin is Union:
            maybe_list_arg = None;
            has_none = False
            for arg in args:
                if _is_none_type(arg):
                    has_none = True
                elif get_origin(arg) is list or arg is list:
                    maybe_list_arg = arg
            if has_none and maybe_list_arg:
                is_list_type = True;
                actual_list_type = maybe_list_arg

        if actual_list_type:
            list_args = get_args(actual_list_type)
            if list_args: item_type = list_args[0]

        return is_list_type, item_type