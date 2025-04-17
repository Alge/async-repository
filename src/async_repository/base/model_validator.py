# -*- coding: utf-8 -*-
"""
Provides the ModelValidator class for validating field paths and values
against a Python class definition with type hints.
"""

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
    get_args,
    get_origin,
    get_type_hints,
)


# --- Validation Exceptions ---

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

def _is_none_type(t: Type) -> bool:
    """Checks if a type is NoneType."""
    # Using `is` comparison for the singleton NoneType
    return t is type(None)  # pylint: disable=unidiomatic-typecheck


# --- Model Validator ---

class ModelValidator:
    """
    Performs validation of field paths and values against a given model type.
    Handles nested structures, lists, dictionaries, Unions, Optionals, etc.
    """

    def __init__(self, model_type: Type):
        """
        Initializes the validator with a model type.

        Args:
            model_type: The class definition (e.g., Pydantic model, dataclass, standard class)
                        against which validation will be performed.

        Raises:
            TypeError: If model_type is not a class.
        """
        if not isclass(model_type):
            raise TypeError(f"model_type must be a class, received {type(model_type)}.")
        self.model_type = model_type
        # Cache for type hints to avoid repeated introspection
        self._type_hints_cache: Dict[Type, Dict[str, Type]] = {}

    def _get_cached_type_hints(self, cls: Type) -> Dict[str, Type]:
        """Gets type hints for a class, using a cache."""
        if cls not in self._type_hints_cache:
            try:
                # Resolve forward references and include extras for comprehensive hints
                self._type_hints_cache[cls] = get_type_hints(cls, include_extras=True)
            except Exception as e:
                # Handle cases where getting hints fails (e.g., complex generics, missing imports)
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

            # Check for dictionary with non-string keys first
            if origin is dict and args and len(args) == 2:
                key_type, value_type = args
                if key_type is not str and part.isdigit():
                    # Dot notation with numeric keys needs special handling for Dict[int, ...]
                    raise InvalidPathError(
                        f"Cannot traverse Dict path '{full_path_str}' with non-string key type {key_type}. "
                        "Use explicit dictionary access if needed."
                    )

            if part.isdigit():
                # --- List/Tuple/Set Index Handling ---
                index = int(part)
                actual_list_type = None
                actual_tuple_type = None
                actual_set_type = None

                # Check for list types
                if origin is list:
                    actual_list_type = current_type
                elif current_type is list:
                    actual_list_type = current_type  # Handle plain 'list' type
                # Check for tuple types
                elif origin is tuple:
                    actual_tuple_type = current_type
                # Check for set types
                elif origin is set:
                    actual_set_type = current_type
                # Check for Optional[List/Tuple/Set]
                elif origin is Union:
                    maybe_list_arg = None
                    maybe_tuple_arg = None
                    maybe_set_arg = None
                    has_none = False
                    for arg in args:
                        if _is_none_type(arg):
                            has_none = True
                        elif get_origin(arg) is list or arg is list:
                            maybe_list_arg = arg
                        elif get_origin(arg) is tuple:
                            maybe_tuple_arg = arg
                        elif get_origin(arg) is set:
                            maybe_set_arg = arg

                    if has_none and maybe_list_arg:
                        actual_list_type = maybe_list_arg
                    elif has_none and maybe_tuple_arg:
                        actual_tuple_type = maybe_tuple_arg
                    elif has_none and maybe_set_arg:
                        actual_set_type = maybe_set_arg

                # Handle lists
                if actual_list_type is not None:
                    list_args = get_args(actual_list_type)
                    # Default to Any if List has no specific item type (e.g., just `list`)
                    current_type = list_args[0] if list_args else Any
                    continue  # Move to next part

                # Handle tuples
                if actual_tuple_type is not None:
                    tuple_args = get_args(actual_tuple_type)
                    if 0 <= index < len(tuple_args):
                        current_type = tuple_args[index]
                    else:
                        # Index out of bounds for the tuple, use Any
                        current_type = Any
                    continue  # Move to next part

                # Handle sets (similar to lists, but with set element type)
                if actual_set_type is not None:
                    set_args = get_args(actual_set_type)
                    # Default to Any if Set has no specific item type
                    current_type = set_args[0] if set_args else Any
                    continue  # Move to next part

                # If neither list, tuple, nor set, raise error
                raise InvalidPathError(
                    f"Cannot apply index '{part}' to non-list/tuple/set field "
                    f"'{parent_path_str}' (type: {current_type}) in path '{field_path}'."
                )

            # --- Attribute/Dict Key Handling ---
            if origin is Union:
                # Check if this is an Optional type (Union[T, None])
                non_none_args = [arg for arg in args if not _is_none_type(arg)]
                if len(non_none_args) == 1 and any(_is_none_type(arg) for arg in args):
                    # This is an Optional[T], so extract T and continue traversal
                    current_type = non_none_args[0]
                else:
                    # Cannot reliably traverse into a general Union type with dot notation
                    raise InvalidPathError(
                        f"Cannot access nested field '{part}' on Union type field "
                        f"'{parent_path_str}' (type: {current_type}). Ambiguous path in '{field_path}'."
                    )

            if isclass(current_type) and hasattr(current_type, "__annotations__"):
                # Standard class attribute access
                type_hints = self._get_cached_type_hints(current_type)
                if part not in type_hints:
                    raise InvalidPathError(
                        f"Field '{part}' does not exist in type {current_type.__name__} "
                        f"(path: '{full_path_str}')."
                    )
                current_type = type_hints[part]

            elif origin is dict and args and len(args) == 2:
                # Access into a typed Dict (e.g., Dict[str, int])
                key_type, value_type = args
                # Only allow traversal via dot notation if key type is str
                if key_type is not str:
                    raise InvalidPathError(
                        f"Cannot traverse Dict path '{full_path_str}' with non-string key type {key_type}. "
                        "Use explicit dictionary access if needed."
                    )
                # Note: We don't validate the 'part' against key_type here, just assume string access.
                # We only care about the value type for further traversal.
                current_type = value_type  # Continue traversal with the value type
            elif origin is Any or current_type is Any:
                # If we encounter Any, subsequent path parts are also Any
                current_type = Any
            else:
                # Cannot traverse further (e.g., accessing attribute on primitive type)
                raise InvalidPathError(
                    f"Cannot access nested field '{part}'. Parent field "
                    f"'{parent_path_str}' is not a traversable class or Dict[str, ...] "
                    f"(type: {current_type}) in path '{field_path}'."
                )

            # Optimization: if we hit Any, no further type info is available down the path
            if current_type is Any:
                break

        return current_type

    def get_field_type(self, field_path: str) -> Type:
        """
        Gets the expected type hint for a given field path within the model.

        Args:
            field_path: The dot-separated path (e.g., "user.address.zipcode", "items.0.id").

        Returns:
            The type hint of the specified field. Returns `Any` if the path leads to an
            unspecified type (like a plain `list` or `dict`).

        Raises:
            ValueError: If field_path is empty.
            InvalidPathError: If the path is invalid for the model structure.
        """
        if not field_path:
            raise ValueError("field_path cannot be empty.")
        return self._traverse_path(field_path)

    def validate_value(self, value: Any, expected_type: Type,
                       error_context_path: str = "value") -> None:
        """
        Recursively validates a value against an expected type hint.

        Args:
            value: The value to validate.
            expected_type: The type hint the value should conform to.
            error_context_path: String describing the value's location for clearer error messages.

        Raises:
            ValueTypeError: If the value does not match the expected type.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Handle special case for bool vs int
        # Bool is a subclass of int in Python, but for validation we want to treat them as distinct
        if expected_type is int and isinstance(value, bool):
            raise ValueTypeError(
                f"Path '{error_context_path}': expected int, got bool ({repr(value)})."
            )

        # 1. Handle Optional[T] (represented as Union[T, NoneType])
        if origin is Union and args and _is_none_type(args[-1]):
            if value is None:
                return  # None is valid for Optional[T]
            # Validate against the non-None type(s) in the Optional
            non_none_types = tuple(t for t in args if not _is_none_type(t))
            if len(non_none_types) == 1:
                # Reduce Optional[T] to T for further validation
                expected_type = non_none_types[0]
                origin = get_origin(expected_type)  # Update origin/args for T
                args = get_args(expected_type)
            else:
                # Reduce Optional[Union[A,B]] to Union[A,B]
                expected_type = Union[non_none_types]
                # Fall through to Union validation logic

        # 2. Handle Any: Allows any value (except None unless Optional[Any])
        if expected_type is Any:
            if value is None:
                raise ValueTypeError(
                    f"Path '{error_context_path}': received None but expected {expected_type}.") # Keep Any as Any
            return

        # 3. Handle None value when type is not Optional (checked after Optional handling)
        if value is None:
            # If we reach here, expected_type is not Optional or Union[..., None]
            raise ValueTypeError(
                f"Path '{error_context_path}': received None but expected {expected_type.__name__}.") # MODIFIED: Use __name__

        # 4. Handle Union[A, B, ...] (that are not Optional)
        if origin is Union:
            # Special case: If we have a bool value and int is one of the Union types,
            # but str is not in the Union, we need to reject the bool
            if isinstance(value, bool) and int in args and str not in args:
                raise ValueTypeError(
                    f"Path '{error_context_path}': value {repr(value)} (type bool) does not match any type in {expected_type}. "
                    f"Bool values are not accepted for int in Union types."
                )

            # Value must match at least one of the types in the Union
            for possible_type in args:
                try:
                    # Recursively validate against each type in the Union
                    self.validate_value(value, possible_type, error_context_path)
                    return  # Matched one type in the Union
                except ValueTypeError:
                    continue  # Try the next type
            # If loop finishes, value didn't match any type in the Union
            raise ValueTypeError(
                f"Path '{error_context_path}': value {repr(value)} (type {type(value).__name__}) "
                f"does not match any type in {expected_type}." # Keep full Union representation here
            )

        # 5. Handle List, Set, Tuple
        if origin in (list, set, tuple):
            if not isinstance(value, origin):
                raise ValueTypeError(
                    f"Path '{error_context_path}': expected {origin.__name__}, got {type(value).__name__}."
                )

            # Special handling for tuples with specific types for each position
            if origin is tuple and args:
                # Check if this is a variable-length tuple (Tuple[int, ...])
                is_variadic = len(args) == 2 and args[1] == ...

                if is_variadic:
                    # Variadic tuple (Tuple[int, ...]) - all elements must match the first type
                    item_type = args[0]
                    for i, item in enumerate(value):
                        self.validate_value(item, item_type, f"{error_context_path}[{i}]")
                else:
                    # Fixed-length tuple with potentially different types (Tuple[int, str, bool])
                    if len(value) != len(args):
                        raise ValueTypeError(
                            f"Path '{error_context_path}': expected tuple of length {len(args)}, "
                            f"got length {len(value)}."
                        )

                    # Validate each position with its corresponding type
                    for i, (item, item_type) in enumerate(zip(value, args)):
                        self.validate_value(item, item_type, f"{error_context_path}[{i}]")

            # Handle lists and sets (all elements have the same type)
            elif args and args[0] is not Any:
                item_type = args[0]
                for i, item in enumerate(value):
                    self.validate_value(item, item_type, f"{error_context_path}[{i}]")

            return

        # 6. Handle Dict
        if origin is dict:
            if not isinstance(value, dict):
                raise ValueTypeError(
                    f"Path '{error_context_path}': expected dict, got {type(value).__name__}."
                )
            # If type has arguments (e.g., Dict[str, int]), validate keys and values
            if len(args) == 2 and (args[0] is not Any or args[1] is not Any):
                key_type, value_type = args
                for k, v in value.items():
                    # Use recursive calls for key/value validation
                    if key_type is not Any: self.validate_value(k, key_type,
                                                                f"{error_context_path} key '{k}'")
                    if value_type is not Any: self.validate_value(v, value_type,
                                                                  f"{error_context_path}['{k}']")
            return

        # 7. Handle Specific Classes (Pydantic models, standard classes with hints)
        if isclass(expected_type):
            # Case 7a: Value is a dictionary, expected type is a class.
            # Validate dictionary contents against the class annotations.
            if isinstance(value, dict) and hasattr(expected_type, "__annotations__"):
                type_hints = self._get_cached_type_hints(expected_type)

                # Check if this is a Pydantic model with field aliases
                is_pydantic = hasattr(expected_type, "__fields__") or hasattr(expected_type,
                                                                               "model_fields")
                field_aliases = {}

                # Get aliases mapping if it's a Pydantic model
                if is_pydantic:
                    # Support both Pydantic v1 and v2
                    if hasattr(expected_type, "__fields__"):  # Pydantic v1
                        for field_name, field_info in expected_type.__fields__.items():
                            if field_info.alias != field_name:
                                field_aliases[field_name] = field_info.alias
                    elif hasattr(expected_type, "model_fields"):  # Pydantic v2
                        for field_name, field_info in expected_type.model_fields.items():
                            if field_info.alias != field_name:
                                field_aliases[field_name] = field_info.alias

                # Validate each field defined in the class against the dictionary
                for field_name, field_type in type_hints.items():
                    # For Pydantic models, check if value exists by either field name or alias
                    field_alias = field_aliases.get(field_name)
                    field_present = field_name in value or (field_alias and field_alias in value)

                    if field_present:
                        # Field present in dict (by name or alias), validate its value recursively
                        field_value = value.get(field_name, value.get(field_alias))
                        self.validate_value(field_value, field_type,
                                            f"{error_context_path}.{field_name}")
                    else:
                        # Field defined in class but missing in dict. Check if Optional.
                        field_origin = get_origin(field_type)
                        field_args = get_args(field_type)
                        is_optional = (field_origin is Union and field_args and _is_none_type(
                            field_args[-1]))
                        if not is_optional:
                            # Field is required but missing
                            if field_alias:
                                raise ValueTypeError(
                                    f"Path '{error_context_path}': dictionary is missing required field '{field_name}' "
                                    f"(alias: '{field_alias}') for expected type {expected_type.__name__}."
                                )
                            else:
                                raise ValueTypeError(
                                    f"Path '{error_context_path}': dictionary is missing required field '{field_name}' "
                                    f"for expected type {expected_type.__name__}."
                                )
                        # If optional and missing, it's valid.

                # Optional: Check for extra keys in the dict not defined in the class annotations?
                # for key in value:
                #     if key not in type_hints:
                #         warnings.warn(f"Dictionary for field '{error_context_path}' has extra key '{key}' not defined in {expected_type.__name__}.")

                return  # Dictionary structure validated successfully

            # Case 7b: Value is an instance of the expected class (or subclass).
            if isinstance(value, expected_type):
                # Value is already the correct class instance. Validation passed.
                # Optional: Could recursively validate instance attributes here if desired.
                return

            # Case 7c: Value is neither a compatible dict nor an instance.
            raise ValueTypeError(
                f"Path '{error_context_path}': expected type {expected_type.__name__} " # Uses __name__ - OK
                f"or compatible dict, got {repr(value)} (type: {type(value).__name__})."
            )

        # 8. Handle basic types (int, str, float, bool, etc.) if not handled above
        # This also covers cases where expected_type is a TypeVar bound to a basic type.
        if expected_type is int and isinstance(value, bool):
            # Special case: bool is a subclass of int in Python, but for validation
            # purposes we want to treat them as distinct types
            raise ValueTypeError(
                f"Path '{error_context_path}': expected int, got bool ({repr(value)})."
            )
        elif not isinstance(value, expected_type):
            if value is None:
                # Special case for None to get consistent error message format
                raise ValueTypeError(
                    f"Path '{error_context_path}': received None but expected {expected_type.__name__}." # MODIFIED: Use __name__
                )
            else:
                raise ValueTypeError(
                    f"Path '{error_context_path}': expected type {expected_type.__name__}, " # MODIFIED: Use __name__
                    f"got {repr(value)} (type: {type(value).__name__})."
                )

    def validate_value_for_path(self, field_path: str, value: Any) -> None:
        """
        Validates that a value is compatible with the type expected at a specific field path.

        Args:
            field_path: The dot-separated path within the model.
            value: The value to validate.

        Raises:
            InvalidPathError: If the path is invalid for the model.
            ValueTypeError: If the value's type is incompatible with the field's type.
        """
        expected_type = self.get_field_type(field_path)
        self.validate_value(value, expected_type, field_path)

    def _is_single_type_numeric(self, single_type: Type) -> bool:
        """Helper to check if a non-Union type is numeric."""
        # Exclude Any before isclass check
        if single_type is Any:
            return False
        # Ensure it's a class, subclass of int/float, and importantly, not bool
        return isclass(single_type) and issubclass(single_type,
                                                   (int, float)) and single_type is not bool

    def is_field_numeric(self, field_path: str) -> bool:
        """
        Checks if the field at the given path is a numeric type (int, float,
        or an Optional/Union containing them, excluding bool).

        Args:
            field_path: The dot-separated path.

        Returns:
            True if the field type is considered numeric, False otherwise.
        """
        try:
            field_type = self.get_field_type(field_path)
        except InvalidPathError:
            return False  # Path doesn't exist, therefore not numeric

        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is Union:
            # Check if any non-None part of the Union is numeric
            return any(self._is_single_type_numeric(arg) for arg in args if not _is_none_type(arg))

        # Check if the base type itself is numeric
        return self._is_single_type_numeric(field_type)

    def get_list_item_type(self, field_path: str) -> Tuple[bool, Type]:
        """
        Checks if field is a List or Optional[List] and returns its item type.

        Args:
            field_path: The dot-separated path.

        Returns:
            A tuple: (is_list_type, item_type).
            'is_list_type' is True if the field type is List or Optional[List].
            'item_type' is the list's item type (e.g., int for List[int]), or Any
            if the list type is unspecified (e.g., `list`) or if the field is not a list type.

        Raises:
            InvalidPathError: If the path itself is invalid (e.g., parent doesn't exist).
        """
        # get_field_type will raise InvalidPathError if path is fundamentally wrong
        field_type = self.get_field_type(field_path)
        origin = get_origin(field_type)
        args = get_args(field_type)

        actual_list_type = None
        is_list_type = False
        item_type = Any

        # Check if it's a generic List[T] type
        if origin is list:
            is_list_type = True
            actual_list_type = field_type
        # Check if it's a plain 'list' type (no generics)
        elif field_type is list:
            is_list_type = True
            # For plain list, item_type remains Any
        # Check for Optional[List[...]]
        elif origin is Union:
            maybe_list_arg = None
            has_none = False
            for arg in args:
                if _is_none_type(arg):
                    has_none = True
                elif get_origin(arg) is list or arg is list:
                    maybe_list_arg = arg
            if has_none and maybe_list_arg:
                is_list_type = True
                actual_list_type = maybe_list_arg

        # Extract item type if we confirmed it's a list type with generics
        if actual_list_type:
            list_args = get_args(actual_list_type)
            if list_args:  # If it's like List[T]
                item_type = list_args[0]
            # else: it's a plain `list`, item_type remains Any

        return is_list_type, item_type