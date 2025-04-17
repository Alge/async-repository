# -*- coding: utf-8 -*-
"""
Provides the Update class for building MongoDB-style update operations with optional type validation.
"""
import warnings  # Used for deprecation warning
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


# Helper function to check if a value is None
# Needed because `type(None)` is `NoneType`, which is not directly comparable in some contexts.
def _is_none_type(t: Type) -> bool:
    return t is type(None)  # pylint: disable=unidiomatic-typecheck


class Update:
    """
    A builder for MongoDB-style update instructions with type validation.

    This class provides a fluent interface to construct update payloads
    for operations like $set, $push, $pop, $unset, $pull, $inc, $min, $max, and $mul.
    If initialized with a model type (e.g., a Pydantic model or a class with type hints),
    it validates that fields exist and that the values provided match the expected types,
    including handling nested fields and list indices.

    Example usage:

        # Without type validation
        update_no_type = (Update()
                          .set("name", "New Name")
                          .push("tags", "new_tag")
                          .unset("obsolete_field")
                          .pull("items", "remove_this"))
        payload_no_type = update_no_type.build()
        # payload_no_type:
        # {
        #    "$set": {"name": "New Name"},
        #    "$push": {"tags": "new_tag"},
        #    "$unset": {"obsolete_field": ""},
        #    "$pull": {"items": "remove_this"}
        # }


        # With type validation (assuming a User class exists)
        from typing import List, Optional
        class User:
            name: str
            age: int
            tags: List[str]
            score: Optional[float]

        update_with_type = (Update(User)
                            .set("name", "Validated Name")
                            .push("tags", "validated_tag")
                            .set("score", 95.5)
                            # .set("age", "thirty") # Would raise TypeError
                           )
        payload_with_type = update_with_type.build()
        # payload_with_type:
        # {
        #     "$set": {"name": "Validated Name", "score": 95.5},
        #     "$push": {"tags": "validated_tag"}
        # }

    Attributes:
        _operations (Dict[str, Dict[str, Any]]): Stores the update operations.
        _model_type (Optional[Type]): The model type used for validation, if provided.
    """

    def __init__(self, model_type: Optional[Type] = None) -> None:
        """
        Initializes the Update builder.

        Args:
            model_type: An optional class (e.g., Pydantic model or standard class
                        with type hints) to use for validating fields and types.
        """
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._model_type: Optional[Type] = model_type

    def _get_field_type(self, field: str) -> Type:
        """
        Traverses a potentially nested field path (e.g., "user.address.street", "items.0.name")
        and returns the expected type hint of the final field based on the _model_type.

        Args:
            field: The field path string.

        Returns:
            The type hint of the specified field.

        Raises:
            TypeError: If the field path is invalid (e.g., field doesn't exist,
                       indexing a non-list, accessing attributes on a non-class type)
                       or if _model_type is not set.
        """
        if self._model_type is None:
            return Any

        field_parts = field.split(".")
        current_type = self._model_type
        current_field_path_parts = []

        for part in field_parts:
            current_field_path_parts.append(part)
            parent_field_path = ".".join(current_field_path_parts[:-1])

            origin = get_origin(current_type)
            args = get_args(current_type)

            if part.isdigit():
                actual_list_type = None
                if origin is list:
                    actual_list_type = current_type
                elif origin is Union:
                    maybe_list_arg = None
                    has_none = False
                    for arg in args:
                        if _is_none_type(arg):
                            has_none = True
                        elif get_origin(arg) is list:
                            maybe_list_arg = arg
                    if has_none and maybe_list_arg:
                        actual_list_type = maybe_list_arg

                if actual_list_type is None:
                    raise TypeError(
                        f"Cannot apply index '{part}' to non-list field "
                        f"'{parent_field_path}' (type: {current_type})."
                    )

                list_args = get_args(actual_list_type)
                current_type = list_args[0] if list_args else Any
                continue

            if origin is Union:
                raise TypeError(
                    f"Cannot access nested field '{part}' on Union type field "
                    f"'{parent_field_path}'. Ambiguous target type."
                )

            if not isclass(current_type) or not hasattr(
                current_type, "__annotations__"
            ):
                if (
                    origin is dict
                    and args
                    and len(args) == 2
                    and args[0] is str
                ):
                    current_type = args[1]
                elif origin is Any or current_type is Any:
                    current_type = Any
                else:
                    raise TypeError(
                        f"Cannot access nested field '{part}'. Parent field "
                        f"'{parent_field_path}' is not a class with annotations or a valid Dict "
                        f"(type: {current_type})."
                    )

            if current_type is Any:
                continue

            try:
                # Resolve forward references and inheritance
                type_hints = get_type_hints(current_type, include_extras=True)
            except Exception as e:
                raise TypeError(
                    f"Could not get type hints for '{parent_field_path}' (type: {current_type}). Error: {e}"
                ) from e

            if part not in type_hints:
                raise TypeError(
                    f"Field '{part}' does not exist in type {current_type.__name__} "
                    f"(path: {''.join(current_field_path_parts)}')"
                )

            current_type = type_hints[part]

        return current_type

    def _validate_value(
        self, field_path_for_error: str, value: Any, expected_type: Type
    ) -> None:
        """
        Recursively validates that a given value matches the expected type hint.
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # 1. Handle Optional[T] (represented as Union[T, NoneType])
        if origin is Union and args and _is_none_type(args[-1]):
            if value is None:
                return  # None is valid for Optional[T]
            # Validate against the non-None type(s) in the Optional
            non_none_types = tuple(t for t in args if not _is_none_type(t))
            if len(non_none_types) == 1:
                # Reduce Optional[T] to T for further validation
                expected_type = non_none_types[0]
                origin = get_origin(expected_type)
                args = get_args(expected_type)
            else:
                # Reduce Optional[Union[A,B]] to Union[A,B]
                expected_type = Union[non_none_types]
                # Fall through to Union validation logic

        # 2. Handle Any: Allows any value (except None unless Optional[Any])
        if expected_type is Any:
            return

        # 3. Handle None value when type is not Optional (checked after Optional handling)
        if value is None:
            raise TypeError(
                f"Field '{field_path_for_error}' received None but expected type {expected_type}."
            )

        # 4. Handle Union[A, B, ...] (that are not Optional)
        if origin is Union:
            possible_types = args
            for possible_type in possible_types:
                try:
                    # Recursively validate against each type in the Union
                    self._validate_value(
                        field_path_for_error, value, possible_type
                    )
                    return  # Matched one type in the Union
                except TypeError:
                    continue  # Try the next type
            # If loop finishes, value didn't match any type in the Union
            raise TypeError(
                f"Field '{field_path_for_error}' received value {repr(value)} (type: {type(value).__name__}) "
                f"which does not match any type in Union {expected_type}."
            )

        # 5. Handle List, Set, Tuple
        if origin in (list, set, tuple):
            if not isinstance(value, origin):
                raise TypeError(
                    f"Field '{field_path_for_error}' expected type {origin.__name__} "
                    f"but received type {type(value).__name__}."
                )
            # If type has arguments (e.g., List[int]), validate items
            if args and args[0] is not Any:
                item_type = args[0]
                for i, item in enumerate(value):
                    self._validate_value(
                        f"{field_path_for_error}[{i}]", item, item_type
                    )
            return

        # 6. Handle Dict
        if origin is dict:
            if not isinstance(value, dict):
                raise TypeError(
                    f"Field '{field_path_for_error}' expected type dict "
                    f"but received type {type(value).__name__}."
                )
            # If type has arguments (e.g., Dict[str, int]), validate keys and values
            if (
                len(args) == 2
                and (args[0] is not Any or args[1] is not Any)
            ):
                key_type, value_type = args
                for k, v in value.items():
                    if key_type is not Any:
                        self._validate_value(
                            f"{field_path_for_error} key '{k}'", k, key_type
                        )
                    if value_type is not Any:
                        self._validate_value(
                            f"{field_path_for_error}['{k}']", v, value_type
                        )
            return

        # 7. Handle Specific Classes (Pydantic models, standard classes with hints)
        if isclass(expected_type):
            # Case 7a: Value is a dictionary, expected type is a class.
            # Validate dictionary contents against the class annotations.
            if isinstance(value, dict) and hasattr(
                expected_type, "__annotations__"
            ):
                try:
                    # Resolve forward refs, include extras for thoroughness
                    type_hints = get_type_hints(expected_type, include_extras=True)
                except Exception as e:
                    # If hints aren't available, precise validation isn't possible.
                    raise TypeError(
                        f"Could not get type hints for expected class {expected_type.__name__} "
                        f"to validate dictionary value at '{field_path_for_error}'. Error: {e}"
                    ) from e

                # Validate each field defined in the class against the dictionary
                for field_name, field_type in type_hints.items():
                    if field_name in value:
                        # Field present in dict, validate its value
                        field_value = value[field_name]
                        self._validate_value(
                            f"{field_path_for_error}.{field_name}",
                            field_value,
                            field_type,
                        )
                    else:
                        # Field defined in class but missing in dict. Check if Optional.
                        field_origin = get_origin(field_type)
                        field_args = get_args(field_type)
                        is_optional = (
                            field_origin is Union
                            and field_args
                            and _is_none_type(field_args[-1])
                        )
                        if not is_optional:
                            # Field is required but missing
                            raise TypeError(
                                f"Dictionary provided for field '{field_path_for_error}' "
                                f"(expected type {expected_type.__name__}) is missing required field '{field_name}'."
                            )
                        # If optional and missing, it's valid.

                # Optional: Check for extra keys in the dict not defined in the class annotations.
                # for key in value:
                #     if key not in type_hints:
                #         warnings.warn(f"Dictionary for field '{field_path_for_error}' has extra key '{key}' not defined in {expected_type.__name__}.")

                return  # Dictionary structure validated successfully

            # Case 7b: Value is an instance of the expected class (or subclass).
            if isinstance(value, expected_type):
                # Value is already the correct class instance. Validation passed.
                # Optional: Could recursively validate instance attributes here if desired.
                return

            # Case 7c: Value is neither a compatible dict nor an instance.
            raise TypeError(
                f"Field '{field_path_for_error}' expected type {expected_type.__name__} "
                f"or a compatible dict, but received value {repr(value)} (type: {type(value).__name__})."
            )

        # 8. Handle basic types (int, str, float, bool, etc.) if not handled above
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Field '{field_path_for_error}' expected type {expected_type.__name__} "
                f"but received value {repr(value)} (type: {type(value).__name__})."
            )

    def _is_numeric_type(self, type_hint: Type) -> bool:
        """
        Checks if a type hint represents a numeric type (int, float)
        or an Optional/Union containing numeric types.
        """
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union:
            return any(
                self._is_numeric_type(arg)
                for arg in args
                if not _is_none_type(arg)
            )

        return (
            isclass(type_hint)
            and issubclass(type_hint, (int, float))
            and type_hint is not bool
        )

    def _validate_list_operation(self, field: str, operation: str) -> Type:
        """
        Helper to validate the field type for list operations (push, pop, pull).
        Ensures the target field is List[T] or Optional[List[T]]. Returns item type T.
        """
        if not self._model_type:
            return Any

        field_type = self._get_field_type(field)
        origin = get_origin(field_type)
        args = get_args(field_type)

        actual_list_type = None
        item_type = Any

        if origin is list:
            actual_list_type = field_type
        elif origin is Union:
            maybe_list_arg = None
            has_none = False
            for arg in args:
                if _is_none_type(arg):
                    has_none = True
                elif get_origin(arg) is list:
                    maybe_list_arg = arg
            if has_none and maybe_list_arg:
                actual_list_type = maybe_list_arg

        if actual_list_type is None:
            raise TypeError(
                f"Cannot perform ${operation} on field '{field}'. "
                f"Expected List or Optional[List], but got {field_type}."
            )

        list_args = get_args(actual_list_type)
        if list_args:
            item_type = list_args[0]

        return item_type

    def _validate_numeric_operation(self, field: str, operation: str):
        """
        Helper to validate the field type for numeric operations.
        """
        if not self._model_type:
            return

        field_type = self._get_field_type(field)
        if not self._is_numeric_type(field_type):
            raise TypeError(
                f"Cannot perform ${operation} on field '{field}'. "
                f"Expected a numeric type (int, float, Optional/Union thereof), but got {field_type}."
            )

    def set(self, field: str, value: Any) -> "Update":
        """
        Sets the value of a field ($set).
        """
        if self._model_type:
            field_type = self._get_field_type(field)
            self._validate_value(field, value, field_type)

        self._operations.setdefault("set", {})[field] = value
        return self

    def push(self, field: str, value: Any) -> "Update":
        """
        Adds an element to an array field ($push).
        """
        if self._model_type:
            item_type = self._validate_list_operation(field, "push")
            if item_type is not Any:
                self._validate_value(f"{field} item", value, item_type)

        self._operations.setdefault("push", {})[field] = value
        return self

    def pop(self, field: str, direction: int = 1) -> "Update":
        """
        Removes the first (-1) or last (1) element of an array field ($pop).
        """
        if direction not in (1, -1):
            raise ValueError(
                f"Direction for pop operation must be 1 (last) or -1 (first), got {direction}."
            )

        if self._model_type:
            _ = self._validate_list_operation(field, "pop")

        self._operations.setdefault("pop", {})[field] = direction
        return self

    def unset(self, field: str) -> "Update":
        """
        Removes a field ($unset).
        """
        if self._model_type:
            _ = self._get_field_type(field)

        self._operations.setdefault("unset", {})[field] = ""
        return self

    def pull(self, field: str, value: Any) -> "Update":
        """
        Removes instances of a value or condition from an array field ($pull).
        """
        if self._model_type:
            item_type = self._validate_list_operation(field, "pull")
            # Check if the value looks like a MongoDB query operator dict
            is_operator_dict = isinstance(value, dict) and any(
                k.startswith("$") for k in value.keys()
            )

            # Validate the value against the item type unless it's Any
            # or it's a dictionary that looks like a query operator.
            if item_type is not Any and not is_operator_dict:
                self._validate_value(f"{field} item", value, item_type)

        self._operations.setdefault("pull", {})[field] = value
        return self

    def increment(self, field: str, amount: Union[int, float] = 1) -> "Update":
        """
        Increments a numeric field ($inc).
        """
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Increment amount must be numeric, got {type(amount).__name__}."
            )

        if self._model_type:
            self._validate_numeric_operation(field, "inc")

        current_inc = self._operations.setdefault("inc", {})
        current_inc[field] = current_inc.get(field, 0) + amount
        return self

    def decrement(self, field: str, amount: Union[int, float] = 1) -> "Update":
        """
        Decrements a numeric field (using $inc with negative amount).
        """
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Decrement amount must be numeric, got {type(amount).__name__}."
            )
        return self.increment(field, -amount)

    def min(self, field: str, value: Union[int, float]) -> "Update":
        """
        Sets field to `value` if `value` is less than current field value ($min).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$min requires a numeric value, got {type(value).__name__}."
            )

        if self._model_type:
            self._validate_numeric_operation(field, "min")

        self._operations.setdefault("min", {})[field] = value
        return self

    def max(self, field: str, value: Union[int, float]) -> "Update":
        """
        Sets field to `value` if `value` is greater than current field value ($max).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$max requires a numeric value, got {type(value).__name__}."
            )

        if self._model_type:
            self._validate_numeric_operation(field, "max")

        self._operations.setdefault("max", {})[field] = value
        return self

    def mul(self, field: str, value: Union[int, float]) -> "Update":
        """
        Multiplies numeric field by `value` ($mul).
        """
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$mul requires a numeric value, got {type(value).__name__}."
            )

        if self._model_type:
            self._validate_numeric_operation(field, "mul")

        self._operations.setdefault("mul", {})[field] = value
        return self

    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively serializes values, converting known models (like Pydantic) to dicts.
        """
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try:
                return value.model_dump(mode="json", by_alias=True)
            except TypeError:
                try:
                    return value.model_dump(by_alias=True)
                except TypeError:
                    try:
                        return value.model_dump()
                    except Exception:
                        return value
        elif hasattr(value, "dict") and callable(value.dict):
            try:
                return value.dict(by_alias=True)
            except TypeError:
                try:
                    return value.dict()
                except Exception:
                    return value
        elif isinstance(value, dict):
            return {
                self._serialize_value(k): self._serialize_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple, set)):
            return [self._serialize_value(item) for item in value]
        return value

    def build(self) -> Dict[str, Any]:
        """
        Constructs the final MongoDB update payload dictionary.
        """
        mongo_operator_map = {
            "set": "$set",
            "push": "$push",
            "pop": "$pop",
            "unset": "$unset",
            "pull": "$pull",
            "inc": "$inc",
            "min": "$min",
            "max": "$max",
            "mul": "$mul",
        }
        payload: Dict[str, Any] = {}

        for internal_op, data in self._operations.items():
            mongo_op = mongo_operator_map.get(internal_op)
            if not mongo_op:
                warnings.warn(
                    f"Unknown internal update operation '{internal_op}' encountered during build."
                )
                continue

            # Serialize values; do NOT filter $inc: 0 here anymore
            serialized_data = {
                field: self._serialize_value(val)
                for field, val in data.items()
            }

            if serialized_data:  # Ensure we don't add empty operator blocks
                payload[mongo_op] = serialized_data

        return payload

    def __repr__(self) -> str:
        """
        Provides a developer-friendly string representation of the Update object.
        """
        model_name = f"({self._model_type.__name__})" if self._model_type else ""
        if not self._operations:
            return f"Update{model_name}"

        final_parts = [f"Update{model_name}"]
        # Represent the *final* state of operations, as built
        # Note: build() no longer filters $inc:0, so __repr__ needs to handle it if desired.
        final_ops = self.build()
        sorted_ops = sorted(final_ops.items())

        method_map = {
            "$set": "set",
            "$push": "push",
            "$pop": "pop",
            "$unset": "unset",
            "$pull": "pull",
            "$inc": "increment",
            "$min": "min",
            "$max": "max",
            "$mul": "mul",
        }

        for mongo_op, fields_data in sorted_ops:
            op = method_map.get(mongo_op, mongo_op)
            sorted_fields = sorted(fields_data.items())
            for field, value in sorted_fields:
                value_repr = repr(value)
                if op == "pop":
                    final_parts.append(
                        f".{op}({repr(field)}, direction={value_repr})"
                    )
                elif op == "unset":
                    final_parts.append(f".unset({repr(field)})")
                elif mongo_op == "$inc":
                    # Represent the final $inc value, including 0 if present
                    if value == 1:
                        final_parts.append(f".increment({repr(field)})")
                    elif value == -1:
                        final_parts.append(f".decrement({repr(field)})")
                    elif value > 0:
                        final_parts.append(
                            f".increment({repr(field)}, amount={value_repr})"
                        )
                    elif value < 0:
                        final_parts.append(
                            f".decrement({repr(field)}, amount={repr(abs(value))})"
                        )
                    else: # value == 0
                        final_parts.append(
                             f".increment({repr(field)}, amount=0)" # Explicitly show inc 0
                        )
                elif op in ("set", "push", "pull", "min", "max", "mul"):
                    final_parts.append(f".{op}({repr(field)}, {value_repr})")

        return "".join(final_parts)

    def __bool__(self) -> bool:
        """Returns True if any update operations have been added."""
        return bool(self._operations)

    def __len__(self) -> int:
        """Returns the total number of fields being modified."""
        return sum(len(data) for data in self._operations.values())