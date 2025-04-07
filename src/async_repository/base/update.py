import logging
from typing import Any, Dict, Type, get_type_hints, get_origin, get_args, Union
import inspect
from inspect import isclass


class Update:
    """
    A builder for update instructions. This class provides a fluent interface
    to build update operations such as set, push, pop, unset, and pull.

    Example usage:

        update = (Update()
                  .set("name", "New Name")
                  .push("tags", "new_tag")
                  .unset("obsolete_field")
                  .pull("items", "remove_this"))

        payload = update.build()
        # payload will be:
        # {
        #    "$set": {"name": "New Name"},
        #    "$push": {"tags": "new_tag"},
        #    "$unset": {"obsolete_field": ""},
        #    "$pull": {"items": "remove_this"}
        # }

    If a type is provided, all operations will validate field values against that type.
    """

    def __init__(self, model_type: Type = None) -> None:
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._model_type = model_type

    def _validate_field(self, field: str, value: Any) -> None:
        """
        Validates that the field exists in the model and the value matches the expected type.
        Performs recursive validation for nested fields and container types.
        """
        if self._model_type is None:
            return

        # Handle nested fields (e.g., "user.address.street")
        field_parts = field.split(".")
        current_type = self._model_type
        current_field_path = []

        for i, part in enumerate(field_parts):
            current_field_path.append(part)
            current_field = ".".join(current_field_path)

            if not hasattr(current_type, "__annotations__"):
                raise TypeError(
                    f"Cannot validate nested field '{current_field}': parent is not a class with annotations"
                )

            type_hints = get_type_hints(current_type)
            if part not in type_hints:
                raise TypeError(
                    f"Field '{current_field}' does not exist in {current_type.__name__}"
                )

            expected_type = type_hints[part]

            # If this is the last part, validate the value
            if i == len(field_parts) - 1:
                self._validate_value(current_field, value, expected_type)
            else:
                # For nested fields, get the type of the next level
                origin = get_origin(expected_type)
                if origin is dict:
                    raise TypeError(
                        f"Cannot validate nested field '{current_field}': Dict type cannot be used for nested field validation"
                    )
                current_type = expected_type

    def _validate_value(self, field: str, value: Any, expected_type: Type) -> None:
        """
        Recursively validates that the value matches the expected type.
        Handles container types like List, Dict, etc., and now also allows dicts for custom types,
        as well as recursively validates model instances.
        """
        # Handle None for Optional types
        if value is None:
            origin = get_origin(expected_type)
            if origin is not None and origin is Union:
                args = get_args(expected_type)
                if type(None) in args:
                    return
            raise TypeError(f"Field '{field}' cannot be None, expected {expected_type}")

        # Check for generic container types
        origin = get_origin(expected_type)
        if origin is not None:
            if origin in (list, set, tuple):
                if not isinstance(value, origin):
                    raise TypeError(
                        f"Field '{field}' expected {origin.__name__}, got {type(value).__name__}"
                    )
                args = get_args(expected_type)
                if args and args[0] != Any:
                    item_type = args[0]
                    for i, item in enumerate(value):
                        self._validate_value(f"{field}[{i}]", item, item_type)
            elif origin is dict:
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Field '{field}' expected dict, got {type(value).__name__}"
                    )
                args = get_args(expected_type)
                if len(args) == 2 and args[1] != Any:
                    key_type, value_type = args
                    for k, v in value.items():
                        if key_type != Any:
                            self._validate_value(f"{field} key", k, key_type)
                        if value_type != Any:
                            self._validate_value(f"{field}[{k}]", v, value_type)
            elif origin is Union:
                args = get_args(expected_type)
                valid = False
                for arg_type in args:
                    try:
                        self._validate_value(field, value, arg_type)
                        valid = True
                        break
                    except TypeError:
                        continue
                if not valid:
                    raise TypeError(
                        f"Field '{field}', type: {type(value)} and value {value} does not match any type in {expected_type}"
                    )
            return  # After processing a generic container or union, we can return

        # Regular type checking for non-generic types
        elif isclass(expected_type):
            # Allow a dict for a model type (if the class defines annotations)
            if isinstance(value, dict) and hasattr(expected_type, "__annotations__"):
                return
            # If the value is an instance of the expected type, recursively validate its attributes.
            if isinstance(value, expected_type):
                if hasattr(expected_type, "__annotations__"):
                    for attr, attr_type in get_type_hints(expected_type).items():
                        if hasattr(value, attr):
                            self._validate_value(
                                f"{field}.{attr}", getattr(value, attr), attr_type
                            )
                return
            raise TypeError(
                f"Field '{field}' expected {expected_type.__name__}, got {type(value).__name__}"
            )

    def set(self, field: str, value: Any) -> "Update":
        self._validate_field(field, value)
        self._operations.setdefault("set", {})[field] = value
        return self

    def push(self, field: str, value: Any) -> "Update":
        # For push, we need to validate that the field is a list and the value matches the list's item type
        if self._model_type:
            field_parts = field.split(".")
            current_type = self._model_type

            for i, part in enumerate(field_parts[:-1]):
                type_hints = get_type_hints(current_type)
                if part not in type_hints:
                    raise TypeError(
                        f"Field '{'.'.join(field_parts[:i + 1])}' does not exist in {current_type.__name__}"
                    )
                current_type = type_hints[part]

            # Get the type of the final field
            last_part = field_parts[-1]
            type_hints = get_type_hints(current_type)
            if last_part not in type_hints:
                raise TypeError(
                    f"Field '{field}' does not exist in {current_type.__name__}"
                )

            field_type = type_hints[last_part]
            origin = get_origin(field_type)

            # Ensure the field is a list type
            if origin is not list:
                raise TypeError(f"Cannot push to field '{field}' which is not a list")

            # Validate the value against the list's item type
            args = get_args(field_type)
            if args and args[0] != Any:
                self._validate_value(f"{field} item", value, args[0])

        self._operations.setdefault("push", {})[field] = value
        return self

    def pop(self, field: str, direction: int = 1) -> "Update":
        """
        Pop an element from a list field.
        direction: 1 pops the last element, -1 pops the first.
        """
        # Verify the field is a list type
        if self._model_type:
            field_parts = field.split(".")
            current_type = self._model_type

            for i, part in enumerate(field_parts[:-1]):
                type_hints = get_type_hints(current_type)
                if part not in type_hints:
                    raise TypeError(
                        f"Field '{'.'.join(field_parts[:i + 1])}' does not exist in {current_type.__name__}"
                    )
                current_type = type_hints[part]

            # Get the type of the final field
            last_part = field_parts[-1]
            type_hints = get_type_hints(current_type)
            if last_part not in type_hints:
                raise TypeError(
                    f"Field '{field}' does not exist in {current_type.__name__}"
                )

            field_type = type_hints[last_part]
            origin = get_origin(field_type)

            # Ensure the field is a list type
            if origin is not list:
                raise TypeError(f"Cannot pop from field '{field}' which is not a list")

            # Validate direction is either 1 or -1
            if direction not in (1, -1):
                raise ValueError(
                    "Direction must be either 1 (last element) or -1 (first element)"
                )

        self._operations.setdefault("pop", {})[field] = direction
        return self

    def unset(self, field: str) -> "Update":
        # Verify the field exists in the model
        if self._model_type:
            field_parts = field.split(".")
            current_type = self._model_type

            for i, part in enumerate(field_parts[:-1]):
                type_hints = get_type_hints(current_type)
                if part not in type_hints:
                    raise TypeError(
                        f"Field '{'.'.join(field_parts[:i + 1])}' does not exist in {current_type.__name__}"
                    )
                current_type = type_hints[part]

            # Check if the final field exists
            last_part = field_parts[-1]
            type_hints = get_type_hints(current_type)
            if last_part not in type_hints:
                raise TypeError(
                    f"Field '{field}' does not exist in {current_type.__name__}"
                )

        # In MongoDB, the $unset operator ignores the value; here we set it to an empty string.
        self._operations.setdefault("unset", {})[field] = ""
        return self

    def pull(self, field: str, value: Any) -> "Update":
        # For pull, we need to validate that the field is a list and the value matches the list's item type
        if self._model_type:
            field_parts = field.split(".")
            current_type = self._model_type

            for i, part in enumerate(field_parts[:-1]):
                type_hints = get_type_hints(current_type)
                if part not in type_hints:
                    raise TypeError(
                        f"Field '{'.'.join(field_parts[:i + 1])}' does not exist in {current_type.__name__}"
                    )
                current_type = type_hints[part]

            # Get the type of the final field
            last_part = field_parts[-1]
            type_hints = get_type_hints(current_type)
            if last_part not in type_hints:
                raise TypeError(
                    f"Field '{field}' does not exist in {current_type.__name__}"
                )

            field_type = type_hints[last_part]
            origin = get_origin(field_type)

            # Ensure the field is a list type
            if origin is not list:
                raise TypeError(f"Cannot pull from field '{field}' which is not a list")

            # Validate the value against the list's item type
            args = get_args(field_type)
            if args and args[0] != Any:
                self._validate_value(f"{field} item", value, args[0])

        self._operations.setdefault("pull", {})[field] = value
        return self

    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively converts Pydantic models to dictionaries using model_dump(mode="json", by_alias=True)
        if the method exists.
        """
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump(mode="json", by_alias=True)
            except Exception:
                return value.model_dump(by_alias=True)
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    def build(self) -> Dict[str, Any]:
        """
        Builds and returns the update payload.
        The payload maps internal operation keys to the corresponding update operators.
        All values are recursively processed to serialize any Pydantic models.
        """
        mapping = {
            "set": "$set",
            "push": "$push",
            "pop": "$pop",
            "unset": "$unset",
            "pull": "$pull",
        }
        payload = {}
        for op, data in self._operations.items():
            # Recursively convert any Pydantic model instances to dicts.
            serialized_data = {
                field: self._serialize_value(val) for field, val in data.items()
            }
            payload[mapping[op]] = serialized_data
        return payload

    def __repr__(self) -> str:
        """
        Returns a string representation of the Update object that shows
        all the configured operations in a readable format.
        """
        if not self._operations:
            return f"Update({self._model_type.__name__ if self._model_type else ''})"

        parts = [f"Update({self._model_type.__name__ if self._model_type else ''})"]
        for op, fields in self._operations.items():
            for field, value in fields.items():
                if isinstance(value, str):
                    formatted_value = f'"{value}"'
                else:
                    formatted_value = repr(value)
                parts.append(f".{op}({repr(field)}, {formatted_value})")
        return "".join(parts)
