# -*- coding: utf-8 -*-
"""
Provides the Update class for building MongoDB-style update operations with optional type validation.
"""
import warnings  # Used for deprecation warning
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    Union,
)

from async_repository.base.model_validator import ValueTypeError, InvalidPathError, ModelValidator

class Update:
    """
    A builder for MongoDB-style update instructions. Uses ModelValidator for optional type validation.
    """
    def __init__(self, model_type: Optional[Type] = None) -> None:
        """Initializes the Update builder."""
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._model_type: Optional[Type] = model_type
        self._validator: Optional[ModelValidator] = None
        if model_type:
            try:
                self._validator = ModelValidator(model_type)
            except TypeError as e:
                warnings.warn(f"Failed to initialize ModelValidator for type {model_type}: {e}. Validation disabled.")
                self._model_type = None

    def set(self, field: str, value: Any) -> "Update":
        """Sets the value of a field ($set)."""
        if self._validator:
            self._validator.validate_value_for_path(field, value)
        self._operations.setdefault("set", {})[field] = value
        return self

    def push(self, field: str, value: Any) -> "Update":
        """Adds an element to an array field ($push)."""
        if self._validator:
            is_list, item_type = self._validator.get_list_item_type(field)
            if not is_list:
                 raise InvalidPathError(f"Cannot push to field '{field}': not a List or Optional[List].")
            if item_type is not Any:
                 try:
                    self._validator.validate_value(value, item_type, f"{field} item")
                 except ValueTypeError as e:
                     raise ValueTypeError(f"Invalid value for push to '{field}': {e}") from e
        self._operations.setdefault("push", {})[field] = value
        return self

    def pop(self, field: str, direction: int = 1) -> "Update":
        """Removes the first (-1) or last (1) element of an array field ($pop)."""
        if direction not in (1, -1):
            raise ValueError(f"Direction for pop must be 1 (last) or -1 (first), got {direction}.")
        if self._validator:
            is_list, _ = self._validator.get_list_item_type(field)
            if not is_list:
                 raise InvalidPathError(f"Cannot pop from field '{field}': not a List or Optional[List].")
        self._operations.setdefault("pop", {})[field] = direction
        return self

    def unset(self, field: str) -> "Update":
        """Removes a field ($unset)."""
        if self._validator:
            self._validator.get_field_type(field)
        self._operations.setdefault("unset", {})[field] = ""
        return self

    def pull(self, field: str, value: Any) -> "Update":
        """Removes instances of a value or condition from an array field ($pull)."""
        if self._validator:
            is_list, item_type = self._validator.get_list_item_type(field)
            if not is_list:
                 raise InvalidPathError(f"Cannot pull from field '{field}': not a List or Optional[List].")
            is_operator_dict = isinstance(value, dict) and any(k.startswith("$") for k in value.keys())
            if item_type is not Any and not is_operator_dict:
                 try:
                     self._validator.validate_value(value, item_type, f"{field} item")
                 except ValueTypeError as e:
                     raise ValueTypeError(f"Invalid value for pull from '{field}': {e}") from e
        self._operations.setdefault("pull", {})[field] = value
        return self

    def increment(self, field: str, amount: Union[int, float] = 1) -> "Update":
        """Increments a numeric field ($inc)."""
        if not isinstance(amount, (int, float)):
            raise TypeError(f"Increment amount must be numeric, got {type(amount).__name__}.")
        if self._validator:
            if not self._validator.is_field_numeric(field):
                 raise ValueTypeError(f"Cannot increment non-numeric field '{field}'.")
        current_inc = self._operations.setdefault("inc", {})
        current_inc[field] = current_inc.get(field, 0) + amount
        return self

    def decrement(self, field: str, amount: Union[int, float] = 1) -> "Update":
        """Decrements a numeric field (using $inc with negative amount)."""
        if not isinstance(amount, (int, float)):
            raise TypeError(f"Decrement amount must be numeric, got {type(amount).__name__}.")
        return self.increment(field, -amount)

    def min(self, field: str, value: Union[int, float]) -> "Update":
        """Sets field to `value` if `value` is less than current field value ($min)."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"$min requires a numeric value, got {type(value).__name__}.")
        if self._validator:
            if not self._validator.is_field_numeric(field):
                 raise ValueTypeError(f"Cannot apply $min to non-numeric field '{field}'.")
        self._operations.setdefault("min", {})[field] = value
        return self

    def max(self, field: str, value: Union[int, float]) -> "Update":
        """Sets field to `value` if `value` is greater than current field value ($max)."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"$max requires a numeric value, got {type(value).__name__}.")
        if self._validator:
            if not self._validator.is_field_numeric(field):
                 raise ValueTypeError(f"Cannot apply $max to non-numeric field '{field}'.")
        self._operations.setdefault("max", {})[field] = value
        return self

    def mul(self, field: str, value: Union[int, float]) -> "Update":
        """Multiplies numeric field by `value` ($mul)."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"$mul requires a numeric value, got {type(value).__name__}.")
        if self._validator:
            if not self._validator.is_field_numeric(field):
                 raise ValueTypeError(f"Cannot apply $mul to non-numeric field '{field}'.")
        self._operations.setdefault("mul", {})[field] = value
        return self

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serializes values, converting known models (like Pydantic) to dicts."""
        if hasattr(value, "model_dump") and callable(value.model_dump):
            try: return value.model_dump(mode="json", by_alias=True)
            except TypeError:
                try: return value.model_dump(by_alias=True)
                except TypeError:
                    try: return value.model_dump()
                    except Exception: return value
        elif hasattr(value, "dict") and callable(value.dict):
            try: return value.dict(by_alias=True)
            except TypeError:
                try: return value.dict()
                except Exception: return value
        elif isinstance(value, dict):
            return { self._serialize_value(k): self._serialize_value(v) for k, v in value.items() }
        elif isinstance(value, (list, tuple, set)):
            return [self._serialize_value(item) for item in value]
        return value

    def build(self) -> Dict[str, Any]:
        """Constructs the final MongoDB update payload dictionary."""
        mongo_operator_map = {
            "set": "$set", "push": "$push", "pop": "$pop", "unset": "$unset",
            "pull": "$pull", "inc": "$inc", "min": "$min", "max": "$max", "mul": "$mul",
        }
        payload: Dict[str, Any] = {}
        current_operations = self._operations

        for internal_op, data in current_operations.items():
            mongo_op = mongo_operator_map.get(internal_op)
            if not mongo_op:
                warnings.warn(f"Unknown internal op '{internal_op}' during build.")
                continue
            serialized_data = { field: self._serialize_value(val) for field, val in data.items() }
            if serialized_data:
                payload[mongo_op] = serialized_data
        return payload

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the Update object."""
        model_name = f"({self._model_type.__name__})" if self._model_type else ""
        if not self._operations: return f"Update{model_name}"
        final_parts = [f"Update{model_name}"]
        final_ops = self.build()
        if not final_ops: return f"Update{model_name}"
        sorted_ops = sorted(final_ops.items())
        method_map = {
            "$set": "set", "$push": "push", "$pop": "pop", "$unset": "unset",
            "$pull": "pull", "$inc": "increment", "$min": "min", "$max": "max", "$mul": "mul",
        }
        for mongo_op, fields_data in sorted_ops:
            op = method_map.get(mongo_op, mongo_op)
            sorted_fields = sorted(fields_data.items())
            for field, value in sorted_fields:
                value_repr = repr(value)
                if op == "pop": final_parts.append(f".{op}({repr(field)}, direction={value_repr})")
                elif op == "unset": final_parts.append(f".unset({repr(field)})")
                elif mongo_op == "$inc":
                    if value == 1: final_parts.append(f".increment({repr(field)})")
                    elif value == -1: final_parts.append(f".decrement({repr(field)})")
                    elif value > 0 : final_parts.append(f".increment({repr(field)}, amount={value_repr})")
                    elif value < 0: final_parts.append(f".decrement({repr(field)}, amount={repr(abs(value))})")
                    else: final_parts.append(f".increment({repr(field)}, amount=0)")
                elif op in ("set", "push", "pull", "min", "max", "mul"):
                    final_parts.append(f".{op}({repr(field)}, {value_repr})")
        return "".join(final_parts)

    def __bool__(self) -> bool:
        """Returns True if any update operations have been added."""
        return bool(self._operations)

    def __len__(self) -> int:
        """Returns the total number of fields being modified across all operations."""
        return sum(len(data) for data in self._operations.values())