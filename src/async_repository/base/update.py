# -*- coding: utf-8 -*-
"""
Provides the Update class for building MongoDB-style update operations
with static analysis support for fields and optional type validation.
"""
import logging  # Use logging instead of warnings for consistency
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    Union,
    TypeVar,  # Added
    Generic,  # Added
)
from typing import dataclass_transform  # Added

# Assuming the statically analyzable Field class exists from query_builder_static.py
from .query import Field, _generate_fields_proxy

# Assuming the generic ModelValidator exists
from .model_validator import ValueTypeError, InvalidPathError, ModelValidator

# Generic Type Variable for the Model
M = TypeVar("M")


@dataclass_transform()  # Added decorator
class Update(Generic[M]):  # Make class Generic
    """
    A builder for MongoDB-style update instructions, providing static analysis
    for field access via the `fields` attribute. Uses ModelValidator for runtime type validation.
    """

    model_cls: Type[M]  # Store model class
    fields: Any  # The fields proxy (type inferred by dataclass_transform)
    _validator: Optional[ModelValidator[M]]  # Use generic validator
    _operations: Dict[str, Dict[str, Any]]
    _logger: logging.Logger  # Use logger

    def __init__(self, model_cls: Type[M]) -> None:
        """
        Initializes the Update builder for a specific model class.

        Args:
            model_cls: The data model class (e.g., User, Product).
        """
        self.model_cls = model_cls
        self._operations = {}
        self._logger = logging.getLogger(__name__)
        self._validator = None  # Initialize validator attribute

        try:
            # Generate fields proxy and initialize validator
            self.fields = _generate_fields_proxy(model_cls)  # Generate fields proxy
            self._validator = ModelValidator(model_cls)  # Initialize generic validator
            self._logger.debug(
                f"Initialized Update builder for model: {model_cls.__name__}"
            )
        except Exception as e:
            self._logger.error(
                f"Failed to initialize Update builder for {model_cls.__name__}: {e}",
                exc_info=True,
            )
            # Allow creation without validation? Or fail fast? Let's fail fast.
            raise ValueError(
                f"Could not initialize Update builder for model {model_cls.__name__}"
            ) from e

    # --- Update Methods (Modified Signatures) ---

    # Use Field[Any] as the type hint is mainly for access, validation handles specifics.
    def set(self, field: Field[Any], value: Any) -> "Update[M]":
        """Sets the value of a field ($set)."""
        field_path = field.path  # Extract path
        if self._validator:
            # Validate value against the type expected at this path
            self._validator.validate_value_for_path(field_path, value)
        # Use field_path as the key
        self._operations.setdefault("set", {})[field_path] = value
        return self

    def push(self, field: Field[Any], value: Any) -> "Update[M]":
        """Adds an element to an array field ($push)."""
        field_path = field.path
        if self._validator:
            is_list, item_type = self._validator.get_list_item_type(field_path)
            if not is_list:
                raise InvalidPathError(
                    f"Cannot $push to field '{field_path}': not a List or Optional[List]."
                )
            # Validate the item being pushed if the list item type is known
            if item_type is not Any:
                try:
                    self._validator.validate_value(
                        value, item_type, f"item for $push to '{field_path}'"
                    )
                except ValueTypeError as e:
                    # Wrap error for context
                    raise ValueTypeError(
                        f"Invalid value for $push to '{field_path}': {e}"
                    ) from e
        self._operations.setdefault("push", {})[field_path] = value
        return self

    def pop(self, field: Field[Any], direction: int = 1) -> "Update[M]":
        """Removes the first (-1) or last (1) element of an array field ($pop)."""
        field_path = field.path
        if direction not in (1, -1):
            raise ValueError(
                f"Direction for $pop must be 1 (last) or -1 (first), got {direction}."
            )
        if self._validator:
            # Check if the target field is actually a list
            is_list, _ = self._validator.get_list_item_type(field_path)
            if not is_list:
                raise InvalidPathError(
                    f"Cannot $pop from field '{field_path}': not a List or Optional[List]."
                )
        self._operations.setdefault("pop", {})[field_path] = direction
        return self

    def unset(self, field: Field[Any]) -> "Update[M]":
        """Removes a field ($unset)."""
        field_path = field.path
        if self._validator:
            # Check the path exists, even though we are removing it
            # This prevents unsetting completely invalid paths accidentally
            try:
                self._validator.get_field_type(field_path)
            except InvalidPathError as e:
                # Re-raise with context
                raise InvalidPathError(
                    f"Cannot $unset field at invalid path '{field_path}': {e}"
                ) from e
        # MongoDB $unset value doesn't matter, often "" or 1
        self._operations.setdefault("unset", {})[field_path] = ""
        return self

    def pull(self, field: Field[Any], value_or_condition: Any) -> "Update[M]":
        """Removes instances of a value or condition from an array field ($pull)."""
        field_path = field.path
        value = value_or_condition  # Rename for clarity inside method
        if self._validator:
            is_list, item_type = self._validator.get_list_item_type(field_path)
            if not is_list:
                raise InvalidPathError(
                    f"Cannot $pull from field '{field_path}': not a List or Optional[List]."
                )
            # Check if the value is a condition dict (e.g. {"$gt": 5}) vs a literal value
            is_operator_dict = isinstance(value, dict) and any(
                k.startswith("$") for k in value.keys()
            )
            # Only validate the literal value against the item type
            if item_type is not Any and not is_operator_dict:
                try:
                    # Validate the literal value against the expected list item type
                    self._validator.validate_value(
                        value, item_type, f"value for $pull from '{field_path}'"
                    )
                except ValueTypeError as e:
                    raise ValueTypeError(
                        f"Invalid value for $pull from '{field_path}': {e}"
                    ) from e
            # TODO: Could potentially validate the fields used *inside* the condition dict if needed
        self._operations.setdefault("pull", {})[field_path] = value
        return self

    def increment(
        self, field: Field[Any], amount: Union[int, float] = 1
    ) -> "Update[M]":
        """Increments a numeric field ($inc)."""
        field_path = field.path
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Increment amount must be numeric, got {type(amount).__name__}."
            )
        if self._validator:
            # Check if the target field is numeric
            if not self._validator.is_field_numeric(field_path):
                raise ValueTypeError(f"Cannot $inc non-numeric field '{field_path}'.")
        # Accumulate increments correctly if called multiple times
        current_inc = self._operations.setdefault("inc", {})
        current_inc[field_path] = current_inc.get(field_path, 0) + amount
        return self

    def decrement(
        self, field: Field[Any], amount: Union[int, float] = 1
    ) -> "Update[M]":
        """Decrements a numeric field (using $inc with negative amount)."""
        # Validation happens within self.increment
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Decrement amount must be numeric, got {type(amount).__name__}."
            )
        return self.increment(
            field, -abs(amount)
        )  # Ensure amount is positive before negating

    def min(self, field: Field[Any], value: Union[int, float]) -> "Update[M]":
        """Sets field to `value` if `value` is less than current field value ($min)."""
        field_path = field.path
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$min requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            if not self._validator.is_field_numeric(field_path):
                raise ValueTypeError(
                    f"Cannot apply $min to non-numeric field '{field_path}'."
                )
            # Optional: validate 'value' against the specific field type (e.g., int vs float)
            # self._validator.validate_value_for_path(field_path, value) # Might be too strict?
        self._operations.setdefault("min", {})[field_path] = value
        return self

    def max(self, field: Field[Any], value: Union[int, float]) -> "Update[M]":
        """Sets field to `value` if `value` is greater than current field value ($max)."""
        field_path = field.path
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$max requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            if not self._validator.is_field_numeric(field_path):
                raise ValueTypeError(
                    f"Cannot apply $max to non-numeric field '{field_path}'."
                )
            # Optional: validate 'value' against the specific field type
            # self._validator.validate_value_for_path(field_path, value)
        self._operations.setdefault("max", {})[field_path] = value
        return self

    def mul(self, field: Field[Any], value: Union[int, float]) -> "Update[M]":
        """Multiplies numeric field by `value` ($mul)."""
        field_path = field.path
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$mul requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            if not self._validator.is_field_numeric(field_path):
                raise ValueTypeError(
                    f"Cannot apply $mul to non-numeric field '{field_path}'."
                )
            # Optional: validate 'value' against the specific field type
            # self._validator.validate_value_for_path(field_path, value)
        self._operations.setdefault("mul", {})[field_path] = value
        return self

    # --- Build and Utility Methods ---

    # _serialize_value remains the same, it operates on the values being set/pushed etc.
    def _serialize_value(self, value: Any) -> Any:
        """Recursively serializes values, converting known models (like Pydantic) to dicts."""
        # --- PASTE YOUR ORIGINAL _serialize_value IMPLEMENTATION HERE ---
        if hasattr(value, "model_dump") and callable(value.model_dump):  # Pydantic v2
            # Try different arguments for broader compatibility
            try:
                return value.model_dump(mode="json", by_alias=True)
            except TypeError:
                try:
                    return value.model_dump(by_alias=True)
                except TypeError:
                    return value.model_dump()  # Fallback
        elif hasattr(value, "dict") and callable(value.dict):  # Pydantic v1
            try:
                return value.dict(by_alias=True)
            except TypeError:
                return value.dict()
        elif isinstance(value, dict):
            return {
                self._serialize_value(k): self._serialize_value(v)
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple, set)):
            # Convert tuples/sets to lists for JSON compatibility if needed by backend
            return [self._serialize_value(item) for item in value]
        # Add handling for other types like datetime, ObjectId if necessary here
        return value

    def build(self) -> Dict[str, Any]:
        """Constructs the final MongoDB update payload dictionary."""
        # This method remains the same as it operates on the internal _operations dict
        # which now uses field paths as keys.
        # --- PASTE YOUR ORIGINAL build IMPLEMENTATION HERE ---
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
        # Use a copy to avoid modifying internal state if build is called multiple times
        current_operations = self._operations.copy()

        for internal_op, data in current_operations.items():
            mongo_op = mongo_operator_map.get(internal_op)
            if not mongo_op:
                self._logger.warning(
                    f"Unknown internal op '{internal_op}' during build."
                )
                continue
            # Ensure data is not empty before serializing and adding
            if data:
                # Serialize values within the data dict for the specific operator
                serialized_data = {
                    field_path: self._serialize_value(val)
                    for field_path, val in data.items()
                }
                if serialized_data:  # Check again after serialization
                    payload[mongo_op] = serialized_data
        return payload

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation of the Update object."""
        # Representing with .fields might be complex. Show the built structure instead.
        model_name = self.model_cls.__name__
        built_ops = self.build()
        if not built_ops:
            return f"Update<{model_name}>{{}}"
        # Format the built operations nicely
        ops_repr = ", ".join(
            f"{op}: {data!r}" for op, data in sorted(built_ops.items())
        )
        return f"Update<{model_name}>{{{ops_repr}}}"

    def __bool__(self) -> bool:
        """Returns True if any update operations have been added."""
        return bool(self._operations)

    def __len__(self) -> int:
        """Returns the total number of fields being modified across all operations."""
        return sum(len(data) for data in self._operations.values())
