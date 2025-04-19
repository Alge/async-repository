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

# Import the Field class and proxy generator
from .query import Field, _generate_fields_proxy, GenericFieldsProxy

# Import the validator
from .model_validator import ValueTypeError, InvalidPathError, ModelValidator

# Generic Type Variable for the Model
M = TypeVar("M")


@dataclass_transform()  # Added decorator
class Update(Generic[M]):
    """
    A builder for MongoDB-style update instructions, providing static analysis
    for field access via the `fields` attribute. Uses ModelValidator for runtime type validation.

    Can be used in two ways:
    1. With a model class: `Update(User)` - Provides static type checking and validation
    2. Without a model: `Update()` - Allows flexible updates without validation
    """

    model_cls: Optional[Type[M]]  # Store model class, now Optional
    fields: Any  # The fields proxy
    _validator: Optional[ModelValidator]  # Validator
    _operations: Dict[str, Dict[str, Any]]
    _logger: logging.Logger  # Use logger

    def __init__(self, model_cls: Optional[Type] = None) -> None:
        """
        Initializes the Update builder for a specific model class.

        Args:
            model_cls: The data model class (e.g., User, Product).
                       If None, allows updates without validation.
        """
        self.model_cls = model_cls
        self._operations = {}
        self._logger = logging.getLogger(__name__)
        self._validator = None  # Initialize validator attribute
        self.fields = None  # Initialize fields

        if model_cls is not None:
            try:
                # Generate fields proxy and initialize validator
                self.fields = _generate_fields_proxy(model_cls)
                self._validator = ModelValidator(model_cls)
                self._logger.debug(
                    f"Initialized Update builder for model: {model_cls.__name__}"
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to initialize Update builder for {model_cls.__name__}: {e}",
                    exc_info=True,
                )
                # Fallback to generic fields if model-specific initialization fails
                self.fields = GenericFieldsProxy()
                raise ValueError(
                    f"Could not initialize Update builder for model {model_cls.__name__}"
                ) from e
        else:
            # For model-less operation, use a generic fields proxy
            self.fields = GenericFieldsProxy()
            self._logger.debug("Initialized Update builder without model validation")

    def _get_field_path(self, field: Union[str, Field]) -> str:
        """Helper method to extract field path from either string or Field object."""
        if isinstance(field, str):
            return field
        elif hasattr(field, 'path'):
            return field.path
        else:
            raise TypeError(
                f"Expected field to be str or Field, got {type(field).__name__}")

    # --- Update Methods ---

    def set(self, field: Union[str, Field[Any]], value: Any) -> "Update[M]":
        """Sets the value of a field ($set)."""
        field_path = self._get_field_path(field)
        if self._validator:
            try:
                # Validate value against the type expected at this path
                self._validator.validate_value_for_path(field_path, value)
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        # Use field_path as the key
        self._operations.setdefault("set", {})[field_path] = value
        return self

    def push(self, field: Union[str, Field[Any]], value: Any) -> "Update[M]":
        """Adds an element to an array field ($push)."""
        field_path = self._get_field_path(field)
        if self._validator:
            try:
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
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("push", {})[field_path] = value
        return self

    def pop(self, field: Union[str, Field[Any]], direction: int = 1) -> "Update[M]":
        """Removes the first (-1) or last (1) element of an array field ($pop)."""
        field_path = self._get_field_path(field)
        if direction not in (1, -1):
            raise ValueError(
                f"Direction for $pop must be 1 (last) or -1 (first), got {direction}."
            )
        if self._validator:
            try:
                # Check if the target field is actually a list
                is_list, _ = self._validator.get_list_item_type(field_path)
                if not is_list:
                    raise InvalidPathError(
                        f"Cannot $pop from field '{field_path}': not a List or Optional[List]."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("pop", {})[field_path] = direction
        return self

    def unset(self, field: Union[str, Field[Any]]) -> "Update[M]":
        """Removes a field ($unset)."""
        field_path = self._get_field_path(field)
        if self._validator:
            try:
                # Check the path exists, even though we are removing it
                # This prevents unsetting completely invalid paths accidentally
                self._validator.get_field_type(field_path)
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        # MongoDB $unset value doesn't matter, often "" or 1
        self._operations.setdefault("unset", {})[field_path] = ""
        return self

    def pull(self, field: Union[str, Field[Any]],
             value_or_condition: Any) -> "Update[M]":
        """Removes instances of a value or condition from an array field ($pull)."""
        field_path = self._get_field_path(field)
        value = value_or_condition  # Rename for clarity inside method
        if self._validator:
            try:
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
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("pull", {})[field_path] = value
        return self

    def increment(
            self, field: Union[str, Field[Any]], amount: Union[int, float] = 1
    ) -> "Update[M]":
        """
        Increments a numeric field ($inc).

        Only one increment or decrement operation is allowed per field in an update.
        """
        field_path = self._get_field_path(field)
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Increment amount must be numeric, got {type(amount).__name__}."
            )

        # Check if field already has an increment or decrement operation
        current_inc = self._operations.get("inc", {})
        if field_path in current_inc:
            raise ValueError(
                f"Field '{field_path}' already has an increment operation. "
                f"Multiple increment/decrement operations on the same field are not allowed."
            )

        if self._validator:
            try:
                # Check if the target field is numeric
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError(
                        f"Cannot $inc non-numeric field '{field_path}'.")
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))

        # Set the increment
        self._operations.setdefault("inc", {})[field_path] = amount
        return self

    def decrement(
            self, field: Union[str, Field[Any]], amount: Union[int, float] = 1
    ) -> "Update[M]":
        """
        Decrements a numeric field (using $inc with negative amount).

        Only one increment or decrement operation is allowed per field in an update.
        """
        # Validation happens within self.increment
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Decrement amount must be numeric, got {type(amount).__name__}."
            )

        field_path = self._get_field_path(field)

        # Check if field already has an increment or decrement operation
        current_inc = self._operations.get("inc", {})
        if field_path in current_inc:
            raise ValueError(
                f"Field '{field_path}' already has an increment operation. "
                f"Multiple increment/decrement operations on the same field are not allowed."
            )

        # Validate field through the increment method
        if self._validator:
            try:
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError(
                        f"Cannot $inc non-numeric field '{field_path}'.")
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))

        # Set the decrement (negative increment)
        self._operations.setdefault("inc", {})[field_path] = -amount
        return self

    def min(self, field: Union[str, Field[Any]],
            value: Union[int, float]) -> "Update[M]":
        """Sets field to `value` if `value` is less than current field value ($min)."""
        field_path = self._get_field_path(field)
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$min requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            try:
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError(
                        f"Cannot apply $min to non-numeric field '{field_path}'."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("min", {})[field_path] = value
        return self

    def max(self, field: Union[str, Field[Any]],
            value: Union[int, float]) -> "Update[M]":
        """Sets field to `value` if `value` is greater than current field value ($max)."""
        field_path = self._get_field_path(field)
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$max requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            try:
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError(
                        f"Cannot apply $max to non-numeric field '{field_path}'."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("max", {})[field_path] = value
        return self

    def mul(self, field: Union[str, Field[Any]],
            value: Union[int, float]) -> "Update[M]":
        """Multiplies numeric field by `value` ($mul)."""
        field_path = self._get_field_path(field)
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"$mul requires a numeric value, got {type(value).__name__}."
            )
        if self._validator:
            try:
                if not self._validator.is_field_numeric(field_path):
                    raise ValueTypeError(
                        f"Cannot apply $mul to non-numeric field '{field_path}'."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise TypeError(str(e))
        self._operations.setdefault("mul", {})[field_path] = value
        return self

    # --- Build and Utility Methods ---

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serializes values, converting known models (like Pydantic) to dicts."""
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
        model_name = self.model_cls.__name__ if self.model_cls else "Anonymous"
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