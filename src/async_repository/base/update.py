# src/async_repository/base/update.py

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    TypeVar,
    Generic,
    Literal,
    get_origin,
    get_args,
)
from typing import dataclass_transform

# Import the Field class and proxy generator
from .query import Field, _generate_fields_proxy, GenericFieldsProxy

# Import the validator and exceptions
from .model_validator import (
    ValueTypeError,
    InvalidPathError,
    ModelValidator,
    ValidationError,
    _is_none_type,
)

# Import utils for serialization
from .utils import prepare_for_storage


# --- Agnostic Update Operation Classes ---
@dataclass
class UpdateOperation:
    field_path: str


@dataclass
class SetOperation(UpdateOperation):
    value: Any


@dataclass
class UnsetOperation(UpdateOperation):
    pass


@dataclass
class IncrementOperation(UpdateOperation):
    amount: Union[int, float]


@dataclass
class MultiplyOperation(UpdateOperation):
    factor: Union[int, float]


@dataclass
class MinOperation(UpdateOperation):
    value: Union[int, float]


@dataclass
class MaxOperation(UpdateOperation):
    value: Union[int, float]


@dataclass
class PushOperation(UpdateOperation):
    items: List[Any]


@dataclass
class PopOperation(UpdateOperation):
    position: Literal[-1, 1]


@dataclass
class PullOperation(UpdateOperation):
    value_or_condition: Any


# --- End Agnostic Update Operation Classes ---

M = TypeVar("M")


@dataclass_transform()
class Update(Generic[M]):
    model_cls: Optional[Type[M]]
    fields: Any
    _validator: Optional[ModelValidator]
    _operations: List[UpdateOperation]
    _logger: logging.Logger

    def __init__(self, model_cls: Optional[Type] = None) -> None:
        self.model_cls = model_cls
        self._operations = []
        self._logger = logging.getLogger(__name__)
        self._validator = None
        self.fields = None
        if model_cls is not None:
            try:
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
                self.fields = GenericFieldsProxy()
                raise ValueError(
                    f"Could not initialize Update builder for model {model_cls.__name__}"
                ) from e
        else:
            self.fields = GenericFieldsProxy()
            self._logger.debug("Initialized Update builder without model validation")

    def _get_field_path(self, field: Union[str, Field]) -> str:
        if isinstance(field, str):
            return field
        if isinstance(field, Field):
            return field.path
        raise TypeError(
            f"Expected field to be str or Field, got {type(field).__name__}"
        )

    def _check_field_conflict(self, field_path: str) -> None:
        """
        Check if the field already has any operation applied to it or conflicts with parent/child fields.

        A field conflicts with another if:
        1. They are the exact same field path
        2. One is a parent of the other (e.g., "metadata" and "metadata.key1")
        3. One is a child of the other (e.g., "metadata.key1" and "metadata")
        """
        for op in self._operations:
            existing_path = op.field_path

            # Check for exact field path match
            if existing_path == field_path:
                conflict_type = "exact match"

            # Check if existing path is a parent of new field path
            elif field_path.startswith(existing_path + "."):
                conflict_type = "child"

            # Check if new field path is a parent of existing path
            elif existing_path.startswith(field_path + "."):
                conflict_type = "parent"

            else:
                # No conflict found with this operation
                continue

            # We found a conflict - log and raise an error
            self._logger.warning(
                f"Field conflict detected: '{field_path}' conflicts with existing operation on '{existing_path}' ({conflict_type}). "
                f"SQL databases don't support operations that affect the same fields in a single UPDATE statement. "
                f"Split into separate UPDATE calls to apply operations sequentially."
            )

            # Customize the error message based on the type of conflict
            if conflict_type == "exact match":
                message = f"Field '{field_path}' already has an operation. Multiple operations on the same field are not allowed in a single update."
            elif conflict_type == "child":
                message = f"Field '{field_path}' conflicts with existing operation on parent field '{existing_path}'. Parent-child field conflicts are not allowed in a single update."
            else:  # parent
                message = f"Field '{field_path}' conflicts with existing operation on child field '{existing_path}'. Parent-child field conflicts are not allowed in a single update."

            raise ValueError(message)

    def _is_nested_path_in_dict(self, field_path: str) -> bool:
        """Check if a field path is a nested path within a dictionary field."""
        if "." not in field_path:
            return False

        # Get the base field (first segment of the path)
        base_field = field_path.split(".")[0]

        try:
            if not self._validator:
                return False

            field_type = self._validator.get_field_type(base_field)
            origin = get_origin(field_type)
            args = get_args(field_type)

            # Check if base field is a dictionary type directly
            is_dict_type = origin in (dict, Dict)

            # Or if it's an Optional[Dict]
            if origin is Union:
                is_dict_type = any(
                    get_origin(arg) in (dict, Dict)
                    for arg in args
                    if not _is_none_type(arg)
                )

            return is_dict_type
        except Exception:
            # If we can't determine the type, assume it's not a dict
            return False

    # --- Update Methods ---
    def set(self, field: Union[str, Field[Any]], value: Any) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        serialized_value = prepare_for_storage(value)
        if self._validator:
            try:
                self._validator.validate_value_for_path(field_path, serialized_value)
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during set: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for set on '{field_path}': {e}"
                ) from e
        self._operations.append(
            SetOperation(field_path=field_path, value=serialized_value)
        )
        return self

    def push(self, field: Union[str, Field[Any]], value: Any) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        serialized_item = prepare_for_storage(value)

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing push operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                PushOperation(field_path=field_path, items=[serialized_item]))
            return self

        # Original validation logic
        if self._validator:
            try:
                is_list, item_type = self._validator.get_list_item_type(field_path)
                if not is_list:
                    raise InvalidPathError(
                        f"Cannot push to field '{field_path}': not a List or Optional[List]."
                    )
                if item_type is not Any:
                    try:
                        self._validator.validate_value(
                            serialized_item,
                            item_type,
                            f"item for push to '{field_path}'",
                        )
                    except ValueTypeError as e:
                        raise ValueTypeError(
                            f"Invalid value type for push to '{field_path}': {e}"
                        ) from e
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during push: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for push on '{field_path}': {e}"
                ) from e

        self._operations.append(
            PushOperation(field_path=field_path, items=[serialized_item])
        )
        return self

    def pop(
            self, field: Union[str, Field[Any]], position: Literal[-1, 1] = 1
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if position not in (1, -1):
            raise ValueError(
                f"Position for pop must be 1 (last) or -1 (first), got {position}."
            )

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing pop operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                PopOperation(field_path=field_path, position=position))
            return self

        # Original validation logic
        if self._validator:
            try:
                is_list, _ = self._validator.get_list_item_type(field_path)
                if not is_list:
                    raise InvalidPathError(
                        f"Cannot pop from field '{field_path}': not a List or Optional[List]."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during pop: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for pop on '{field_path}': {e}"
                ) from e

        self._operations.append(PopOperation(field_path=field_path, position=position))
        return self

    def unset(self, field: Union[str, Field[Any]]) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if self._validator:
            try:
                self._validator.get_field_type(field_path)
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during unset: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for unset on '{field_path}': {e}"
                ) from e
        self._operations.append(UnsetOperation(field_path=field_path))
        return self

    def pull(
            self, field: Union[str, Field[Any]], value_or_condition: Any
    ) -> "Update[M]":
        """Adds a 'pull' operation (removes matching items from an array)."""
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        is_operator_dict = isinstance(value_or_condition, dict) and any(
            k.startswith("$") for k in value_or_condition.keys()
        )

        # Store the original value if it's an operator dict, otherwise serialize
        value_to_store = (
            value_or_condition
            if is_operator_dict
            else prepare_for_storage(value_or_condition)
        )

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing pull operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                PullOperation(field_path=field_path, value_or_condition=value_to_store)
            )
            return self

        # Original validation logic
        if self._validator:
            try:
                is_list, item_type = self._validator.get_list_item_type(field_path)
                if not is_list:
                    raise InvalidPathError(
                        f"Cannot pull from field '{field_path}': not a List or Optional[List]."
                    )

                # Validate the item unless it's an operator dict OR the item type is Any
                # We NOW validate non-operator dictionaries against the item_type.
                should_validate_item = item_type is not Any and not is_operator_dict

                if should_validate_item:
                    # Validate the value_to_store (which is serialized if not operator dict)
                    self._validator.validate_value(
                        value_to_store, item_type, f"value for pull from '{field_path}'"
                    )
                # If it's an operator_dict, we skip the item validation.

            except (InvalidPathError, ValueTypeError) as e:
                # Need to wrap ValueTypeError raised during validation for context
                if isinstance(e, ValueTypeError):
                    raise ValueTypeError(
                        f"Invalid value type for pull from '{field_path}': {e}"
                    ) from e
                else:
                    raise e  # Re-raise InvalidPathError directly
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during pull: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for pull on '{field_path}': {e}"
                ) from e

        self._operations.append(
            PullOperation(field_path=field_path, value_or_condition=value_to_store)
        )
        return self

    def increment(
            self, field: Union[str, Field[Any]], amount: Union[int, float] = 1
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Increment amount must be numeric, got {type(amount).__name__}."
            )

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing increment operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                IncrementOperation(field_path=field_path, amount=amount)
            )
            return self

        # Original validation logic
        if self._validator:
            try:
                field_type = self._validator.get_field_type(field_path)
                origin = get_origin(field_type)
                args = get_args(field_type)
                is_field_numeric = False
                if origin is Union:
                    is_field_numeric = any(
                        self._validator._is_single_type_numeric(a)
                        for a in args
                        if not _is_none_type(a)
                    )
                else:
                    is_field_numeric = self._validator._is_single_type_numeric(
                        field_type
                    )
                if not is_field_numeric:
                    raise ValueTypeError(
                        f"Cannot increment non-numeric field '{field_path}' (type: {self._validator._get_type_name(field_type)})."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during increment: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for increment on '{field_path}': {e}"
                ) from e

        self._operations.append(
            IncrementOperation(field_path=field_path, amount=amount)
        )
        return self

    def decrement(
            self, field: Union[str, Field[Any]], amount: Union[int, float] = 1
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        if not isinstance(amount, (int, float)):
            raise TypeError(
                f"Decrement amount must be numeric, got {type(amount).__name__}."
            )
        # Use the logic of the increment method with negative value
        # We don't call self.increment to avoid adding the operation twice
        self._check_field_conflict(field_path)

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing decrement operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                IncrementOperation(field_path=field_path, amount=-amount)
            )
            return self

        # Original validation logic
        if self._validator:
            try:
                field_type = self._validator.get_field_type(field_path)
                origin = get_origin(field_type)
                args = get_args(field_type)
                is_field_numeric = False
                if origin is Union:
                    is_field_numeric = any(
                        self._validator._is_single_type_numeric(a)
                        for a in args
                        if not _is_none_type(a)
                    )
                else:
                    is_field_numeric = self._validator._is_single_type_numeric(
                        field_type
                    )
                if not is_field_numeric:
                    raise ValueTypeError(
                        f"Cannot decrement non-numeric field '{field_path}' (type: {self._validator._get_type_name(field_type)})."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during decrement: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for decrement on '{field_path}': {e}"
                ) from e

        self._operations.append(
            IncrementOperation(field_path=field_path, amount=-amount)
        )
        return self

    def min(
            self, field: Union[str, Field[Any]], value: Union[int, float]
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if not isinstance(value, (int, float)):
            raise TypeError(f"Min value must be numeric, got {type(value).__name__}.")

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing min operation on nested path '{field_path}' within dictionary field")
            self._operations.append(MinOperation(field_path=field_path, value=value))
            return self

        # Original validation logic
        if self._validator:
            try:
                field_type = self._validator.get_field_type(field_path)
                origin = get_origin(field_type)
                args = get_args(field_type)
                is_field_numeric = False
                if origin is Union:
                    is_field_numeric = any(
                        self._validator._is_single_type_numeric(a)
                        for a in args
                        if not _is_none_type(a)
                    )
                else:
                    is_field_numeric = self._validator._is_single_type_numeric(
                        field_type
                    )
                if not is_field_numeric:
                    raise ValueTypeError(
                        f"Cannot apply min to non-numeric field '{field_path}' (type: {self._validator._get_type_name(field_type)})."
                    )
                self._validator.validate_value(
                    value, field_type, f"value for min on '{field_path}'"
                )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during min: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for min on '{field_path}': {e}"
                ) from e

        self._operations.append(MinOperation(field_path=field_path, value=value))
        return self

    def max(
            self, field: Union[str, Field[Any]], value: Union[int, float]
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if not isinstance(value, (int, float)):
            raise TypeError(f"Max value must be numeric, got {type(value).__name__}.")

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing max operation on nested path '{field_path}' within dictionary field")
            self._operations.append(MaxOperation(field_path=field_path, value=value))
            return self

        # Original validation logic
        if self._validator:
            try:
                field_type = self._validator.get_field_type(field_path)
                origin = get_origin(field_type)
                args = get_args(field_type)
                is_field_numeric = False
                if origin is Union:
                    is_field_numeric = any(
                        self._validator._is_single_type_numeric(a)
                        for a in args
                        if not _is_none_type(a)
                    )
                else:
                    is_field_numeric = self._validator._is_single_type_numeric(
                        field_type
                    )
                if not is_field_numeric:
                    raise ValueTypeError(
                        f"Cannot apply max to non-numeric field '{field_path}' (type: {self._validator._get_type_name(field_type)})."
                    )
                self._validator.validate_value(
                    value, field_type, f"value for max on '{field_path}'"
                )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during max: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for max on '{field_path}': {e}"
                ) from e

        self._operations.append(MaxOperation(field_path=field_path, value=value))
        return self

    def mul(
            self, field: Union[str, Field[Any]], factor: Union[int, float]
    ) -> "Update[M]":
        field_path = self._get_field_path(field)
        self._check_field_conflict(field_path)  # Check for field conflict
        if not isinstance(factor, (int, float)):
            raise TypeError(
                f"Multiply factor must be numeric, got {type(factor).__name__}."
            )

        # Skip validation for nested paths in dictionary fields
        if self._validator and self._is_nested_path_in_dict(field_path):
            self._logger.debug(
                f"Allowing multiply operation on nested path '{field_path}' within dictionary field")
            self._operations.append(
                MultiplyOperation(field_path=field_path, factor=factor))
            return self

        # Original validation logic
        if self._validator:
            try:
                field_type = self._validator.get_field_type(field_path)
                origin = get_origin(field_type)
                args = get_args(field_type)
                is_field_numeric = False
                if origin is Union:
                    is_field_numeric = any(
                        self._validator._is_single_type_numeric(a)
                        for a in args
                        if not _is_none_type(a)
                    )
                else:
                    is_field_numeric = self._validator._is_single_type_numeric(
                        field_type
                    )
                if not is_field_numeric:
                    raise ValueTypeError(
                        f"Cannot apply multiply to non-numeric field '{field_path}' (type: {self._validator._get_type_name(field_type)})."
                    )
            except (InvalidPathError, ValueTypeError) as e:
                raise e
            except Exception as e:
                self._logger.error(
                    f"Unexpected validation error during mul: {e}", exc_info=True
                )
                raise RuntimeError(
                    f"Unexpected validation error for mul on '{field_path}': {e}"
                ) from e

        self._operations.append(MultiplyOperation(field_path=field_path, factor=factor))
        return self

    # --- Build and Utility Methods ---
    def build(self) -> List[UpdateOperation]:
        return list(self._operations)

    def __repr__(self) -> str:
        model_name = self.model_cls.__name__ if self.model_cls else "Anonymous"
        if not self._operations:
            return f"Update<{model_name}>([])"
        ops_repr = ", ".join(repr(op) for op in self._operations)
        return f"Update<{model_name}>([{ops_repr}])"

    def __bool__(self) -> bool:
        return bool(self._operations)

    def __len__(self) -> int:
        return len(self._operations)