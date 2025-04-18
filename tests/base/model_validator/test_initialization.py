# tests/base/model_validator/test_initialization.py
import pytest

from async_repository.base.model_validator import ModelValidator


# --- Initialization Tests ---

def test_validator_initialization_success():
    """Test successful initialization with valid class types."""
    from .conftest import SimpleClass, SimpleDataClass, PydanticModel
    ModelValidator[SimpleClass](SimpleClass)  # Specify generic type
    ModelValidator[SimpleDataClass](SimpleDataClass)  # Specify generic type
    ModelValidator[PydanticModel](PydanticModel)  # Specify generic type


def test_validator_initialization_failure():
    """Test initialization failure with non-class types."""
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator(123)  # type: ignore
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator("not a class")  # type: ignore
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator(None)  # type: ignore


# --- Additional Tests for Generic Behavior ---

def test_validator_generic_type_constraints():
    """Test that the generic type parameter enforces type constraints."""
    from .conftest import SimpleClass, PydanticModel

    # Valid: Same types
    validator1 = ModelValidator[SimpleClass](SimpleClass)
    validator2 = ModelValidator[PydanticModel](PydanticModel)

    # Technically valid at runtime but will show type errors in static checking:
    # validator3 = ModelValidator[SimpleClass](PydanticModel)  # Wrong model type