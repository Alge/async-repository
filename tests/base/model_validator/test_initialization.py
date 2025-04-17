import pytest

from async_repository.base.model_validator import ModelValidator


# --- Initialization Tests ---

def test_validator_initialization_success():
    """Test successful initialization with valid class types."""
    from .conftest import SimpleClass, SimpleDataClass, PydanticModel
    ModelValidator(SimpleClass)
    ModelValidator(SimpleDataClass)
    ModelValidator(PydanticModel)


def test_validator_initialization_failure():
    """Test initialization failure with non-class types."""
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator(123)  # type: ignore
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator("not a class")  # type: ignore
    with pytest.raises(TypeError, match="model_type must be a class"):
        ModelValidator(None)  # type: ignore
