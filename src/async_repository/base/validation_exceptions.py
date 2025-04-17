# validation_exceptions.py (or within the validator module)
class ValidationError(TypeError):
    """Base class for validation errors related to model types."""
    pass

class InvalidPathError(ValidationError, AttributeError):
    """Error raised when a field path does not exist or is invalid for the model."""
    pass

class ValueTypeError(ValidationError, TypeError):
    """Error raised when a value's type is incompatible with the expected field type."""
    pass