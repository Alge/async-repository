class ObjectNotFoundException(Exception):
    """Exception raised when an object with the specified identifier does not exist."""

    def __init__(self, message: str = "The requested object was not found."):
        super().__init__(message)


class KeyAlreadyExistsException(Exception):
    """Exception raised when trying to insert an entity that would violate a unique constraint."""

    def __init__(self, message: str = "An object with the same key already exists."):
        super().__init__(message)


class ObjectValidationException(Exception):
    def __init__(self, message: str = "Object validation failed"):
        super().__init__(message)

