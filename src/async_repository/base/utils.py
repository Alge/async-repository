import logging
from dataclasses import is_dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)

import logging
from dataclasses import is_dataclass, asdict
from typing import Any

logger = logging.getLogger(__name__)


def prepare_for_storage(data: Any) -> Any:
    """
    Recursively convert Pydantic models, dataclasses, and special types to storage-compatible formats.

    This function ensures that complex objects are serialized properly for storage in any repository type.
    It handles:
    - Pydantic BaseModel instances with field aliases
    - Python dataclasses
    - Dictionaries (processing values recursively)
    - Lists, tuples and sets (processing each item)
    - Pydantic URL types (converting to strings)

    Args:
        data: The data to convert

    Returns:
        The converted data, ready for storage
    """
    # Handle None
    if data is None:
        return None

    # Handle dataclasses
    if is_dataclass(data) and not isinstance(data, type):
        return prepare_for_storage(asdict(data))

    # Handle Pydantic models
    if hasattr(data, "model_dump") and callable(getattr(data, "model_dump")):
        try:
            # For Pydantic v2: Use model_dump with json mode and by_alias=True
            # This is the key to preserving field aliases
            serialized = data.model_dump(mode="json", by_alias=True)
            return prepare_for_storage(serialized)
        except Exception as e:
            logger.debug(f"Error using model_dump(mode='json', by_alias=True): {e}")
            try:
                # Fallback to model_dump without json mode
                serialized = data.model_dump(by_alias=True)
                return prepare_for_storage(serialized)
            except Exception as e2:
                logger.debug(f"Error using model_dump(by_alias=True): {e2}")
                try:
                    # Fallback for Pydantic v1
                    serialized = data.dict(by_alias=True)
                    return prepare_for_storage(serialized)
                except Exception as e3:
                    logger.debug(f"Error using dict(by_alias=True): {e3}")
                    # Last resort fallback
                    return prepare_for_storage(dict(data.__dict__))

    # Handle dictionaries
    if isinstance(data, dict):
        return {k: prepare_for_storage(v) for k, v in data.items()}

    # Handle lists
    if isinstance(data, list):
        return [prepare_for_storage(item) for item in data]

    # Handle tuples
    if isinstance(data, tuple):
        return tuple(prepare_for_storage(item) for item in data)

    # Handle sets
    if isinstance(data, set):
        return [prepare_for_storage(item) for item in data]

    # Handle Pydantic URL types and other special types
    if hasattr(data, "__class__") and data.__class__.__module__ == "pydantic.networks":
        return str(data)

    # Return primitives as-is
    return data
