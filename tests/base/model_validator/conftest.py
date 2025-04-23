# tests/base/model_validator/conftest.py
import pytest
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Set,
    Tuple,
    NewType,
    TypeVar,
    Generic,
)
from dataclasses import dataclass
from pydantic import BaseModel, Field as PydanticField

from async_repository.base.model_validator import (
    ModelValidator,
)

# --- Test Models ---


# Simple classes and dataclasses
class SimpleClass:
    name: str
    age: int
    is_active: bool = True


@dataclass
class SimpleDataClass:
    id: str
    value: float
    tags: Optional[List[str]] = None


# Nested structures
class Inner:
    val: int
    description: Optional[str]


@dataclass
class OuterDataClass:
    inner: Inner
    items: List[int]


class NestedClass:
    outer: OuterDataClass
    config: Dict[str, Any]
    maybe_inner: Optional[Inner]


# Using Pydantic for alias handling and more complex types
class PydanticInner(BaseModel):
    p_val: float = PydanticField(..., alias="pValue")


class PydanticModel(BaseModel):
    id: str
    count: int
    nested: PydanticInner
    optional_list: Optional[List[str]] = None
    union_field: Union[int, str]
    any_field: Any
    simple_list: list  # Plain list
    typed_dict: Dict[str, int]
    tuple_field: Tuple[int, str, bool]
    set_field: Set[float]


# Model with forward references (requires proper handling in get_type_hints)
class Node:
    value: int
    next_node: Optional["Node"] = None


MyInt = NewType("MyInt", int)
T = TypeVar("T")


class GenericModel(BaseModel, Generic[T]):
    data: T
    index: MyInt


# --- Fixtures ---


@pytest.fixture(
    params=[
        SimpleClass,
        SimpleDataClass,
        NestedClass,
        PydanticModel,
        Node,
        GenericModel[str],  # Test with a concrete generic type
    ]
)
def validator(request):
    """Provides a ModelValidator instance for various model types."""
    model_cls = request.param
    return ModelValidator[model_cls](model_cls)  # Specify the generic type


@pytest.fixture
def simple_validator():
    return ModelValidator[SimpleClass](SimpleClass)  # Specify the generic type


@pytest.fixture
def nested_validator():
    return ModelValidator[NestedClass](NestedClass)  # Specify the generic type


@pytest.fixture
def pydantic_validator():
    return ModelValidator[PydanticModel](PydanticModel)  # Specify the generic type
