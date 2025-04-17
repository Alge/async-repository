import pytest
from typing import List, Dict, Optional, Union
from datetime import datetime


# Define test model classes
class Address:
    street: str
    city: str
    zipcode: str

    def __init__(self, street="", city="", zipcode=""):
        self.street = street
        self.city = city
        self.zipcode = zipcode


# Define a class for metadata to handle nested field validation
class Metadata:
    key1: str
    key2: int
    flag: bool

    def __init__(self, key1="", key2=0, flag=False):
        self.key1 = key1
        self.key2 = key2
        self.flag = flag


class User:
    name: str
    age: int
    email: Optional[str]
    active: bool
    tags: List[str]
    addresses: List[Address]
    metadata: Metadata
    points: int
    balance: float
    score: Union[int, float]

    def __init__(
            self,
            name="",
            age=0,
            email=None,
            active=True,
            tags=None,
            addresses=None,
            metadata=None,
            points=0,
            balance=0.0,
            score=0,
    ):
        self.name = name
        self.age = age
        self.email = email
        self.active = active
        self.tags = tags or []
        self.addresses = addresses or []
        self.metadata = metadata or Metadata()
        self.points = points
        self.balance = balance
        self.score = score


class Inner:
    val: int

    def __init__(self, val=0):
        self.val = val


class Outer:
    inner: Inner

    def __init__(self, inner=None):
        self.inner = inner or Inner()


class ComplexItem:
    name: str
    value: int

    def __init__(self, name="", value=0):
        self.name = name
        self.value = value


class NestedTypes:
    simple_list: List[int]
    str_list: List[str]
    dict_field: Dict[str, str]
    nested: Outer
    complex_list: List[ComplexItem]
    counter: int

    def __init__(self):
        self.simple_list = []
        self.str_list = []
        self.dict_field = {}
        self.nested = Outer()
        self.complex_list = []
        self.counter = 0


class ModelWithUnions:
    field: Union[str, int]
    container: List[Union[str, int, bool]]
    counter: Union[int, float]

    def __init__(self):
        self.field = ""
        self.container = []
        self.counter = 0


class NumericModel:
    int_field: int
    float_field: float
    optional_int: Optional[int]
    union_numeric: Union[int, float]

    def __init__(self):
        self.int_field = 0
        self.float_field = 0.0
        self.optional_int = None
        self.union_numeric = 0


class Category:
    name: str
    items: List[str]
    tags: List[str]
    counts: List[int]

    def __init__(self, name="", items=None, tags=None, counts=None):
        self.name = name
        self.items = items or []
        self.tags = tags or []
        self.counts = counts or []


class Department:
    name: str
    categories: List[Category]
    members: List[str]

    def __init__(self, name="", categories=None, members=None):
        self.name = name
        self.categories = categories or []
        self.members = members or []


class Organization:
    name: str
    departments: List[Department]
    metadata: Metadata

    def __init__(self, name="", departments=None, metadata=None):
        self.name = name
        self.departments = departments or []
        self.metadata = metadata or Metadata()

        # Initialize with a sample department and category for testing
        if not departments:
            category = Category("General", ["item1", "item2"], ["tag1"], [1, 2, 3])
            department = Department("Main", [category], ["member1"])
            self.departments = [department]