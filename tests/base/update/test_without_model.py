# tests/base/update/test_without_model.py

import pytest
from async_repository.base.update import (
    Update,
    SetOperation,
    IncrementOperation,
    PushOperation,
    MinOperation,
    MaxOperation,
    MultiplyOperation,
    PopOperation,
    PullOperation,
    UnsetOperation,
)
from tests.base.conftest import assert_operation_present


# --- Test basic operations without a model ---

def test_set_without_model():
    """Test set operation without a model."""
    update = Update().set("name", "John Doe")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "name", {"value": "John Doe"})

def test_increment_without_model():
    """Test increment operation without a model."""
    update = Update().increment("counter", 5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter", {"amount": 5})

def test_push_without_model():
    """Test push operation without a model."""
    update = Update().push("tags", "new_tag")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "tags", {"items": ["new_tag"]})


# --- Test fields proxy usage without a model ---

def test_fields_proxy_set_without_model():
    """Test set with fields proxy without a model."""
    update = Update()
    update.set(update.fields.name, "Jane Doe")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "name", {"value": "Jane Doe"})

def test_fields_proxy_increment_without_model():
    """Test increment with fields proxy without a model."""
    update = Update()
    update.increment(update.fields.visits, 3)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "visits", {"amount": 3})

def test_fields_proxy_push_without_model():
    """Test push with fields proxy without a model."""
    update = Update()
    update.push(update.fields.items_list, "item_a")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "items_list", {"items": ["item_a"]})

def test_fields_proxy_nested_set_without_model():
    """Test set on a nested path via fields proxy without a model."""
    update = Update()
    update.set(update.fields.user.profile.age, 35)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, SetOperation, "user.profile.age", {"value": 35})

def test_fields_proxy_deeply_nested_push_without_model():
    """Test push to a deeply nested path via fields proxy without a model."""
    update = Update()
    update.push(update.fields.blog.posts.comments.text, "A new comment")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "blog.posts.comments.text", {"items": ["A new comment"]})


# --- Test various complex operations without a model (on distinct fields) ---

def test_min_operation_without_model():
    update = Update().min("min_score_field", 0)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MinOperation, "min_score_field", {"value": 0})

def test_max_operation_without_model():
    update = Update().max("max_score_field", 100)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MaxOperation, "max_score_field", {"value": 100})

def test_mul_operation_without_model():
    update = Update().mul("price_multiplier", 1.5)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, MultiplyOperation, "price_multiplier", {"factor": 1.5})

def test_push_dict_item_without_model():
    update = Update().push("product_items", {"id": 101, "name": "Product A"})
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PushOperation, "product_items", {"items": [{"id": 101, "name": "Product A"}]})

def test_pop_operation_without_model():
    update = Update().pop("recent_logins", -1) # Remove first item
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PopOperation, "recent_logins", {"position": -1})

def test_pull_operation_without_model():
    update = Update().pull("expired_tags", "old_promo_tag")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, PullOperation, "expired_tags", {"value_or_condition": "old_promo_tag"})

def test_unset_nested_operation_without_model():
    update = Update().unset("user_profile.temporary_auth_data")
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, UnsetOperation, "user_profile.temporary_auth_data")

def test_increment_nested_operation_without_model():
    update = Update().increment("site_stats.page_visits", 1)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "site_stats.page_visits", {"amount": 1})


# --- Test increment/decrement restrictions (still apply without model) ---

def test_second_increment_on_same_field_rejected_without_model():
    """Test that a second increment on the same field is rejected."""
    update = Update()
    update.increment("counter_field", 5) # First is fine
    with pytest.raises(ValueError, match=r"Field 'counter_field' already has an operation"):
        update.increment("counter_field", 3) # Second on same field
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "counter_field", {"amount": 5})

def test_decrement_after_increment_on_same_field_rejected_without_model():
    """Test decrement after increment on same field is rejected."""
    update = Update()
    update.increment("value_field", 10)
    with pytest.raises(ValueError, match=r"Field 'value_field' already has an operation"):
        update.decrement("value_field", 2)
    result = update.build()
    assert len(result) == 1
    assert_operation_present(result, IncrementOperation, "value_field", {"amount": 10})

def test_increment_on_different_fields_allowed_without_model():
    """Test increment on different fields is allowed."""
    update = Update()
    update.increment("counter_one", 7)
    update.increment("counter_two", 3) # Different field
    result = update.build()
    assert len(result) == 2
    assert_operation_present(result, IncrementOperation, "counter_one", {"amount": 7})
    assert_operation_present(result, IncrementOperation, "counter_two", {"amount": 3})