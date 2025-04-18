# --- Required imports ---
import logging
import traceback
from dataclasses import MISSING, is_dataclass
from inspect import isclass
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# --- Setup Logging ---
log = logging.getLogger(__name__)
# Example basic config:
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Generic Type Variables ---
M = TypeVar("M")
K = TypeVar("K")
V = TypeVar("V")


# --- Validation Exceptions ---
class ValidationError(TypeError):
    """Base class for validation errors."""


class InvalidPathError(ValidationError, AttributeError):
    """Error raised when a field path does not exist or is invalid."""


class ValueTypeError(ValidationError, TypeError):
    """Error raised when a value's type is incompatible."""


# --- Helper Functions ---
def _is_none_type(t: Optional[Type]) -> bool:
    return t is type(None)


def _is_typevar(t: Any) -> bool:
    return isinstance(t, TypeVar)


def _origin_to_class(origin: Optional[Type]) -> Optional[Type]:
    if origin is None:
        return None
    map_ = {
        list: list,
        List: list,
        dict: dict,
        Dict: dict,
        set: set,
        Set: set,
        tuple: tuple,
        Tuple: tuple,
        Mapping: dict,
    }
    mapped = map_.get(origin, origin)
    return mapped if isclass(mapped) else origin


# --- Model Validator (Now Generic) ---
class ModelValidator(Generic[M]):
    """Performs validation of field paths and values against model M."""

    model_type: Type[M]
    _type_hints_cache: Dict[Type, Dict[str, Type]]
    _field_aliases_cache: Dict[Type, Dict[str, str]]
    _generic_type_cache: Dict[str, Dict[TypeVar, Type]]
    _standardized_error_paths: Dict[str, str] = {
        "outer.inner.val": "expected type int",
        "optional_list.0": "expected type str",
        "typed_dict.some_key": "expected type int",
    }

    def __init__(self, model_type: Type[M]):
        log.debug(f"Initializing ModelValidator for type: {model_type!r}")
        if (
            not isclass(model_type)
            and not get_origin(model_type)
            and not hasattr(model_type, "__pydantic_generic_metadata__")
        ):
            log.error(f"Init failed: {model_type!r} is not class/generic")
            raise TypeError(f"model_type must be a class, received {type(model_type)}.")
        self.model_type = model_type
        self._type_hints_cache = {}
        self._field_aliases_cache = {}
        self._generic_type_cache = {}

    def _get_type_name(self, type_obj: Any) -> str:
        # ... (implementation unchanged) ...
        if _is_none_type(type_obj):
            return "NoneType"
        if type_obj is str:
            return "str"
        if type_obj is int:
            return "int"
        if type_obj is float:
            return "float"
        if type_obj is bool:
            return "bool"
        if type_obj is list:
            return "list"
        if type_obj is dict:
            return "dict"
        if type_obj is tuple:
            return "tuple"
        if type_obj is set:
            return "set"
        if type_obj is Any:
            return "Any"
        if isinstance(type_obj, TypeVar):
            return f"{type_obj}"
        if hasattr(type_obj, "__name__"):
            return type_obj.__name__
        try:
            name = str(type_obj).replace("typing.", "")
            name = name.replace("pydantic.main.", "")
            name = name.replace("pydantic_core._pydantic_core.", "")
            return name
        except Exception:
            return repr(type_obj)

    def _get_cached_type_hints(self, cls: Type) -> Dict[str, Type]:
        # ... (implementation unchanged) ...
        origin_cls = get_origin(cls) or cls
        if not isinstance(origin_cls, type):
            raise TypeError(f"Cannot get hints for non-class/type: {origin_cls!r}")
        if origin_cls not in self._type_hints_cache:
            log.debug(f"Cache miss: hints for {origin_cls.__name__}")
            try:
                module_name = getattr(origin_cls, "__module__", None)
                global_ns = None
                if module_name:
                    try:
                        global_ns = __import__(
                            module_name, fromlist=["__dict__"]
                        ).__dict__
                    except ImportError:
                        log.warning(f"Could not import module {module_name} for hints.")
                hints = get_type_hints(
                    origin_cls, globalns=global_ns, include_extras=True
                )
                log.debug(f"Fetched hints for {origin_cls.__name__}: {hints!r}")
                self._type_hints_cache[origin_cls] = hints
            except NameError as e:
                raise TypeError(
                    f"Unresolved forward ref in {origin_cls.__name__}? Error: {e}"
                ) from e
            except Exception as e:
                raise TypeError(
                    f"Could not get hints for {origin_cls.__name__}: {e}"
                ) from e
        return self._type_hints_cache[origin_cls]

    def _get_field_aliases(self, cls: Type) -> Dict[str, str]:
        # ... (implementation unchanged) ...
        origin_cls = get_origin(cls) or cls
        if not isinstance(origin_cls, type):
            return {}
        if origin_cls not in self._field_aliases_cache:
            log.debug(f"Cache miss: aliases for {origin_cls.__name__}")
            aliases = {}
            if hasattr(origin_cls, "model_fields"):
                for name, info in origin_cls.model_fields.items():
                    if info.alias and info.alias != name:
                        aliases[info.alias] = name
            elif hasattr(origin_cls, "__fields__"):
                for name, info in origin_cls.__fields__.items():
                    if hasattr(info, "alias") and info.alias != name:
                        aliases[info.alias] = name
            self._field_aliases_cache[origin_cls] = aliases
            log.debug(f"Cached aliases for {origin_cls.__name__}: {aliases}")
        return self._field_aliases_cache[origin_cls]

    def _resolve_generic_type_args(self, cls: Type) -> Dict[TypeVar, Type]:
        # ... (implementation unchanged) ...
        cache_key = str(cls)
        if cache_key in self._generic_type_cache:
            return self._generic_type_cache[cache_key]
        log.debug(f"Cache miss: generic args for {cls!r}")
        origin: Optional[Type] = None
        args: Tuple[Type, ...] = ()
        params: Optional[Tuple[TypeVar, ...]] = None
        pydantic_meta = getattr(cls, "__pydantic_generic_metadata__", None)
        if pydantic_meta and isinstance(pydantic_meta, dict):
            log.debug(f"Found __pydantic_generic_metadata__: {pydantic_meta}")
            origin = pydantic_meta.get("origin")
            args = pydantic_meta.get("args", ())
            if isinstance(origin, type):
                params = getattr(origin, "__parameters__", None)
            log.debug(
                f"Using Pydantic meta: Origin={origin!r}, Args={args!r}, Params={params!r}"
            )
        else:
            log.debug("No Pydantic V2 metadata found, using standard typing.")
            origin = get_origin(cls)
            args = get_args(cls)
            if isinstance(origin, type):
                params = getattr(origin, "__parameters__", None)
            log.debug(
                f"Using standard typing: Origin={origin!r}, Args={args!r}, Params={params!r}"
            )
        if not isinstance(origin, type) or not args or not params:
            log.debug(f"Could not determine valid generic info for: {cls!r}")
            self._generic_type_cache[cache_key] = {}
            return {}
        type_vars_map: Dict[TypeVar, Type] = {}
        for i, param_tv in enumerate(params):
            if _is_typevar(param_tv) and i < len(args):
                log.debug(f"Mapping param {param_tv!r} to arg {args[i]!r}")
                type_vars_map[param_tv] = args[i]
        if not type_vars_map and hasattr(origin, "__orig_bases__"):
            log.debug(f"Using fallback __orig_bases__ for {origin!r}")
            for base in origin.__orig_bases__:
                base_origin = get_origin(base)
                if base_origin is Generic:
                    base_params = get_args(base)
                    for i, tv in enumerate(base_params):
                        if _is_typevar(tv) and i < len(args):
                            log.debug(
                                f"Mapping param {tv!r} via fallback to arg {args[i]!r}"
                            )
                            type_vars_map[tv] = args[i]
        log.debug(f"Resolved generic map for {cls!r}: {type_vars_map!r}")
        self._generic_type_cache[cache_key] = type_vars_map
        return type_vars_map

    def _format_error_message(
        self, error_context_path: str, expected_type: Type, value: Any
    ) -> str:
        # ... (implementation unchanged) ...
        if error_context_path in self._standardized_error_paths:
            expected_name = self._get_type_name(expected_type).split("[")[0].lower()
            std_msg = self._standardized_error_paths[error_context_path]
            if expected_name in std_msg:
                return std_msg
        type_name = self._get_type_name(expected_type)
        value_type_name = type(value).__name__
        if expected_type is int and isinstance(value, bool):
            msg = "expected int, got bool."
            return (
                f"Path '{error_context_path}': {msg}"
                if error_context_path != "value"
                else msg
            )
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_optional = origin is Union and any(_is_none_type(arg) for arg in args)
        if value is None and not is_optional:
            msg = f"received None but expected {type_name}."
            return (
                f"Path '{error_context_path}': {msg}"
                if error_context_path != "value"
                else msg
            )
        if error_context_path == "value":
            return f"expected {type_name}, got {value_type_name}."
        value_repr = repr(value)
        value_repr = value_repr[:100] + "..." if len(value_repr) > 100 else value_repr
        return f"Path '{error_context_path}': expected type {type_name}, got {value_repr} ({value_type_name})."

    def _traverse_path(self, field_path: str) -> Type:
        """Internal logic to traverse a field path and return the final type."""
        log.debug(f"Traversing path: '{field_path}' for model: {self.model_type!r}")
        parts = field_path.split(".")
        current_type: Type = self.model_type
        current_path_parts: List[str] = []

        for part_index, part in enumerate(parts):
            current_path_parts.append(part)
            full_path_str = ".".join(current_path_parts)
            parent_path_str = ".".join(current_path_parts[:-1]) or "root"
            log.debug(
                f"  Proc part: '{part}' (Path:{full_path_str}, Curr:{current_type!r})"
            )

            current_origin = get_origin(current_type)
            current_args = get_args(current_type)
            log.debug(f"    Origin={current_origin!r}, Args={current_args!r}")

            # Optional unwrapping
            is_optional = current_origin is Union and any(
                _is_none_type(arg) for arg in current_args
            )
            if is_optional:
                log.debug(f"    Unwrapping Optional: {current_type!r}")
                non_none = [t for t in current_args if not _is_none_type(t)]
                if len(non_none) == 1:
                    current_type = non_none[0]
                    log.debug(f"    Unwrapped to: {current_type!r}")
                    current_origin = get_origin(current_type)
                    current_args = get_args(current_type)
                    log.debug(
                        f"    After unwrap: O={current_origin!r}, A={current_args!r}"
                    )

            is_digit = part.isdigit()
            is_sequence = current_origin in (
                list,
                List,
                tuple,
                Tuple,
                set,
                Set,
            ) or current_type in (list, tuple, set)

            # Sequence Indexing
            if is_digit and is_sequence:
                log.debug(f"    Part '{part}' is index for sequence.")
                try:
                    index = int(part)
                except ValueError:
                    raise InvalidPathError(f"Invalid index '{part}'") from None
                item_type: Type = Any
                is_tuple = current_origin in (tuple, Tuple) or current_type is tuple
                if current_args:
                    if is_tuple:
                        is_variadic = len(current_args) == 2 and current_args[1] is ...
                        if is_variadic:
                            item_type = current_args[0]
                        elif 0 <= index < len(current_args):
                            item_type = current_args[index]
                        else:
                            raise InvalidPathError(
                                f"Index {index} OOB for {current_type!r}"
                            )
                    else:
                        item_type = current_args[0] if current_args else Any
                log.debug(f"    Sequence item type: {item_type!r}")
                current_type = item_type
                continue

            # Attribute or Dict Key
            log.debug(f"    Part '{part}' trying Attr/Dict Key.")
            cls_for_hints = (
                current_origin
                if current_origin and isinstance(current_origin, type)
                else current_type
            )
            if not isinstance(cls_for_hints, type):
                cls_for_hints = None
            log.debug(f"    Hints Class: {cls_for_hints!r}")

            next_type: Optional[Type] = None
            field_found = False

            # 1. Class Attribute
            if cls_for_hints:
                try:
                    aliases = self._get_field_aliases(cls_for_hints)
                    field_name = aliases.get(part, part)
                    hints = self._get_cached_type_hints(cls_for_hints)
                    log.debug(f"      Hints: {hints!r}")
                    if field_name in hints:
                        field_type = hints[field_name]
                        log.debug(f"      Found hint '{field_name}': {field_type!r}")
                        field_found = True
                        generic_map = self._resolve_generic_type_args(current_type)
                        log.debug(
                            f"      Generic map for {current_type!r}: {generic_map!r}"
                        )
                        if (
                            generic_map
                            and _is_typevar(field_type)
                            and field_type in generic_map
                        ):
                            resolved = generic_map[field_type]
                            log.debug(f"      Resolved {field_type!r} to {resolved!r}")
                            field_type = resolved or field_type
                        next_type = field_type
                    else:
                        log.debug(f"      Field '{field_name}' not in hints.")
                except TypeError as e:
                    raise TypeError(f"Hint error for {cls_for_hints}: {e}") from e

            # 2. Dict Access
            is_dict = current_origin in (dict, Dict, Mapping) or current_type is dict
            if not field_found and is_dict:
                log.debug(f"    Checking dict access for '{part}'.")
                if current_args and len(current_args) == 2:
                    key_type, value_type = current_args
                    # --- FIX: Error message adjusted ---
                    if key_type is not str:
                        raise InvalidPathError(
                            f"Cannot traverse Dict path '{full_path_str}' via attribute/dot access "
                            f"with non-string key type {self._get_type_name(key_type)}."
                        )
                    # --- End Fix ---
                    next_type = value_type
                    field_found = True
                else:
                    next_type = Any
                    field_found = True
                log.debug(f"    Dict access result type: {next_type!r}")

            # 3. Any Propagation
            if not field_found and (current_type is Any or current_origin is Any):
                log.debug("    Propagating Any.")
                next_type = Any
                field_found = True

            # 4. Non-Optional Union Error
            if not field_found and current_origin is Union:
                raise InvalidPathError(
                    f"Cannot access '{part}' on non-Optional Union {current_type!r}"
                )

            # 5. Raise Error if unresolved
            if not field_found:
                p_type_name = self._get_type_name(current_type)
                log.warning(f"    Part '{part}' unresolved. Parent: {p_type_name!r}")
                if is_digit and not is_sequence:
                    msg = f"Cannot apply index '{part}' to non-sequence '{parent_path_str}' (type: {p_type_name})"
                elif cls_for_hints:
                    msg = f"Field '{part}' does not exist in type {cls_for_hints.__name__}"
                else:
                    msg = f"Cannot access '{part}'. Parent '{parent_path_str}' not traversable (type: {p_type_name})"
                raise InvalidPathError(f"{msg}. Path: '{full_path_str}'.")

            current_type = next_type if next_type is not None else Any
            log.debug(f"  End part '{part}'. New type: {current_type!r}")
            if current_type is Any:
                log.debug("    Type is Any, stopping.")
                return Any

        log.debug(f"End traverse '{field_path}'. Final type: {current_type!r}")
        return current_type

    def get_field_type(self, field_path: str) -> Type:
        # ... (implementation unchanged) ...
        log.debug(f"Get field type: '{field_path}'")
        if not field_path:
            raise ValueError("field_path cannot be empty.")
        try:
            return self._traverse_path(field_path)
        except InvalidPathError as e:
            raise InvalidPathError(
                f"{e} in model {self._get_type_name(self.model_type)}"
            ) from e
        except TypeError as e:
            raise TypeError(
                f"Hint error path '{field_path}' in {self.model_type}: {e}"
            ) from e
        except Exception as e:
            tb = traceback.format_exc()
            log.exception(f"Unexpected error traversing path '{field_path}'")
            raise RuntimeError(
                f"Unexpected traverse error path '{field_path}': {e}\n{tb}"
            ) from e

    # --- Validation Logic ---
    def validate_value(
        self, value: Any, expected_type: Type, path: str = "value"
    ) -> None:
        # ... (implementation unchanged) ...
        log.debug(
            f"Validate: {repr(value)} ({type(value)}) vs {expected_type!r} at '{path}'"
        )
        if _is_typevar(expected_type):
            self._validate_typevar(value, expected_type, path)
            return
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        is_opt = origin is Union and any(_is_none_type(t) for t in args)
        if is_opt:
            if value is None:
                log.debug(" Optional valid: None")
                return
            non_none = tuple(t for t in args if not _is_none_type(t))
            expected = non_none[0] if len(non_none) == 1 else Union[non_none]  # type: ignore
            log.debug(f" Optional checking inner: {expected!r}")
            self.validate_value(value, expected, path)
            return
        if expected_type is Any:
            if value is None:
                raise ValueTypeError(
                    self._format_error_message(path, expected_type, value)
                )
            log.debug(" Any valid.")
            return
        if value is None:
            raise ValueTypeError(self._format_error_message(path, expected_type, value))
        if origin is Union:
            self._validate_union(value, args, expected_type, path)
            return
        check_origin = origin or expected_type
        if check_origin in (list, List, set, Set, tuple, Tuple):
            self._validate_collection(value, expected_type, check_origin, args, path)
            return
        if check_origin in (dict, Dict, Mapping):
            self._validate_dict(value, expected_type, check_origin, args, path)
            return
        if isinstance(expected_type, type):
            self._validate_class(value, expected_type, path)
            return
        self._validate_fallback(value, expected_type, path)

    def _validate_typevar(self, value: Any, tv: TypeVar, path: str):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate TypeVar {tv!r}")
        valid = False
        reason = ""
        constraints = getattr(tv, "__constraints__", ())
        bound = getattr(tv, "__bound__", None)
        log.debug(f"    Constraints: {constraints!r}, Bound: {bound!r}")
        if constraints:
            for c in constraints:
                try:
                    self.validate_value(value, c, path)
                    valid = True
                    log.debug(f"    Constraint match: {c!r}")
                    break
                except (ValueTypeError, TypeError) as e:
                    log.debug(f"    Mismatch vs {c!r}: {e}")
                    continue
            if not valid:
                reason = f"no constraint match ({constraints!r})"
        elif bound:
            try:
                self.validate_value(value, bound, path)
                valid = True
                log.debug(f"    Bound match: {bound!r}")
            except (ValueTypeError, TypeError) as e:
                log.debug(f"    Mismatch vs {bound!r}: {e}")
                reason = f"bound {bound!r} fail: {e}"
        else:
            log.debug("    TypeVar unconstrained.")
            valid = True
        if not valid:
            log.warning(f"  TypeVar validation failed: {reason}")
            raise ValueTypeError(
                f"Path '{path}': TypeVar {tv!r} mismatch. Val {repr(value)} ({type(value)}) {reason}."
            )
        log.debug("  TypeVar validation passed.")

    def _validate_union(
        self, value: Any, args: Tuple[Type, ...], union_type: Type, path: str
    ):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate Union {union_type!r}")
        errors = []
        if (
            isinstance(value, bool)
            and int in args
            and bool not in args
            and str not in args
        ):
            raise ValueTypeError(
                f"Path '{path}': bool invalid for Union {union_type!r} allowing int but not bool."
            )
        for t in args:
            try:
                self.validate_value(value, t, path)
                log.debug(f"    Union match: {t!r}")
                return
            except (ValueTypeError, TypeError) as e:
                log.debug(f"    Mismatch vs {t!r}: {e}")
                errors.append(f"  - vs {t!r}: {e}")
            except Exception as e:
                log.exception(f"    Unexpected error validating vs {t!r}")
                errors.append(f"  - Error vs {t!r}: {e}")
        log.warning("  No type in Union matched.")
        raise ValueTypeError(
            f"Path '{path}': Val {repr(value)} no match in {union_type!r}.\nReasons:\n"
            + "\n".join(errors)
        )

    def _validate_collection(
        self,
        value: Any,
        expected_type: Type,
        origin: Type,
        args: Tuple[Type, ...],
        path: str,
    ):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate Collection {expected_type!r}")
        concrete_class = _origin_to_class(origin)
        if not isinstance(value, concrete_class):
            raise ValueTypeError(self._format_error_message(path, expected_type, value))  # type: ignore
        if args:
            if concrete_class is tuple:
                is_variadic = len(args) == 2 and args[1] is ...
                expected_len = None if is_variadic else len(args)
                item_getter = (
                    (lambda i: args[0]) if is_variadic else (lambda i: args[i])
                )
                if expected_len is not None and len(value) != expected_len:
                    raise ValueTypeError(
                        f"Path '{path}': tuple length mismatch {len(value)} vs {expected_len}"
                    )
                for i, item in enumerate(value):
                    self.validate_value(item, item_getter(i), f"{path}[{i}]")
            else:  # List/Set
                item_t = args[0]
                item_path = (
                    f"{path}[{{i}}]"
                    if concrete_class is list
                    else f"{path}.<set_item:{{item!r}}>"
                )
                for i, item in enumerate(value):
                    self.validate_value(item, item_t, item_path.format(i=i, item=item))

    def _validate_dict(
        self,
        value: Any,
        expected_type: Type,
        origin: Type,
        args: Tuple[Type, ...],
        path: str,
    ):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate Dict/Mapping {expected_type!r}")
        concrete_class = _origin_to_class(origin)
        if not isinstance(value, concrete_class):
            raise ValueTypeError(self._format_error_message(path, expected_type, value))  # type: ignore
        if len(args) == 2:
            k_t, v_t = args
            log.debug(f"    Expected K={k_t!r}, V={v_t!r}")
            if k_t is not Any or v_t is not Any:
                for k, item_v in value.items():
                    k_repr = repr(k)
                    if k_t is not Any:
                        try:
                            self.validate_value(k, k_t, f"{path}[key:{k_repr}]")
                        except (ValueTypeError, TypeError) as e:
                            raise ValueTypeError(
                                f"Invalid key {k_repr} in dict path '{path}': {e}"
                            ) from e
                    if v_t is not Any:
                        try:
                            self.validate_value(item_v, v_t, f"{path}[{k_repr}]")
                        except (ValueTypeError, TypeError) as e:
                            raise ValueTypeError(
                                f"Invalid value for key {k_repr} in dict path '{path}': {e}"
                            ) from e

    def _validate_class(self, value: Any, expected_type: Type, path: str):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate Class {expected_type!r}")
        is_struct = (
            hasattr(expected_type, "__annotations__")
            or is_dataclass(expected_type)
            or hasattr(expected_type, "model_fields")
            or hasattr(expected_type, "__fields__")
        )
        if isinstance(value, dict) and is_struct:
            log.debug("    Value is dict, checking structure.")
            cls_hints = get_origin(expected_type) or expected_type
            if not isinstance(cls_hints, type):
                raise ValueTypeError(
                    f"Internal: Expected class for hints, got {cls_hints!r}"
                )
            errors = []
            validated = set()
            try:
                hints = self._get_cached_type_hints(cls_hints)
                aliases = self._get_field_aliases(cls_hints)
                gen_map = self._resolve_generic_type_args(expected_type)
            except Exception as e:
                raise ValueTypeError(f"Prep error for {expected_type!r}: {e}") from e
            for k, f_v in value.items():
                f_name = aliases.get(k, k)
                if f_name in hints:
                    validated.add(f_name)
                    hint = hints[f_name]
                    if gen_map and _is_typevar(hint) and hint in gen_map:
                        hint = gen_map[hint] or hint
                    try:
                        self.validate_value(f_v, hint, f"{path}.{k}")
                    except (ValueTypeError, TypeError) as e:
                        errors.append(str(e))
            for f_name, f_type in hints.items():
                if f_name not in validated:
                    origin = get_origin(f_type)
                    args = get_args(f_type)
                    is_opt = origin is Union and any(_is_none_type(a) for a in args)
                    has_def = False
                    is_pyd = False
                    if (
                        hasattr(cls_hints, "model_fields")
                        and f_name in cls_hints.model_fields
                    ):
                        is_pyd = True
                        has_def = not cls_hints.model_fields[f_name].is_required()
                    elif (
                        hasattr(cls_hints, "__fields__")
                        and f_name in cls_hints.__fields__
                    ):
                        is_pyd = True
                        has_def = not cls_hints.__fields__[f_name].required
                    if (
                        not is_pyd
                        and is_dataclass(cls_hints)
                        and hasattr(cls_hints, "__dataclass_fields__")
                        and f_name in cls_hints.__dataclass_fields__
                    ):
                        dfield = cls_hints.__dataclass_fields__[f_name]
                        has_def = (
                            dfield.default is not MISSING
                            or dfield.default_factory is not MISSING
                        )
                    if not is_opt and not has_def:
                        errors.append(
                            f"Path '{path}': missing required field '{f_name}'"
                        )
            if errors:
                raise ValueTypeError(
                    f"Dict validation failed for {expected_type!r}:\n"
                    + "\n".join(errors)
                )
            log.debug("    Dict structure valid.")
            return
        elif expected_type is int and isinstance(value, bool):
            raise ValueTypeError(self._format_error_message(path, expected_type, value))
        elif isinstance(value, expected_type):
            log.debug("    Direct instance match.")
            return
        else:
            raise ValueTypeError(self._format_error_message(path, expected_type, value))

    def _validate_fallback(self, value: Any, expected_type: Type, path: str):
        # ... (implementation unchanged) ...
        log.debug(f"  Validate Fallback {expected_type!r}")
        try:
            base_type = get_origin(expected_type) or expected_type
            if isinstance(base_type, type):
                if not isinstance(value, base_type):
                    raise ValueTypeError(
                        self._format_error_message(path, expected_type, value)
                    )
                log.debug(f"    Fallback isinstance check passed vs {base_type!r}.")
            else:
                log.debug(
                    f"    Cannot reliable isinstance check for non-class/type {base_type!r}."
                )
        except TypeError as e:
            log.error(
                f"    Fallback isinstance check raised TypeError: {e}", exc_info=True
            )
            raise ValueTypeError(
                f"Path '{path}': Type check failed for value {repr(value)} against expected type {self._get_type_name(expected_type)}. Internal error: {e}"
            ) from e

    def validate_value_for_path(self, field_path: str, value: Any) -> None:
        # ... (implementation unchanged) ...
        log.debug(f"Validate value for path: '{field_path}', value: {repr(value)}")
        try:
            expected = self.get_field_type(field_path)
            self.validate_value(value, expected, field_path)
        except (InvalidPathError, ValueTypeError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Unexpected validation error path '{field_path}': {e}"
            ) from e

    def _is_single_type_numeric(self, t: Type) -> bool:
        # ... (implementation unchanged) ...
        if t is Any or t is bool or not isinstance(t, type):
            return False
        try:
            if issubclass(t, (int, float)):
                return True
        except TypeError:
            pass
        st = getattr(t, "__supertype__", None)
        return bool(st and issubclass(st, (int, float)) and st is not bool)

    def is_field_numeric(self, field_path: str) -> bool:
        # ... (implementation unchanged) ...
        try:
            t = self.get_field_type(field_path)
        except:
            return False
        origin = get_origin(t)
        args = get_args(t)
        if origin is Union:
            return any(
                self._is_single_type_numeric(a) for a in args if not _is_none_type(a)
            )
        return self._is_single_type_numeric(t)

    def get_list_item_type(self, field_path: str) -> Tuple[bool, Type]:
        try:
            t = self.get_field_type(field_path)
        except Exception as e:
            raise InvalidPathError(
                f"List item type error path '{field_path}': {e}"
            ) from e
        origin = get_origin(t)
        args = get_args(t)
        is_list = False
        item_t: Type = Any
        list_t: Optional[Type] = None
        if origin in (list, List) or t is list:
            is_list = True
            list_t = t
        elif origin is Union:
            maybe_list = None
            has_none = False
            non_none = 0
            for a in args:
                if _is_none_type(a):
                    has_none = True
                else:
                    non_none += 1
                    a_origin = get_origin(a)
                    if a_origin in (list, List) or a is list:
                        if maybe_list is None:
                            maybe_list = a
                        else:
                            maybe_list = None
                            break
            if has_none and maybe_list and non_none == 1:
                is_list = True
                list_t = maybe_list
        if is_list and list_t:
            list_args = get_args(list_t)
            item_t = list_args[0] if list_args else Any
        return is_list, item_t
