import sys

from .weight_naming import get_clean_weight_string


def clear_dynamic_namespace_cache(
    cache: dict[tuple[float, ...], type[object]],
    prefix: str,
) -> None:
    main_module = sys.modules["__main__"]

    for weight_key in list(cache.keys()):
        clean_weight_string = get_clean_weight_string(weight_key)
        class_name = f"{prefix}{clean_weight_string}"
        if hasattr(main_module, class_name):
            delattr(main_module, class_name)

    cache.clear()
