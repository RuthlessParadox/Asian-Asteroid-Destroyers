import sys

from asian_asteroid_destroyers.genetic_algorithm import (
    clear_dynamic_namespace_cache,
)


def test_clear_dynamic_namespace_cache_purges_attributes() -> None:
    class MockClass:
        pass

    mock_cache: dict[tuple[float, ...], type[object]] = {
        (1.0, -1.0): MockClass,
    }
    prefix = "DynamicMockTest_"
    class_name = "DynamicMockTest_1_0_neg1_0"

    main_module = sys.modules["__main__"]
    setattr(main_module, class_name, MockClass)

    assert hasattr(main_module, class_name)

    clear_dynamic_namespace_cache(mock_cache, prefix)

    assert not hasattr(main_module, class_name)
    assert len(mock_cache) == 0
