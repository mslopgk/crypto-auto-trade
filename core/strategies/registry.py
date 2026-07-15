"""Strategy registry: name -> class. Import-time registration via decorator."""
from __future__ import annotations

from core.strategies.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if not cls.NAME or cls.NAME == "base":
        raise ValueError(f"strategy {cls} must define a unique NAME")
    _REGISTRY[cls.NAME] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    _ensure_loaded()
    return _REGISTRY[name]


def all_strategies() -> dict[str, type[Strategy]]:
    _ensure_loaded()
    return dict(_REGISTRY)


_loaded = False


def _ensure_loaded() -> None:
    """Import all strategy modules so their @register decorators run."""
    global _loaded
    if _loaded:
        return
    from core.strategies import (trend, meanrev, volbreakout, ensemble,  # noqa: F401
                                 funding, round4_a, round4_b, round4_c, round4_d)
    _loaded = True
