"""Configuration and package-data resolution for source and wheel installs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterator

import yaml


def runtime_root() -> Path:
    """Return the external working directory used for indexes and operator data."""

    configured = os.getenv("WGLR_RUNTIME_DIR")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def resolve_runtime_path(value: str | Path, *, env_var: str | None = None) -> Path:
    """Resolve a config path outside the installed package."""

    if env_var:
        override = os.getenv(env_var)
        if override:
            return Path(override).expanduser().resolve()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (runtime_root() / path).resolve()


def _packaged_config_resource():
    return files("config").joinpath("config.yaml")


@contextmanager
def packaged_fixture_path() -> Iterator[Path]:
    """Yield the CC0 fixture directory without writing into ``site-packages``."""

    resource = files("data.fixtures")
    with as_file(resource) as path:
        yield Path(path)


@contextmanager
def source_path_for_config(value: str | Path) -> Iterator[Path]:
    """Resolve operator sources, falling back to the bundled CC0 fixture."""

    configured = os.getenv("WGLR_SOURCE_PATH")
    if configured:
        yield Path(configured).expanduser().resolve()
        return
    candidate = resolve_runtime_path(value)
    if candidate.is_dir():
        yield candidate
        return
    normalized = Path(value).as_posix().rstrip("/")
    if normalized in {"data/fixtures", "./data/fixtures"}:
        with packaged_fixture_path() as fixture:
            yield fixture
        return
    yield candidate


def _config_path(config_path: str | None) -> Path | None:
    """Resolve an explicit/runtime config path, or return ``None`` for package data."""

    if config_path is None:
        configured = os.getenv("WGLR_CONFIG")
        if configured:
            config_path = configured
        else:
            local = runtime_root() / "config" / "config.yaml"
            if local.is_file():
                return local
            return None

    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    candidate = runtime_root() / path
    if candidate.is_file():
        return candidate
    # The historical default is also accepted when imported from a wheel.
    if path.as_posix() == "config/config.yaml":
        return None
    return candidate


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load YAML from the runtime directory or bundled package data.

    A checkout prefers ``$WGLR_RUNTIME_DIR/config/config.yaml`` (normally the
    current working directory).  An installed wheel falls back to the read-only
    package resource.  No configuration is copied or written during import.
    """

    path = _config_path(config_path)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    else:
        resource = _packaged_config_resource()
        if not resource.is_file():
            raise FileNotFoundError("packaged config/config.yaml is unavailable")
        with resource.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    return config


def get_config_value(
    key: str,
    default: Any = None,
    config_path: str | None = None,
) -> Any:
    """Return one configuration value, preserving the original helper contract."""

    return load_config(config_path).get(key, default)
