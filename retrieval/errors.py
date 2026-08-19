"""Errors raised while loading or searching the local index."""


class IndexCorruptionError(RuntimeError):
    """The persistent index is missing, malformed, or fails integrity checks."""
