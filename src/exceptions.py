"""Exceptions raised in `mile` package context."""


class ModuleSandboxError(Exception):
    """Base Module Sandbox Exception."""


class MissingConfigError(ModuleSandboxError):
    """Raised when a model does not have a config field."""


class ModelNotFoundError(ModuleSandboxError):
    """Raised when a model is not registered in the models module."""
