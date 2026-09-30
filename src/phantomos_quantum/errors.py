"""Explicit error types used by optional and privileged integrations."""


class PhantomError(Exception):
    """Base class for recoverable PhantomOS errors."""


class DependencyUnavailable(PhantomError):
    """An optional integration was requested but its dependency is not installed."""


class PermissionDenied(PhantomError):
    """The host rejected a native input operation."""


class SafetyLockError(PhantomError):
    """An action was blocked by the safety policy or unlock state."""
