"""Optional compute acceleration for FireLens search."""

from app.acceleration.protocol import (
    AccelerationBackend,
    AccelerationError,
    CapabilityName,
    RankedScores,
    SymbolCandidate,
)
from app.acceleration.mojo_backend import (
    MojoBackend,
    MojoBackendUnavailableError,
)
from app.acceleration.python_backend import PythonBackend

__all__ = [
    "AccelerationBackend",
    "AccelerationError",
    "CapabilityName",
    "MojoBackend",
    "MojoBackendUnavailableError",
    "PythonBackend",
    "RankedScores",
    "SymbolCandidate",
]
