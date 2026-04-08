"""LegaLoom-Env — TDS Compliance Environment."""

from .client import LegaloomEnv
from .models import TDSAction, TDSObservation, TDSState

# Aliases for backward compatibility
LegaloomAction = TDSAction
LegaloomObservation = TDSObservation

__all__ = [
    "LegaloomEnv",
    "TDSAction",
    "TDSObservation",
    "TDSState",
    "LegaloomAction",
    "LegaloomObservation",
]
