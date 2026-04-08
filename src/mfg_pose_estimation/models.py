"""Shared dataclasses extracted from the posterior notebook."""

from .posterior_core import (
    Pose,
    MFGParameters,
    SuperquadricFieldParameters,
    StiffnessParameters,
)

__all__ = [
    "Pose",
    "MFGParameters",
    "SuperquadricFieldParameters",
    "StiffnessParameters",
]
