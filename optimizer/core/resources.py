"""The resource vocabulary: parts of a model or its denoising loop that an
implementation controls exclusively, and the check that keeps ids to it.

A resource is claimed, never merely read: reading goes through capabilities,
which any number of implementations may share. Each id here is a resource the
architecture names as exclusively wanted (docs/architecture.md, "Three
techniques, resolved"; investigation §8). Add one only together with the
implementation or evidence that needs it.
"""

from __future__ import annotations

LINEAR_LAYERS = "linear_layers"  # what the model's linear layers compute with
TRUNK = "trunk"  # the transformer trunk's output for a model call
STEP_PREDICTION = "step_prediction"  # the noise prediction a denoising step uses

RESOURCES = frozenset({LINEAR_LAYERS, TRUNK, STEP_PREDICTION})


def check_resource(resource: str) -> None:
    """A misspelled resource would silently hide a conflict, so it is rejected."""
    if resource not in RESOURCES:
        raise ValueError(f"unknown resource {resource!r}")
