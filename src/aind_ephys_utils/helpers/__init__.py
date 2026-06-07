"""General-purpose helpers for ephys analysis."""

from pathlib import Path

ALLEN_STRUCTURE_TREE_PATH = Path(__file__).with_name(
    "allen_structure_tree.csv"
)

from . import ccf  # noqa: E402,F401

__all__ = ["ALLEN_STRUCTURE_TREE_PATH", "ccf"]
