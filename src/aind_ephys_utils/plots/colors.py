"""Color helpers for brain regions."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Union

_CSV_PATH = Path(__file__).with_name("allen_structure_tree.csv")


def get_color_for_region(region: Union[int, str]) -> str:
    """Return a hex color string for an Allen region id/name/acronym."""
    if isinstance(region, int):
        key = ("id", region)
    else:
        region_str = str(region).strip()
        try:
            key = ("id", int(region_str))
        except ValueError:
            key = ("name", _normalize(region_str))
    return _lookup_color(key)


def _normalize(value: str) -> str:
    """Normalize a name/acronym for matching."""
    return " ".join(value.lower().split())


def _lookup_color(key: Tuple[str, Union[int, str]]) -> str:
    """Resolve a region key to a color hex string."""
    kind, value = key
    ids, names, acronyms = _load_allen_tree()
    if kind == "id":
        color = ids.get(int(value))
    else:
        color = names.get(str(value)) or acronyms.get(str(value))
    if color is None:
        raise ValueError(f"Region {value!r} not found in Allen structure tree.")
    return f"#{color}"


@lru_cache(maxsize=1)
def _load_allen_tree() -> Tuple[Dict[int, str], Dict[str, str], Dict[str, str]]:
    """Load Allen structure metadata into lookup tables."""
    ids: Dict[int, str] = {}
    names: Dict[str, str] = {}
    acronyms: Dict[str, str] = {}
    with _CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            color = (row.get("color_hex_triplet") or "").strip()
            if not color:
                continue
            try:
                rid = int(row["id"])
            except (TypeError, ValueError):
                continue
            ids[rid] = color
            name = _normalize(row.get("name", ""))
            if name:
                names[name] = color
            acronym = _normalize(row.get("acronym", ""))
            if acronym:
                acronyms[acronym] = color
    return ids, names, acronyms


__all__ = ["get_color_for_region"]
