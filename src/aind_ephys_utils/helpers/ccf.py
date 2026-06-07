"""Allen CCF (Common Coordinate Framework) structure-tree utilities.

Ontology helpers over the Allen 2017 structure tree: expand a region to
all of its descendants, and order regions so anatomical neighbours are
adjacent.  A copy of the structure tree ships with the package (the same
CSV used by :func:`aind_ephys_utils.plots.get_color_for_region`), so the
``ccf_df`` argument is optional; pass a DataFrame or CSV path to use a
different tree.

For region colours, use
:func:`aind_ephys_utils.plots.get_color_for_region`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import ALLEN_STRUCTURE_TREE_PATH


@lru_cache(maxsize=1)
def _load_default_ccf() -> pd.DataFrame:
    """Load the bundled Allen 2017 structure tree as a DataFrame."""
    return pd.read_csv(ALLEN_STRUCTURE_TREE_PATH)


def _as_ccf_df(ccf_df: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    """Resolve the ``ccf_df`` argument (``None`` -> bundled tree)."""
    if ccf_df is None:
        return _load_default_ccf()
    if isinstance(ccf_df, pd.DataFrame):
        return ccf_df
    return pd.read_csv(ccf_df)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def _load_ccf_lookups(ccf_df: pd.DataFrame) -> dict[str, dict]:
    """Build id/acronym/name/parent/order lookups from a CCF tree.

    Expected columns: id, name, acronym, parent_structure_id,
    graph_order, structure_id_path, depth.
    """
    df = ccf_df
    expected = {
        "id",
        "name",
        "acronym",
        "parent_structure_id",
        "graph_order",
        "structure_id_path",
        "depth",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"CCF tree missing columns: {missing}")

    id_to_parent = df.set_index("id")["parent_structure_id"].to_dict()
    id_to_graph_order = (
        df.set_index("id")["graph_order"].fillna(-1).astype(int).to_dict()
    )
    id_to_name = df.set_index("id")["name"].to_dict()
    id_to_acronym = df.set_index("id")["acronym"].to_dict()
    id_to_depth = df.set_index("id")["depth"].fillna(-1).astype(int).to_dict()
    id_to_path = df.set_index("id")["structure_id_path"].to_dict()

    acronym_to_id = {
        a: i for i, a in id_to_acronym.items() if isinstance(a, str)
    }
    acronym_ci_to_id = {
        a.lower(): i for i, a in id_to_acronym.items() if isinstance(a, str)
    }
    name_ci_to_id = {
        n.lower(): i for i, n in id_to_name.items() if isinstance(n, str)
    }

    return {
        "df": df,
        "id_to_parent": id_to_parent,
        "id_to_graph_order": id_to_graph_order,
        "id_to_name": id_to_name,
        "id_to_acronym": id_to_acronym,
        "id_to_depth": id_to_depth,
        "id_to_path": id_to_path,
        "acronym_to_id": acronym_to_id,
        "acronym_ci_to_id": acronym_ci_to_id,
        "name_ci_to_id": name_ci_to_id,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionMatch:
    """Result of resolving a region string against the CCF ontology."""

    query: str
    structure_id: int | None
    matched_on: str | None  # "acronym" | "acronym_ci" | "name_ci" | None


def _resolve_region(
    q: str,
    acronym_to_id: dict[str, int],
    acronym_ci_to_id: dict[str, int],
    name_ci_to_id: dict[str, int],
) -> RegionMatch:
    """Resolve a string (acronym first, then name) to a structure id."""
    if q in acronym_to_id:
        return RegionMatch(q, acronym_to_id[q], "acronym")
    ql = q.lower()
    if ql in acronym_ci_to_id:
        return RegionMatch(q, acronym_ci_to_id[ql], "acronym_ci")
    if ql in name_ci_to_id:
        return RegionMatch(q, name_ci_to_id[ql], "name_ci")
    return RegionMatch(q, None, None)


def _ancestor_chain_ids(
    node_id: int,
    id_to_parent: dict[int, int | None],
    id_to_path: dict[int, str],
) -> tuple[int, ...]:
    """Root->...->node id chain (inclusive).

    Prefer ``structure_id_path``; fall back to parent links.
    """
    path_str = id_to_path.get(node_id, "")
    if isinstance(path_str, str) and path_str.startswith("/"):
        parts = [p for p in path_str.split("/") if p]
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            pass
    chain: list[int] = []
    cur: int | None = node_id
    seen: set[int] = set()
    while cur is not None and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = id_to_parent.get(cur)
    chain.reverse()
    return tuple(chain)


def _ancestor_chain_graph_orders(
    chain_ids: tuple[int, ...], id_to_graph_order: dict[int, int]
) -> tuple[int, ...]:
    """Map an ancestor id chain to its ``graph_order`` tuple."""
    return tuple(id_to_graph_order.get(i, -1) for i in chain_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_descendant_ids(
    structure_id: int,
    ccf_df: pd.DataFrame | str | Path | None = None,
) -> set[int]:
    """Get all descendant structure IDs (including *structure_id*).

    Parameters
    ----------
    structure_id
        The CCF structure ID to find descendants for.
    ccf_df
        CCF structure tree DataFrame or CSV path.  ``None`` uses the
        bundled Allen 2017 tree.

    Returns
    -------
    set[int]
        ``structure_id`` and every structure beneath it in the ontology.
    """
    df = _as_ccf_df(ccf_df)
    descendants = {structure_id}
    for _, row in df.iterrows():
        path_str = row["structure_id_path"]
        if isinstance(path_str, str) and path_str.startswith("/"):
            path_ids = [int(p) for p in path_str.split("/") if p]
            if structure_id in path_ids:
                descendants.add(row["id"])
    return descendants


def get_descendant_acronyms(
    region_acronyms: list[str] | str,
    ccf_df: pd.DataFrame | str | Path | None = None,
) -> set[str]:
    """Get all descendant region acronyms for given acronym(s).

    Expands each acronym to its whole subtree — e.g. ``"BLA"`` pulls in
    ``BLAa``/``BLAp``/``BLAv``.

    Parameters
    ----------
    region_acronyms
        CCF region acronym(s) to find descendants for.
    ccf_df
        CCF structure tree DataFrame or CSV path.  ``None`` uses the
        bundled Allen 2017 tree.

    Returns
    -------
    set[str]
        Acronyms of the requested regions and all their descendants.
    """
    df = _as_ccf_df(ccf_df)
    id_to_acronym = df.set_index("id")["acronym"].to_dict()
    acronym_to_id = {
        v: k for k, v in id_to_acronym.items() if isinstance(v, str)
    }

    if isinstance(region_acronyms, str):
        region_acronyms = [region_acronyms]
    descendant_ids: set[int] = set()
    for acronym in region_acronyms:
        if acronym in acronym_to_id:
            descendant_ids.update(
                get_descendant_ids(acronym_to_id[acronym], df)
            )

    return {
        id_to_acronym[did] for did in descendant_ids if did in id_to_acronym
    }


def order_regions_by_ccf(  # noqa: C901
    regions: Iterable[str],
    ccf_df: pd.DataFrame | str | Path | None = None,
    group_depth: int | None = None,
) -> tuple[list[str], dict[str, int], list[RegionMatch]]:
    """Order region strings by CCF ontology (neighbours adjacent).

    Parameters
    ----------
    regions
        Region identifiers (acronyms preferred; names OK,
        case-insensitive).
    ccf_df
        CCF structure tree DataFrame or CSV path.  ``None`` uses the
        bundled Allen 2017 tree.
    group_depth
        If set, strengthen grouping by the ontology node at/above this
        depth.

    Returns
    -------
    ordered
        Input regions reordered for plotting.
    index_map
        Mapping from region string to its index in *ordered*.
    matches
        Match metadata per input (unresolved items have
        ``structure_id=None`` and sort last).
    """
    lk = _load_ccf_lookups(_as_ccf_df(ccf_df))
    id_to_parent = lk["id_to_parent"]
    id_to_graph_order = lk["id_to_graph_order"]
    id_to_depth = lk["id_to_depth"]
    id_to_path = lk["id_to_path"]

    matches = [
        _resolve_region(
            q,
            lk["acronym_to_id"],
            lk["acronym_ci_to_id"],
            lk["name_ci_to_id"],
        )
        for q in regions
    ]

    big = (10**9,)
    sort_keys: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for m in matches:
        if m.structure_id is None:
            sort_keys.append((big, big))
            continue
        chain_ids = _ancestor_chain_ids(
            m.structure_id, id_to_parent, id_to_path
        )
        chain_go = _ancestor_chain_graph_orders(chain_ids, id_to_graph_order)

        if group_depth is not None:
            depths = tuple(id_to_depth.get(i, -1) for i in chain_ids)
            cutoff = 0
            for idx, d in enumerate(depths):
                if d <= group_depth:
                    cutoff = idx
            prefix = chain_go[: cutoff + 1]
        else:
            prefix = chain_go

        sort_keys.append((prefix, chain_go))

    indices = sorted(
        range(len(matches)),
        key=lambda i: (sort_keys[i][0], sort_keys[i][1]),
    )
    reg_list = list(regions)
    ordered = [reg_list[i] for i in indices]
    index_map = {r: i for i, r in enumerate(ordered)}
    return ordered, index_map, matches


def resolve_regions(
    regions: Iterable[str],
    ccf_df: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Resolve region strings to CCF structure metadata.

    Each input is resolved to a structure id (acronym first, then name,
    case-insensitive) and reported with its canonical id, acronym, and
    name.  Schema-free: returns one tidy row per input so the caller can
    join/map the result however they like — e.g. attach to a units table,
    or add colours via
    :func:`aind_ephys_utils.plots.get_color_for_region`.

    Parameters
    ----------
    regions
        Region identifiers (acronyms preferred; names OK,
        case-insensitive).
    ccf_df
        CCF structure tree DataFrame or CSV path.  ``None`` uses the
        bundled Allen 2017 tree.

    Returns
    -------
    pandas.DataFrame
        One row per input, in input order, with columns ``query``,
        ``structure_id`` (nullable ``Int64``), ``acronym``, ``name``, and
        ``matched_on``.  Unresolved inputs have ``<NA>``/``None``.
    """
    lk = _load_ccf_lookups(_as_ccf_df(ccf_df))
    id_to_acronym = lk["id_to_acronym"]
    id_to_name = lk["id_to_name"]

    rows = []
    for region in regions:
        m = _resolve_region(
            region,
            lk["acronym_to_id"],
            lk["acronym_ci_to_id"],
            lk["name_ci_to_id"],
        )
        sid = m.structure_id
        rows.append(
            {
                "query": region,
                "structure_id": sid,
                "acronym": id_to_acronym.get(sid) if sid is not None else None,
                "name": id_to_name.get(sid) if sid is not None else None,
                "matched_on": m.matched_on,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["query", "structure_id", "acronym", "name", "matched_on"],
    ).astype({"structure_id": "Int64"})
