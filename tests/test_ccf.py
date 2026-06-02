"""Tests for the CCF structure-tree ontology utilities."""

import unittest

import pandas as pd

from aind_ephys_utils.ccf import (
    get_descendant_acronyms,
    get_descendant_ids,
    order_regions_by_ccf,
    resolve_regions,
)


def _toy_tree() -> pd.DataFrame:
    """A 4-node ontology: root(1) -> A(2) -> B(3); root(1) -> C(4)."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["root", "Region A", "Region B", "Region C"],
            "acronym": ["root", "A", "B", "C"],
            "parent_structure_id": [None, 1, 2, 1],
            "graph_order": [0, 1, 2, 3],
            "structure_id_path": ["/1/", "/1/2/", "/1/2/3/", "/1/4/"],
            "depth": [0, 1, 2, 1],
        }
    )


class DescendantsTest(unittest.TestCase):
    """Descendant expansion over the ontology."""

    def test_descendant_ids_toy(self) -> None:
        """A(2) yields itself and its child B(3), not its sibling C(4)."""
        self.assertEqual(get_descendant_ids(2, _toy_tree()), {2, 3})

    def test_descendant_acronyms_toy(self) -> None:
        """A expands to {A, B}; a leaf expands to itself."""
        self.assertEqual(get_descendant_acronyms("A", _toy_tree()), {"A", "B"})
        self.assertEqual(get_descendant_acronyms("B", _toy_tree()), {"B"})

    def test_string_equals_singleton_list(self) -> None:
        """Scalar and one-element-list inputs agree."""
        toy = _toy_tree()
        self.assertEqual(
            get_descendant_acronyms("A", toy),
            get_descendant_acronyms(["A"], toy),
        )

    def test_bundled_tree_default(self) -> None:
        """With no ccf_df, the bundled Allen tree expands BLA."""
        desc = get_descendant_acronyms("BLA")
        self.assertTrue({"BLA", "BLAa", "BLAp", "BLAv"} <= desc)


class OrderRegionsTest(unittest.TestCase):
    """Ontology-ordering of region lists."""

    def test_orders_by_ontology(self) -> None:
        """Reordered so ontology-adjacent regions are adjacent."""
        ordered, index_map, matches = order_regions_by_ccf(
            ["C", "B", "A"], _toy_tree()
        )
        self.assertEqual(ordered, ["A", "B", "C"])
        self.assertEqual(index_map, {"A": 0, "B": 1, "C": 2})
        self.assertTrue(all(m.structure_id is not None for m in matches))

    def test_unresolved_sorts_last(self) -> None:
        """An unknown region resolves to None and sorts to the end."""
        ordered, _, matches = order_regions_by_ccf(["BOGUS", "A"], _toy_tree())
        self.assertEqual(ordered[-1], "BOGUS")
        self.assertIsNone(matches[0].structure_id)
        self.assertEqual(matches[1].matched_on, "acronym")

    def test_missing_columns_raise(self) -> None:
        """A tree missing required columns is rejected."""
        bad = _toy_tree().drop(columns=["depth"])
        with self.assertRaises(ValueError):
            order_regions_by_ccf(["A"], bad)


class ResolveRegionsTest(unittest.TestCase):
    """Schema-free region -> structure metadata resolution."""

    def test_resolves_acronym_name_and_unresolved(self) -> None:
        """Acronym, case-insensitive name, and unknown inputs."""
        df = resolve_regions(["A", "region b", "BOGUS"], _toy_tree())
        self.assertEqual(
            list(df.columns),
            ["query", "structure_id", "acronym", "name", "matched_on"],
        )
        self.assertEqual(df["query"].tolist(), ["A", "region b", "BOGUS"])
        # "A" matched as an acronym -> canonical id/acronym/name.
        self.assertEqual(df.loc[0, "structure_id"], 2)
        self.assertEqual(df.loc[0, "acronym"], "A")
        self.assertEqual(df.loc[0, "matched_on"], "acronym")
        # "region b" matched case-insensitively on name.
        self.assertEqual(df.loc[1, "structure_id"], 3)
        self.assertEqual(df.loc[1, "acronym"], "B")
        self.assertEqual(df.loc[1, "matched_on"], "name_ci")
        # Unknown -> missing in every metadata column.
        self.assertTrue(pd.isna(df.loc[2, "structure_id"]))
        self.assertTrue(pd.isna(df.loc[2, "acronym"]))
        self.assertTrue(pd.isna(df.loc[2, "matched_on"]))

    def test_bundled_default(self) -> None:
        """With no ccf_df the bundled tree resolves a real acronym."""
        df = resolve_regions(["BLA"])
        self.assertEqual(df.loc[0, "acronym"], "BLA")
        self.assertFalse(pd.isna(df.loc[0, "structure_id"]))


if __name__ == "__main__":
    unittest.main()
