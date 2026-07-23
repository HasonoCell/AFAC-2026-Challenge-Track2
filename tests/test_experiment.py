from __future__ import annotations

import unittest

from afac_pipeline.experiment import _assign_family_grouped_splits, _family_key


class ExperimentSplitTest(unittest.TestCase):
    def test_family_grouped_split_has_no_leakage_and_hits_targets(self) -> None:
        families = ["large"] * 4 + ["pair-a"] * 2 + ["pair-b"] * 2 + ["one-a", "one-b"]
        rows = [
            {
                "file_name": f"sample-{index}.jpg",
                "family": family,
                "gt_tables": index % 3,
                "gt_length": 100 + index * 25,
                "pixels": 1_000 + index * 100,
            }
            for index, family in enumerate(families)
        ]

        splits = _assign_family_grouped_splits(
            rows,
            targets={"dev": 2, "validation": 2, "rest": 6},
        )

        self.assertEqual({name: len(items) for name, items in splits.items()}, {
            "dev": 2,
            "validation": 2,
            "rest": 6,
        })
        family_owners: dict[str, set[str]] = {}
        for split_name, items in splits.items():
            for item in items:
                family_owners.setdefault(str(item["family"]), set()).add(split_name)
        self.assertTrue(all(len(owners) == 1 for owners in family_owners.values()))

    def test_table_family_key_masks_values_but_keeps_schema(self) -> None:
        first = (
            "<table><tr><th>Age</th><th>Cash Value</th></tr>"
            "<tr><td>30</td><td>1000</td></tr></table>"
        )
        second = (
            "<table><tr><th>Age</th><th>Cash Value</th></tr>"
            "<tr><td>45</td><td>2500</td></tr></table>"
        )
        other = (
            "<table><tr><th>Year</th><th>Premium</th></tr>"
            "<tr><td>1</td><td>500</td></tr></table>"
        )

        self.assertEqual(_family_key(first), _family_key(second))
        self.assertNotEqual(_family_key(first), _family_key(other))

    def test_family_key_normalizes_fullwidth_title_punctuation(self) -> None:
        halfwidth = "交银康联财富人生两全保险 (2022版B款)\n\n<table></table>"
        fullwidth = "交银康联财富人生两全保险（2023版B款）\n\n<table></table>"

        self.assertEqual(_family_key(halfwidth), _family_key(fullwidth))


if __name__ == "__main__":
    unittest.main()
