from typing import Iterable
import unittest


def coalesce_ranges(
    ranges: Iterable[tuple[int, int]], *, limit: int | None = None
) -> list[tuple[int, int]]:
    """
    Coalesce half-open byte ranges [start, end), merging overlapping or
    touching ranges. Input is consumed once and is not mutated.

    Raises ValueError for invalid endpoints, malformed pairs, or an invalid
    limit. Empty ranges are discarded.
    """
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative exact integer")

    validated: list[tuple[int, int]] = []

    for item in ranges:
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed range pair: {item!r}") from exc

        if type(start) is not int or type(end) is not int:
            raise ValueError(
                f"range endpoints must be exact integers: {item!r}"
            )

        if start < 0 or end < start:
            raise ValueError(f"invalid range: {item!r}")

        if limit is not None and end > limit:
            raise ValueError(f"range end exceeds limit {limit}: {item!r}")

        if start != end:
            validated.append((start, end))

    validated.sort(key=lambda pair: pair[0])

    merged: list[tuple[int, int]] = []
    for start, end in validated:
        if not merged:
            merged.append((start, end))
            continue

        current_start, current_end = merged[-1]
        if start <= current_end:
            if end > current_end:
                merged[-1] = (current_start, end)
        else:
            merged.append((start, end))

    return merged


class TestCoalesceRanges(unittest.TestCase):
    def test_unsorted_overlap(self) -> None:
        self.assertEqual(
            coalesce_ranges([(8, 12), (0, 4), (4, 9), (20, 20), (30, 32)]),
            [(0, 12), (30, 32)],
        )
        self.assertEqual(
            coalesce_ranges([(5, 10), (1, 6), (8, 12)]),
            [(1, 12)],
        )

    def test_adjacency(self) -> None:
        self.assertEqual(
            coalesce_ranges([(1, 2), (2, 3), (3, 4)]),
            [(1, 4)],
        )
        self.assertEqual(
            coalesce_ranges([(4, 8), (0, 4), (8, 10)]),
            [(0, 10)],
        )

    def test_disjoint_and_empty_ranges(self) -> None:
        self.assertEqual(
            coalesce_ranges([(0, 1), (2, 3), (4, 4), (5, 6)]),
            [(0, 1), (2, 3), (5, 6)],
        )
        self.assertEqual(coalesce_ranges([]), [])

    def test_duplicates(self) -> None:
        self.assertEqual(
            coalesce_ranges([(1, 3), (1, 3), (2, 4), (2, 2)]),
            [(1, 4)],
        )

    def test_generator(self) -> None:
        generator = (
            (start, end) for start, end in [(5, 7), (1, 2), (2, 3), (10, 10)]
        )
        self.assertEqual(
            coalesce_ranges(generator),
            [(1, 3), (5, 7)],
        )

    def test_input_preservation(self) -> None:
        data = [(3, 5), (1, 2), (2, 4)]
        original = list(data)

        result = coalesce_ranges(data)

        self.assertEqual(data, original)
        self.assertEqual(result, [(1, 5)])

    def test_limit_zero(self) -> None:
        self.assertEqual(coalesce_ranges([(0, 0)], limit=0), [])
        self.assertEqual(coalesce_ranges([(0, 0), (0, 0)], limit=0), [])

        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=0)

    def test_limit_boundary(self) -> None:
        self.assertEqual(
            coalesce_ranges([(0, 2), (2, 5)], limit=5),
            [(0, 5)],
        )
        self.assertEqual(coalesce_ranges([(5, 5)], limit=5), [])

    def test_limit_overflow(self) -> None:
        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1), (2, 6)], limit=5)

    def test_reversed_and_negative_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            coalesce_ranges([(5, 4)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(-1, 2)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(2, -1)])

    def test_bool_float_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            coalesce_ranges([(True, 2)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(1, False)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(1, 2.0)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(1.0, 2)])
        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=True)

    def test_malformed_pairs(self) -> None:
        for bad in [(1,), (1, 2, 3), 5, None, "ab"]:
            with self.assertRaises(ValueError):
                coalesce_ranges([bad])

        with self.assertRaises(ValueError):
            coalesce_ranges([(1, 2), (3,)])

    def test_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=-1)
        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=1.0)
        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=False)


if __name__ == "__main__":
    unittest.main()
