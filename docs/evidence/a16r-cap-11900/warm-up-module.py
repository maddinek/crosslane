from typing import Iterable, Optional
import unittest


def coalesce_ranges(
    ranges: Iterable[tuple[int, int]],
    *,
    limit: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Coalesce half-open byte ranges [start, end), merging overlaps and adjacency."""

    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be an exact nonnegative integer")

    parsed: list[tuple[int, int]] = []

    for item in ranges:
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise ValueError("each range must be a pair") from exc

        if type(start) is not int or type(end) is not int:
            raise ValueError("range endpoints must be exact integers")

        if start < 0 or start > end:
            raise ValueError("range must satisfy 0 <= start <= end")

        if limit is not None and end > limit:
            raise ValueError("range end exceeds limit")

        if start < end:
            parsed.append((start, end))

    parsed.sort(key=lambda pair: pair[0])

    if not parsed:
        return []

    merged: list[tuple[int, int]] = []
    cur_start, cur_end = parsed[0]

    for start, end in parsed[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end

    merged.append((cur_start, cur_end))
    return merged


class CoalesceRangesTests(unittest.TestCase):
    def test_unsorted_overlap_example(self) -> None:
        self.assertEqual(
            coalesce_ranges([(8, 12), (0, 4), (4, 9), (20, 20), (30, 32)]),
            [(0, 12), (30, 32)],
        )

    def test_adjacency(self) -> None:
        self.assertEqual(coalesce_ranges([(4, 9), (0, 4)]), [(0, 9)])
        self.assertEqual(coalesce_ranges([(0, 4), (4, 9), (9, 10)]), [(0, 10)])

    def test_disjoint_and_empty_ranges(self) -> None:
        self.assertEqual(
            coalesce_ranges([(0, 1), (20, 20), (2, 3)]),
            [(0, 1), (2, 3)],
        )
        self.assertEqual(coalesce_ranges([]), [])

    def test_duplicates(self) -> None:
        self.assertEqual(
            coalesce_ranges([(0, 2), (0, 2), (1, 3)]),
            [(0, 3)],
        )

    def test_generator(self) -> None:
        def make_ranges():
            yield (8, 12)
            yield (0, 4)
            yield (4, 9)

        self.assertEqual(coalesce_ranges(make_ranges()), [(0, 12)])

    def test_input_preservation(self) -> None:
        ranges = [(8, 12), (0, 4), (4, 9)]
        snapshot = list(ranges)

        result = coalesce_ranges(ranges)

        self.assertEqual(ranges, snapshot)
        self.assertIsNot(result, ranges)
        self.assertEqual(result, [(0, 12)])

    def test_limit_zero(self) -> None:
        self.assertEqual(coalesce_ranges([(0, 0)], limit=0), [])

        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 1)], limit=0)

    def test_limit_boundary(self) -> None:
        self.assertEqual(coalesce_ranges([(0, 10)], limit=10), [(0, 10)])

        with self.assertRaises(ValueError):
            coalesce_ranges([(0, 11)], limit=10)

    def test_limit_invalid(self) -> None:
        for bad_limit in (-1, True, 1.0, "10"):
            with self.assertRaises(ValueError):
                coalesce_ranges([], limit=bad_limit)

    def test_reversed_and_negative_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            coalesce_ranges([(5, 4)])

        with self.assertRaises(ValueError):
            coalesce_ranges([(-1, 3)])

        with self.assertRaises(ValueError):
            coalesce_ranges([(0, -1)])

    def test_bool_and_float_endpoints(self) -> None:
        for bad_range in [(True, 2), (1, False), (1.0, 2), (0, 2.5)]:
            with self.assertRaises(ValueError):
                coalesce_ranges([bad_range])

    def test_malformed_pairs(self) -> None:
        for bad_range in [None, (0,), (0, 1, 2), "12"]:
            with self.assertRaises(ValueError):
                coalesce_ranges([bad_range])


if __name__ == "__main__":
    unittest.main()
