from __future__ import annotations

import numpy as np

from calico.selection.patch_match import consistent_shares, pair_share, patch_shares


def unit_rows(rng: np.random.Generator, n: int, d: int = 32) -> np.ndarray:
    rows = rng.normal(size=(n, d))
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float16)


def test_a_photo_shares_every_patch_with_itself_in_any_order() -> None:
    ref = unit_rows(np.random.default_rng(0), 64)
    shuffled = ref[np.random.default_rng(1).permutation(64)]
    shares = patch_shares(np.stack([ref, shuffled]), ref[None])
    assert shares.dtype == np.float32
    assert np.allclose(shares, 1.0)


def test_the_share_counts_the_patches_the_reference_also_has() -> None:
    rng = np.random.default_rng(2)
    ref = unit_rows(rng, 64)
    half = np.concatenate([ref[:32], unit_rows(rng, 32)])
    other = unit_rows(rng, 64)
    same_half, unrelated = patch_shares(np.stack([half, other]), ref[None])
    assert 0.5 <= same_half < 1.0
    assert unrelated < same_half


def test_the_closest_reference_counts() -> None:
    rng = np.random.default_rng(3)
    first, second = unit_rows(rng, 64), unit_rows(rng, 64)
    (share,) = patch_shares(second[None], np.stack([first, second]))
    assert share == 1.0


def test_one_pair_shares_the_same_whichever_comes_first() -> None:
    rng = np.random.default_rng(3)
    a = unit_rows(rng, 64)
    b = np.concatenate([a[:40], unit_rows(rng, 24)])
    assert pair_share(a, b) == pair_share(b, a)
    assert pair_share(a, b) == float(patch_shares(a[None], b[None])[0])
    assert pair_share(a, a) == 1.0


# -- details in their places ----------------------------------------------------------

SIDE = 32


def grid(rng: np.random.Generator) -> np.ndarray:
    """A photo's patches on a ``SIDE`` x ``SIDE`` grid, as ``(row, column, dim)``."""
    return unit_rows(rng, SIDE * SIDE).reshape(SIDE, SIDE, -1)


def flat(cells: np.ndarray) -> np.ndarray:
    return cells.reshape(SIDE * SIDE, -1)


def test_a_photo_is_consistent_with_itself() -> None:
    ref = flat(grid(np.random.default_rng(4)))
    (consistent,) = consistent_shares(ref[None], ref[None])
    assert consistent.dtype == np.float32
    assert consistent == 1.0


def test_details_in_the_wrong_places_share_patches_but_not_their_placement() -> None:
    # A look-alike room: the same windows and doors, arranged differently.
    ref = flat(grid(np.random.default_rng(5)))
    scrambled = ref[np.random.default_rng(6).permutation(len(ref))]
    assert patch_shares(scrambled[None], ref[None])[0] == 1.0
    assert consistent_shares(scrambled[None], ref[None])[0] < 0.05


def test_a_view_re_aimed_sideways_keeps_what_both_frames_hold() -> None:
    rng = np.random.default_rng(7)
    ref = grid(rng)
    moved = np.concatenate([ref[:, 8:], grid(rng)[:, :8]], axis=1)
    # The new columns may match some of the reference by chance, but not in place.
    assert patch_shares(flat(moved)[None], flat(ref)[None])[0] >= 0.75
    assert 0.75 <= consistent_shares(flat(moved)[None], flat(ref)[None])[0] < 0.77


def test_a_closer_shot_of_the_view_keeps_its_details_in_place() -> None:
    rng = np.random.default_rng(8)
    ref = grid(rng)
    closer = ref[8:24, 8:24].repeat(2, axis=0).repeat(2, axis=1)  # twice as close, centred
    (consistent,) = consistent_shares(flat(closer)[None], flat(ref)[None])
    assert consistent == patch_shares(flat(closer)[None], flat(ref)[None])[0] == 0.25


def test_only_the_details_in_place_count() -> None:
    rng = np.random.default_rng(9)
    ref = flat(grid(rng))
    half = ref.copy()
    rows = np.arange(len(ref) // 2, len(ref))
    half[rows] = ref[rows[np.random.default_rng(10).permutation(len(rows))]]
    consistent = consistent_shares(half[None], ref[None])[0]
    assert 0.5 <= consistent < 0.55


def test_the_most_consistent_reference_counts_and_the_answer_is_repeatable() -> None:
    rng = np.random.default_rng(11)
    first, second = flat(grid(rng)), flat(grid(rng))
    photos = np.stack([second, first[np.random.default_rng(12).permutation(len(first))]])
    refs = np.stack([first, second])
    out = consistent_shares(photos, refs)
    assert out[0] == 1.0
    assert out[1] < 0.05
    assert np.array_equal(out, consistent_shares(photos, refs))
