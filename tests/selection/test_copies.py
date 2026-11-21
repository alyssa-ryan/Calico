from __future__ import annotations

import numpy as np

from calico.selection.copies import COPY_SIMILARITY, EDIT_SHARE, copies_of, copy_of, edited_from
from tests.factories import unit


def test_each_photo_points_at_the_first_copy_of_its_frame() -> None:
    frame, burst, other = unit([[1.0, 0.0], [1.0, 0.023], [0.0, 1.0]])
    assert float(frame @ burst) < COPY_SIMILARITY  # a burst frame, as measured: <= 0.99973
    vectors = np.stack([other, frame, burst, frame, other])
    assert copy_of(vectors).tolist() == [0, 1, 2, 1, 0]


def test_copies_are_judged_on_direction_not_length() -> None:
    vectors = np.asarray([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    assert copy_of(vectors).tolist() == [0, 0]


def test_no_photos_no_copies() -> None:
    assert copy_of(np.zeros((0, 2), dtype=np.float32)).tolist() == []


def _edits(has_camera: list[bool], shares: dict[tuple[int, int], float]) -> list[int]:
    pairs = np.asarray(list(shares), dtype=np.intp).reshape(-1, 2)
    return [int(i) for i in edited_from(has_camera, pairs, np.asarray(list(shares.values())))]


def test_a_photo_without_a_camera_sharing_most_details_with_another_is_its_edit() -> None:
    # 0 original, 1 its edit (camera data stripped), 2 another shot of the view.
    assert _edits([True, False, True], {(0, 1): 0.62, (1, 2): 0.30, (0, 2): 0.30}) == [0, 0, 2]


def test_a_photo_with_a_camera_keeps_its_own_date_however_alike() -> None:
    assert _edits([True, True], {(0, 1): 0.95}) == [0, 1]


def test_below_the_edit_share_a_photo_without_a_camera_is_its_own_shot() -> None:
    assert _edits([True, False], {(0, 1): EDIT_SHARE - 0.01}) == [0, 1]


def test_an_edit_takes_the_photo_it_shares_most_with() -> None:
    assert _edits([True, True, False], {(0, 2): 0.55, (1, 2): 0.80}) == [0, 1, 1]


def test_an_edit_of_an_edit_leads_back_to_the_original() -> None:
    assert _edits([False, True, False], {(0, 2): 0.9, (1, 2): 0.6, (0, 1): 0.3}) == [1, 1, 1]


def test_edits_only_of_each_other_take_the_first() -> None:
    assert _edits([False, False], {(0, 1): 0.7}) == [0, 0]


def test_copies_of_a_set_of_frames_are_found_by_direction() -> None:
    frame, burst, other = unit([[1.0, 0.0], [1.0, 0.023], [0.0, 1.0]])
    found = copies_of(np.stack([other, 3 * frame, burst]), np.stack([frame]))
    assert found.tolist() == [False, True, False]
    assert copies_of(np.stack([frame]), np.zeros((0, 2), dtype=np.float32)).tolist() == [False]
    assert copies_of(np.zeros((0, 2), dtype=np.float32), np.stack([frame])).tolist() == []
