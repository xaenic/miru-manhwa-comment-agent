from generate_manhwa_ai_comments import normalize_comment_lines, select_page_indices


def test_select_page_indices_uses_requested_page_for_single_image() -> None:
    assert select_page_indices(total_pages=8, sample_pages=1, requested_page=3) == [2]


def test_select_page_indices_spreads_multiple_samples() -> None:
    assert select_page_indices(total_pages=10, sample_pages=3, requested_page=1) == [0, 4, 9]


def test_normalize_comment_lines_strips_numbering() -> None:
    raw = "1. The panel framing is clean.\n2) That expression hits hard.\n- Love the contrast here."
    assert normalize_comment_lines(raw, 3) == [
        "The panel framing is clean.",
        "That expression hits hard.",
        "Love the contrast here.",
    ]
