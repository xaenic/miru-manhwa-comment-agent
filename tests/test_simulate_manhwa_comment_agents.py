from simulate_manhwa_comment_agents import (
    advance_state,
    resolve_progress_unit_id,
    select_next_target,
    select_target_chapters,
)


def test_resolve_progress_unit_id_prefers_numeric_chapter_id() -> None:
    assert resolve_progress_unit_id("12345", "chapter-7") == "12345"


def test_resolve_progress_unit_id_falls_back_to_entry_slug() -> None:
    assert resolve_progress_unit_id("chapter-7", "chapter-7") == "chapter-7"


def test_select_target_chapters_uses_latest_count_when_entry_slugs_missing() -> None:
    items = [
        {"entry_slug": "chapter-9", "language": "en"},
        {"entry_slug": "chapter-8", "language": "en"},
        {"entry_slug": "chapter-7", "language": "jp"},
        {"entry_slug": "chapter-2", "language": "en"},
        {"entry_slug": "chapter-1", "language": "en"},
    ]

    assert select_target_chapters(items, language="en", entry_slugs=None, latest_count=2) == [
        {"entry_slug": "chapter-1", "language": "en"},
        {"entry_slug": "chapter-2", "language": "en"},
    ]


def test_select_target_chapters_uses_all_chapters_when_count_is_zero() -> None:
    items = [
        {"entry_slug": "chapter-3", "language": "en"},
        {"entry_slug": "chapter-2", "language": "en"},
        {"entry_slug": "chapter-1", "language": "en"},
    ]

    assert select_target_chapters(items, language="en", entry_slugs=None, latest_count=0) == [
        {"entry_slug": "chapter-1", "language": "en"},
        {"entry_slug": "chapter-2", "language": "en"},
        {"entry_slug": "chapter-3", "language": "en"},
    ]


def test_select_target_chapters_filters_to_requested_entry_slugs() -> None:
    items = [
        {"entry_slug": "chapter-9", "language": "en"},
        {"entry_slug": "chapter-8", "language": "en"},
        {"entry_slug": "chapter-7", "language": "en"},
    ]

    assert select_target_chapters(items, language="en", entry_slugs=["chapter-8", "chapter-7"], latest_count=1) == [
        {"entry_slug": "chapter-8", "language": "en"},
        {"entry_slug": "chapter-7", "language": "en"},
    ]


def test_select_next_target_returns_none_when_all_done() -> None:
    targets = []
    assert select_next_target(targets, state={}, state_key="solo:en", comments_per_chapter=1) is None


def test_advance_state_repeats_same_chapter_until_comment_quota_met() -> None:
    state: dict[str, int | str] = {}
    advance_state(
        state,
        state_key="solo:en",
        chapter_index=0,
        comments_done=0,
        comments_per_chapter=2,
    )
    assert state == {
        "solo:en:chapter_index": 0,
        "solo:en:comments_done": 1,
    }


def test_advance_state_moves_to_next_chapter_after_quota() -> None:
    state: dict[str, int | str] = {}
    advance_state(
        state,
        state_key="solo:en",
        chapter_index=0,
        comments_done=1,
        comments_per_chapter=2,
    )
    assert state == {
        "solo:en:chapter_index": 1,
        "solo:en:comments_done": 0,
    }
