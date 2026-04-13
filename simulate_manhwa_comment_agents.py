#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests

from generate_manhwa_ai_comments import (
    ChapterTarget,
    ScriptError,
    call_vision_model,
    fetch_image_as_data_url,
    fetch_series_title,
    load_chapter_pages,
    normalize_language_code,
    select_page_indices,
)


DEFAULT_API_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_PROVIDER = "mangafire"
DEFAULT_INTERVAL_MINUTES = 10
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_STATE_FILE = ".manhwa-agent-state.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Private Miru manhwa agent simulator: create an account, generate Gemini comments from chapter images, and post them."
    )
    parser.add_argument("--series-slug", required=True, help="Miru manga/manhwa series slug.")
    parser.add_argument("--language", default="en", help="Chapter language code. Default: en")
    parser.add_argument(
        "--entry-slug",
        action="append",
        dest="entry_slugs",
        help="Specific chapter entry slug to target. Repeat to target multiple chapters.",
    )
    parser.add_argument(
        "--latest-count",
        type=int,
        default=0,
        help="How many sequential chapters to process when --entry-slug is omitted. 0 means all chapters from the first one onward. Default: 0",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Minutes between cycles. Default: {DEFAULT_INTERVAL_MINUTES}",
    )
    parser.add_argument("--run-once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Optional cycle limit for repeated runs. 0 means no limit.",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based page index to use when --sample-pages=1. Default: 1",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=1,
        help="How many chapter pages to send to the vision model per chapter. Default: 1",
    )
    parser.add_argument(
        "--comment-count",
        type=int,
        default=5,
        help="How many candidate comments to ask the vision model for before selecting one. Default: 5",
    )
    parser.add_argument(
        "--comments-per-chapter",
        type=int,
        default=1,
        help="How many separate users should comment on the same chapter before moving to the next one. Default: 1",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("MIRU_API_BASE_URL", DEFAULT_API_BASE_URL),
        help=f"Miru backend base URL. Default: {DEFAULT_API_BASE_URL}",
    )
    parser.add_argument(
        "--llm-base-url",
        default=(
            os.environ.get("GEMINI_BASE_URL")
            or os.environ.get("GOOGLE_BASE_URL")
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("XAI_BASE_URL")
            or DEFAULT_GEMINI_BASE_URL
        ),
        help=f"OpenAI-compatible vision API base URL. Default: {DEFAULT_GEMINI_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=(
            os.environ.get("GEMINI_MODEL")
            or os.environ.get("GOOGLE_MODEL")
            or os.environ.get("QWEN_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or os.environ.get("XAI_MODEL")
            or DEFAULT_GEMINI_MODEL
        ),
        help=f"Vision model name. Default: {DEFAULT_GEMINI_MODEL}",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"Comment provider field. Default: {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--account-prefix",
        default="miru-agent",
        help="Username prefix for generated accounts. Default: miru-agent",
    )
    parser.add_argument(
        "--password-length",
        type=int,
        default=18,
        help="Generated password length. Default: 18",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and print the selected comments without posting them.",
    )
    parser.add_argument(
        "--state-file",
        default=DEFAULT_STATE_FILE,
        help=f"Progress state file for sequential chapter processing. Default: {DEFAULT_STATE_FILE}",
    )
    return parser.parse_args()


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: int,
    headers: dict[str, str] | None = None,
    json_body: dict[str, object] | None = None,
) -> dict[str, object]:
    response = session.request(method, url, timeout=timeout, headers=headers, json=json_body)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ScriptError(f"Expected JSON object from {url}")
    return payload


def build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def create_account_credentials(prefix: str, password_length: int) -> tuple[str, str]:
    username = f"{prefix}-{secrets.token_hex(5)}"
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_!@#"
    password = "".join(secrets.choice(alphabet) for _ in range(max(10, password_length)))
    return username, password


def resolve_progress_unit_id(chapter_id: str | None, entry_slug: str) -> str:
    if isinstance(chapter_id, str) and chapter_id.strip().isdigit():
        return chapter_id.strip()
    return entry_slug


def list_chapter_items(
    session: requests.Session,
    api_base_url: str,
    *,
    series_slug: str,
    provider: str,
    timeout: int,
) -> list[dict[str, object]]:
    payload = request_json(
        session,
        "GET",
        build_url(
            api_base_url,
            f"/api/v1/manga/series/{quote(series_slug, safe='')}/chapters?provider={quote(provider, safe='')}",
        ),
        timeout=timeout,
    )
    items = payload.get("items")
    if not isinstance(items, list):
        raise ScriptError("Chapter list payload is missing items.")
    return [item for item in items if isinstance(item, dict)]


def select_target_chapters(
    items: list[dict[str, object]],
    *,
    language: str,
    entry_slugs: list[str] | None,
    latest_count: int,
) -> list[dict[str, object]]:
    normalized_language = normalize_language_code(language)
    by_language = [
        item for item in items
        if normalize_language_code(item.get("language")) == normalized_language
    ]
    if entry_slugs:
        selected: list[dict[str, object]] = []
        wanted = [entry_slug.strip() for entry_slug in entry_slugs if entry_slug.strip()]
        by_entry_slug = {
            str(item["entry_slug"]): item
            for item in by_language
            if isinstance(item.get("entry_slug"), str) and str(item["entry_slug"]).strip()
        }
        for entry_slug in wanted:
            item = by_entry_slug.get(entry_slug)
            if item:
                selected.append(item)
        if not selected:
            raise ScriptError("None of the requested entry slugs were found for the selected language.")
        return selected

    selected = list(reversed(by_language))
    if latest_count > 0:
        selected = selected[:latest_count]
    if not selected:
        available_languages = sorted(
            {
                str(item.get("language")).strip()
                for item in items
                if isinstance(item.get("language"), str) and str(item.get("language")).strip()
            }
        )
        available_message = f" Available: {', '.join(available_languages)}." if available_languages else ""
        raise ScriptError(f"No chapters were found for language {language!r}.{available_message}")
    return selected


def build_targets(
    session: requests.Session,
    api_base_url: str,
    *,
    series_slug: str,
    language: str,
    provider: str,
    entry_slugs: list[str] | None,
    latest_count: int,
    timeout: int,
) -> list[ChapterTarget]:
    series_title = fetch_series_title(session, api_base_url, series_slug=series_slug, timeout=timeout)
    items = list_chapter_items(session, api_base_url, series_slug=series_slug, provider=provider, timeout=timeout)
    selected_items = select_target_chapters(items, language=language, entry_slugs=entry_slugs, latest_count=latest_count)
    return [
        ChapterTarget(
            series_slug=series_slug,
            series_title=series_title,
            language=language,
            entry_slug=str(item["entry_slug"]),
            chapter_id=str(item["chapter_id"]).strip() if isinstance(item.get("chapter_id"), str) and str(item["chapter_id"]).strip() else None,
            chapter_number=str(item["number"]).strip() if isinstance(item.get("number"), str) else None,
            chapter_title=str(item["title"]).strip() if isinstance(item.get("title"), str) else None,
        )
        for item in selected_items
        if isinstance(item.get("entry_slug"), str) and str(item["entry_slug"]).strip()
    ]


def load_state(path: Path) -> dict[str, int | str]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScriptError(f"Could not read state file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ScriptError(f"State file {path} must contain a JSON object.")
    return payload


def save_state(path: Path, state: dict[str, int | str]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_state_key(series_slug: str, language: str) -> str:
    return f"{series_slug}:{language}"


def select_next_target(
    targets: list[ChapterTarget],
    *,
    state: dict[str, int | str],
    state_key: str,
    comments_per_chapter: int,
) -> tuple[int, int, ChapterTarget] | None:
    if not targets:
        return None

    chapter_index = state.get(f"{state_key}:chapter_index", 0)
    comments_done = state.get(f"{state_key}:comments_done", 0)

    if not isinstance(chapter_index, int) or chapter_index < 0:
        chapter_index = 0
    if not isinstance(comments_done, int) or comments_done < 0:
        comments_done = 0

    if chapter_index >= len(targets):
        return None

    return chapter_index, comments_done, targets[chapter_index]


def advance_state(
    state: dict[str, int | str],
    *,
    state_key: str,
    chapter_index: int,
    comments_done: int,
    comments_per_chapter: int,
) -> None:
    next_comments_done = comments_done + 1
    next_chapter_index = chapter_index

    if next_comments_done >= max(1, comments_per_chapter):
        next_chapter_index += 1
        next_comments_done = 0

    state[f"{state_key}:chapter_index"] = next_chapter_index
    state[f"{state_key}:comments_done"] = next_comments_done


def register_account(
    session: requests.Session,
    api_base_url: str,
    *,
    username: str,
    password: str,
    timeout: int,
) -> None:
    request_json(
        session,
        "POST",
        build_url(api_base_url, "/api/v1/auth/register"),
        timeout=timeout,
        json_body={"username": username, "password": password},
    )


def sign_in_supabase(
    session: requests.Session,
    *,
    api_base_url: str,
    username: str,
    password: str,
    timeout: int,
) -> str:
    payload = request_json(
        session,
        "POST",
        build_url(api_base_url, "/api/v1/auth/login"),
        timeout=timeout,
        json_body={
            "username": username,
            "password": password,
        },
    )
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ScriptError("Supabase sign-in did not return an access token.")
    return access_token


def choose_comment_text(comments: Iterable[str]) -> str:
    comment_list = [comment.strip() for comment in comments if comment.strip()]
    if not comment_list:
        raise ScriptError("No comment candidates were generated.")
    return secrets.choice(comment_list)


def generate_comment_for_target(
    session: requests.Session,
    *,
    target: ChapterTarget,
    api_base_url: str,
    provider: str,
    llm_base_url: str,
    llm_api_key: str,
    model: str,
    comment_count: int,
    page: int,
    sample_pages: int,
    timeout: int,
) -> tuple[str, list[str]]:
    page_images = load_chapter_pages(session, api_base_url, target=target, provider=provider, timeout=timeout)
    indices = select_page_indices(len(page_images), sample_pages, page)
    selected_image_urls = [page_images[index] for index in indices]
    image_data_urls = [
        fetch_image_as_data_url(session, image_url, timeout=timeout)
        for image_url in selected_image_urls
    ]
    candidate_comments = call_vision_model(
        session,
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=model,
        target=replace(target),
        image_data_urls=image_data_urls,
        comment_count=max(1, comment_count),
        timeout=timeout,
    )
    return choose_comment_text(candidate_comments), selected_image_urls


def post_comment(
    session: requests.Session,
    api_base_url: str,
    *,
    access_token: str,
    provider: str,
    target: ChapterTarget,
    comment_body: str,
    timeout: int,
) -> dict[str, object]:
    return request_json(
        session,
        "POST",
        build_url(api_base_url, "/api/v1/me/comments"),
        timeout=timeout,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json_body={
            "media_type": "manga",
            "provider": provider,
            "series_id": target.series_slug,
            "unit_id": resolve_progress_unit_id(target.chapter_id, target.entry_slug),
            "body": comment_body,
        },
    )


def run_cycle(
    session: requests.Session,
    args: argparse.Namespace,
    *,
    llm_api_key: str,
) -> bool:
    state_file = Path(args.state_file)
    state = load_state(state_file)
    username, password = create_account_credentials(args.account_prefix, args.password_length)
    targets = build_targets(
        session,
        args.api_base_url,
        series_slug=args.series_slug,
        language=args.language,
        provider=args.provider,
        entry_slugs=args.entry_slugs,
        latest_count=args.latest_count,
        timeout=args.timeout,
    )
    state_key = build_state_key(args.series_slug, args.language)
    next_target = select_next_target(
        targets,
        state=state,
        state_key=state_key,
        comments_per_chapter=args.comments_per_chapter,
    )

    if next_target is None:
        print("[done] all selected chapters have been processed")
        return False

    chapter_index, comments_done, target = next_target
    register_account(
        session,
        args.api_base_url,
        username=username,
        password=password,
        timeout=args.timeout,
    )
    access_token = sign_in_supabase(
        session,
        api_base_url=args.api_base_url,
        username=username,
        password=password,
        timeout=args.timeout,
    )

    print(
        f"[cycle] account={username} chapter_index={chapter_index + 1}/{len(targets)} "
        f"repeat={comments_done + 1}/{max(1, args.comments_per_chapter)}"
    )

    comment_body, image_urls = generate_comment_for_target(
        session,
        target=target,
        api_base_url=args.api_base_url,
        provider=args.provider,
        llm_base_url=args.llm_base_url,
        llm_api_key=llm_api_key,
        model=args.model,
        comment_count=args.comment_count,
        page=args.page,
        sample_pages=args.sample_pages,
        timeout=args.timeout,
    )
    unit_id = resolve_progress_unit_id(target.chapter_id, target.entry_slug)
    chapter_label = target.chapter_number or target.entry_slug
    if args.dry_run:
        print(f"[dry-run] chapter={chapter_label} unit_id={unit_id} comment={comment_body}")
        for image_url in image_urls:
            print(f"[dry-run] image={image_url}")
    else:
        payload = post_comment(
            session,
            args.api_base_url,
            access_token=access_token,
            provider=args.provider,
            target=target,
            comment_body=comment_body,
            timeout=args.timeout,
        )
        comment_id = payload.get("id")
        print(f"[posted] chapter={chapter_label} unit_id={unit_id} comment_id={comment_id} comment={comment_body}")

    advance_state(
        state,
        state_key=state_key,
        chapter_index=chapter_index,
        comments_done=comments_done,
        comments_per_chapter=args.comments_per_chapter,
    )
    save_state(state_file, state)
    return True


def main() -> int:
    load_env_file(Path(".env"))
    args = parse_args()

    llm_api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("XAI_API_KEY")
        or ""
    ).strip()

    missing = [
        name
        for name, value in (
            ("GEMINI_API_KEY", llm_api_key),
        )
        if not value
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=os.sys.stderr)
        return 1

    session = requests.Session()
    cycle = 0

    while True:
        cycle += 1
        try:
            should_continue = run_cycle(
                session,
                args,
                llm_api_key=llm_api_key,
            )
        except requests.HTTPError as exc:
            detail = exc.response.text.strip() if exc.response is not None else str(exc)
            print(f"[error] {detail}", file=os.sys.stderr)
            return 1
        except ScriptError as exc:
            print(f"[error] {exc}", file=os.sys.stderr)
            return 1

        if not should_continue:
            return 0
        if args.run_once:
            return 0
        if args.max_cycles > 0 and cycle >= args.max_cycles:
            return 0

        print(f"[sleep] waiting {args.interval_minutes} minutes")
        time.sleep(max(1, args.interval_minutes) * 60)


if __name__ == "__main__":
    raise SystemExit(main())
