#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from mimetypes import guess_type
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_API_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-2-vision-latest"
DEFAULT_TIMEOUT_SECONDS = 30


class ScriptError(Exception):
    """Raised when the script cannot complete the request."""


@dataclass(slots=True)
class ChapterTarget:
    series_slug: str
    series_title: str
    language: str
    entry_slug: str
    chapter_id: str | None
    chapter_number: str | None
    chapter_title: str | None


def normalize_language_code(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short AI comment ideas from a Miru manhwa chapter image using Grok."
    )
    parser.add_argument("--series-slug", required=True, help="Miru manga/manhwa series slug, for example solo-leveling.r8oo")
    parser.add_argument("--entry-slug", help="Chapter entry slug. Defaults to the latest chapter for the chosen language.")
    parser.add_argument("--chapter-id", help="Optional chapter id for reader requests that need it.")
    parser.add_argument("--language", default="en", help="Chapter language code. Default: en")
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based page number to send to Grok when using one image. Default: 1",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=1,
        help="How many chapter pages to send to Grok. Default: 1",
    )
    parser.add_argument(
        "--comment-count",
        type=int,
        default=5,
        help="How many short comments to request from Grok. Default: 5",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("MIRU_API_BASE_URL", DEFAULT_API_BASE_URL),
        help=f"Miru backend base URL. Default: {DEFAULT_API_BASE_URL}",
    )
    parser.add_argument(
        "--grok-base-url",
        default=os.environ.get("XAI_BASE_URL", DEFAULT_GROK_BASE_URL),
        help=f"xAI API base URL. Default: {DEFAULT_GROK_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("XAI_MODEL", DEFAULT_GROK_MODEL),
        help=f"Grok vision model name. Default: {DEFAULT_GROK_MODEL}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a human-readable summary.",
    )
    return parser.parse_args()


def build_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def request_json(session: requests.Session, url: str, *, timeout: int, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ScriptError(f"Expected JSON object from {url}")
    return payload


def fetch_series_title(session: requests.Session, base_url: str, *, series_slug: str, timeout: int) -> str:
    payload = request_json(
        session,
        build_url(base_url, f"/api/v1/manga/series/{quote(series_slug, safe='')}"),
        timeout=timeout,
    )
    title = payload.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return series_slug


def select_latest_chapter(
    session: requests.Session,
    base_url: str,
    *,
    series_slug: str,
    language: str,
    timeout: int,
) -> tuple[str, str | None, str | None, str]:
    payload = request_json(
        session,
        build_url(base_url, f"/api/v1/manga/series/{quote(series_slug, safe='')}/chapters"),
        timeout=timeout,
    )
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ScriptError("The selected series has no chapter entries.")

    for item in items:
        if not isinstance(item, dict):
            continue
        if normalize_language_code(item.get("language")) != normalize_language_code(language):
            continue
        entry_slug = item.get("entry_slug")
        if isinstance(entry_slug, str) and entry_slug.strip():
            chapter_id = item.get("chapter_id") if isinstance(item.get("chapter_id"), str) else None
            chapter_number = item.get("number") if isinstance(item.get("number"), str) else None
            chapter_title = item.get("title") if isinstance(item.get("title"), str) else None
            return entry_slug.strip(), chapter_id, chapter_number, chapter_title

    available_languages = sorted(
        {
            str(item.get("language")).strip()
            for item in items
            if isinstance(item, dict) and isinstance(item.get("language"), str) and str(item.get("language")).strip()
        }
    )
    available_message = f" Available: {', '.join(available_languages)}." if available_languages else ""
    raise ScriptError(f"No chapter entries were found for language {language!r}.{available_message}")


def load_target(args: argparse.Namespace, session: requests.Session) -> ChapterTarget:
    series_title = fetch_series_title(
        session,
        args.api_base_url,
        series_slug=args.series_slug,
        timeout=args.timeout,
    )
    if args.entry_slug:
        return ChapterTarget(
            series_slug=args.series_slug,
            series_title=series_title,
            language=args.language,
            entry_slug=args.entry_slug,
            chapter_id=args.chapter_id,
            chapter_number=None,
            chapter_title=None,
        )

    entry_slug, chapter_id, chapter_number, chapter_title = select_latest_chapter(
        session,
        args.api_base_url,
        series_slug=args.series_slug,
        language=args.language,
        timeout=args.timeout,
    )
    return ChapterTarget(
        series_slug=args.series_slug,
        series_title=series_title,
        language=args.language,
        entry_slug=entry_slug,
        chapter_id=args.chapter_id or chapter_id,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
    )


def load_chapter_pages(
    session: requests.Session,
    base_url: str,
    *,
    target: ChapterTarget,
    timeout: int,
) -> list[str]:
    params = {"chapterId": target.chapter_id} if target.chapter_id else None
    payload = request_json(
        session,
        build_url(
            base_url,
            f"/api/v1/manga/read/{quote(target.series_slug, safe='')}/{quote(target.language, safe='')}/{quote(target.entry_slug, safe='')}/pages",
        ),
        timeout=timeout,
        params=params,
    )
    chapter_number = payload.get("number")
    chapter_title = payload.get("title")
    if isinstance(chapter_number, str) and chapter_number.strip():
        target.chapter_number = chapter_number.strip()
    if isinstance(chapter_title, str) and chapter_title.strip():
        target.chapter_title = chapter_title.strip()

    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ScriptError("The selected chapter does not contain readable page images.")

    images: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        image = page.get("image")
        if isinstance(image, str) and image.strip():
            images.append(image.strip())

    if not images:
        raise ScriptError("The selected chapter returned pages, but no usable image URLs.")
    return images


def select_page_indices(total_pages: int, sample_pages: int, requested_page: int) -> list[int]:
    if total_pages <= 0:
        raise ScriptError("No chapter pages are available.")

    clamped_sample_pages = max(1, min(sample_pages, total_pages))
    if clamped_sample_pages == 1:
        zero_based = max(0, min(requested_page - 1, total_pages - 1))
        return [zero_based]

    if clamped_sample_pages == total_pages:
        return list(range(total_pages))

    last_index = total_pages - 1
    positions = {
        min(last_index, max(0, round(step * last_index / (clamped_sample_pages - 1))))
        for step in range(clamped_sample_pages)
    }
    return sorted(positions)


def guess_mime_type(image_url: str, content_type: str | None) -> str:
    if content_type:
        return content_type.split(";", 1)[0].strip()
    guessed, _ = guess_type(image_url)
    return guessed or "image/jpeg"


def fetch_image_as_data_url(session: requests.Session, image_url: str, *, timeout: int) -> str:
    response = session.get(image_url, timeout=timeout)
    response.raise_for_status()
    mime_type = guess_mime_type(image_url, response.headers.get("Content-Type"))
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_messages(target: ChapterTarget, image_count: int, comment_count: int, image_data_urls: list[str]) -> list[dict[str, Any]]:
    chapter_bits = []
    if target.chapter_number:
        chapter_bits.append(f"Chapter {target.chapter_number}")
    if target.chapter_title:
        chapter_bits.append(target.chapter_title)
    chapter_label = " - ".join(chapter_bits) if chapter_bits else target.entry_slug

    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Look at these manhwa chapter images and write short reader-style comments.\n"
                f"Series: {target.series_title}\n"
                f"Chapter: {chapter_label}\n"
                f"Images provided: {image_count}\n"
                f"Return exactly {comment_count} comments.\n"
                "Rules:\n"
                "- one comment per line\n"
                "- no numbering\n"
                "- each comment under 140 characters\n"
                "- sound like a casual reader reaction\n"
                "- only describe things visible in the image\n"
                "- do not invent names or plot points if unclear\n"
                "- avoid spoilers and sexual content"
            ),
        }
    ]
    user_content.extend({"type": "image_url", "image_url": {"url": image_data_url}} for image_data_url in image_data_urls)

    return [
        {
            "role": "system",
            "content": (
                "You write concise, human-sounding comments for manga and manhwa readers. "
                "Keep them natural, specific to the visible image, and safe."
            ),
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def extract_text_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ScriptError("Grok did not return any choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ScriptError("Grok returned an invalid choice payload.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ScriptError("Grok did not return a message payload.")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    raise ScriptError("Grok response did not contain text content.")


def normalize_comment_lines(raw_text: str, comment_count: int) -> list[str]:
    comments: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if cleaned:
            comments.append(cleaned)
        if len(comments) >= comment_count:
            break
    return comments


def call_grok(
    session: requests.Session,
    *,
    base_url: str,
    api_key: str,
    model: str,
    target: ChapterTarget,
    image_data_urls: list[str],
    comment_count: int,
    timeout: int,
) -> list[str]:
    response = session.post(
        build_url(base_url, "/chat/completions"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.8,
            "messages": build_messages(target, len(image_data_urls), comment_count, image_data_urls),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ScriptError("Grok returned an invalid JSON payload.")
    raw_text = extract_text_content(payload)
    comments = normalize_comment_lines(raw_text, comment_count)
    if not comments:
        raise ScriptError("Grok returned text, but no usable comment lines were found.")
    return comments


def print_text_output(target: ChapterTarget, selected_image_urls: list[str], comments: list[str]) -> None:
    chapter_bits = []
    if target.chapter_number:
        chapter_bits.append(f"Chapter {target.chapter_number}")
    if target.chapter_title:
        chapter_bits.append(target.chapter_title)

    print(f"Series: {target.series_title}")
    print(f"Series slug: {target.series_slug}")
    print(f"Language: {target.language}")
    print(f"Entry slug: {target.entry_slug}")
    if chapter_bits:
        print(f"Chapter: {' - '.join(chapter_bits)}")
    print(f"Images sent: {len(selected_image_urls)}")
    print("Image URLs:")
    for image_url in selected_image_urls:
        print(f"- {image_url}")
    print("Comments:")
    for comment in comments:
        print(f"- {comment}")


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        print("XAI_API_KEY is required.", file=sys.stderr)
        return 1

    session = requests.Session()

    try:
        target = load_target(args, session)
        page_images = load_chapter_pages(
            session,
            args.api_base_url,
            target=target,
            timeout=args.timeout,
        )
        indices = select_page_indices(len(page_images), args.sample_pages, args.page)
        selected_image_urls = [page_images[index] for index in indices]
        image_data_urls = [
            fetch_image_as_data_url(session, image_url, timeout=args.timeout)
            for image_url in selected_image_urls
        ]
        comments = call_grok(
            session,
            base_url=args.grok_base_url,
            api_key=api_key,
            model=args.model,
            target=target,
            image_data_urls=image_data_urls,
            comment_count=max(1, args.comment_count),
            timeout=args.timeout,
        )
    except requests.HTTPError as exc:
        detail = exc.response.text.strip() if exc.response is not None else str(exc)
        print(f"Request failed: {detail}", file=sys.stderr)
        return 1
    except ScriptError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "series_slug": target.series_slug,
                    "series_title": target.series_title,
                    "language": target.language,
                    "entry_slug": target.entry_slug,
                    "chapter_id": target.chapter_id,
                    "chapter_number": target.chapter_number,
                    "chapter_title": target.chapter_title,
                    "image_urls": selected_image_urls,
                    "comments": comments,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
    else:
        print_text_output(target, selected_image_urls, comments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
