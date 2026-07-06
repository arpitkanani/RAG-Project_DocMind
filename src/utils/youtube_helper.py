import sys
from urllib.parse import parse_qs, urlparse

import yaml
from youtube_transcript_api import YouTubeTranscriptApi

from src.exception import CustomException
from src.logger import logging

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

LANGUAGES = config["youtube"]["language"]
MAX_CHARS = config["youtube"]["max_chars"]


def is_youtube_url(url: str) -> bool:
    """Check whether a string looks like a YouTube URL."""
    try:
        return "youtube.com" in url or "youtu.be" in url
    except Exception as e:
        logging.error("Error while checking YouTube URL: %s", e)
        raise CustomException(e, sys)


def extract_video_id(url: str) -> str:
    """Extract the canonical YouTube video ID."""
    try:
        parsed = urlparse(url)
        if parsed.hostname == "youtu.be":
            video_id = parsed.path[1:]
            logging.info("Extracted video ID: %s from %s", video_id, url)
            return video_id

        if parsed.hostname in ("www.youtube.com", "youtube.com"):
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
            if not video_id:
                raise ValueError(f"No video ID found in URL: {url}")
            logging.info("Extracted video ID: %s from %s", video_id, url)
            return video_id

        raise ValueError(f"Invalid YouTube URL: {url}")
    except Exception as e:
        logging.error("Error while extracting YouTube video ID: %s", e)
        raise CustomException(e, sys)


def format_timestamp(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def get_transcript_segments(url: str) -> list[dict[str, str | float]]:
    """Fetch transcript chunks with timestamps preserved."""
    try:
        video_id = extract_video_id(url)
        logging.info("Fetching transcript for video: %s", video_id)

        transcript_items = YouTubeTranscriptApi().fetch(video_id, languages=LANGUAGES)

        segments: list[dict[str, str | float]] = []
        total_chars = 0
        for item in transcript_items:
            text = (item.text or "").strip()
            if not text:
                continue

            if total_chars >= MAX_CHARS:
                break

            allowed_text = text[: max(MAX_CHARS - total_chars, 0)]
            if not allowed_text:
                break

            segments.append(
                {
                    "text": allowed_text,
                    "start": float(item.start),
                    "duration": float(item.duration),
                    "timestamp": format_timestamp(float(item.start)),
                }
            )
            total_chars += len(allowed_text) + 1

        logging.info(
            "Transcript fetched | segments: %d | chars: %d",
            len(segments),
            total_chars,
        )
        return segments
    except Exception as e:
        raise CustomException(e, sys)


def get_transcript(url: str) -> str:
    """Fetch transcript as plain text for backward compatibility."""
    try:
        return " ".join(str(segment["text"]) for segment in get_transcript_segments(url))
    except Exception as e:
        raise CustomException(e, sys)
