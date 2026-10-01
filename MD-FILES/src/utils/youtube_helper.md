# Utility Documentation: `src/utils/youtube_helper.py`

## 1. Overview & Purpose
`src/utils/youtube_helper.py` extracts canonical video IDs from various YouTube URL formats and interfaces with `youtube_transcript_api` to retrieve multilingual transcript segments with timestamps.

---

## 2. Key Components & Functions

### `is_youtube_url(url: str) -> bool`
- Checks for `"youtube.com"` or `"youtu.be"`.

### `extract_video_id(url: str) -> str`
- Parses URL schemes:
  - Shortened URLs: `https://youtu.be/<video_id>` -> extracts path.
  - Standard URLs: `https://www.youtube.com/watch?v=<video_id>` -> extracts query parameter `v`.
- Raises `ValueError` if the ID is missing.

### `format_timestamp(seconds: float) -> str`
- Converts seconds into standard video timestamps: `MM:SS` (or `HH:MM:SS` for videos over 1 hour).

### `get_transcript_segments(url: str) -> list[dict[str, str | float]]`
- Queries `YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi", "gu"])`.
- Filters segments up to `MAX_CHARS` (150,000 characters).
- Returns list of segment dictionaries containing:
  `{"text": "...", "start": 12.4, "duration": 4.2, "timestamp": "00:12"}`.

### `get_transcript(url: str) -> str`
- Backward-compatibility function returning the full transcript as a continuous single string.

---

## 3. Connections & Component Mapping

- **Imported By:**
  - `src.components.document_loader: _load_youtube` (loads transcript segments into windowed retrieval chunks).
  - `src.routers.youtube: process_youtube` (extracts video ID for collection deduplication).
  - `src.pipelines.ingestion_pipeline: _get_collection_name` (names collection `youtube_<id>`).

---

## 4. AI & Developer Guidelines
- YouTube transcripts use the languages defined in `config.yaml` (`languages: ["en", "hi", "gu"]`). If no transcript is available in those languages, `YouTubeTranscriptApi` will raise an exception which is handled gracefully by `IngestionPipeline`.
