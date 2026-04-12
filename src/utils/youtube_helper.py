import sys
import os
import shutil
import yaml
from youtube_transcript_api import YouTubeTranscriptApi
from src.exception import CustomException
from urllib.parse import urlparse, parse_qs
from src.logger import logging
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript


with open("D:\\Langchain Project\\config\\config.yaml") as f:
    config = yaml.safe_load(f)

LANGUAGES = config['youtube']['language']
MAX_CHARS = config['youtube']['max_chars']

def is_youtube_url(url: str) -> bool:
    """check if it's youtube URL or not."""

    try:
        return "youtube.com" in url or "youtu.be" in url
    except Exception as e:
        logging.error(f"Error occurred while checking YouTube URL: {e}")
        raise CustomException(e, sys)# type: ignore
    
def extract_video_id(url:str)->str:
    """Extracts the video ID from a Youtube link provided by the user."""
    try:
        parsed=urlparse(url)
        if parsed.hostname=="youtu.be":
            video_id=parsed.path[1:]
            logging.info(f"Extracted video ID: {video_id} from URL: {url}")
            
            return video_id
        if parsed.hostname in ("www.youtube.com", "youtube.com") :
            query=parse_qs(parsed.query)
            video_id=query.get("v",[None])[0]
            if not video_id:
                raise ValueError(f"No video ID found in URL: {url}")
            logging.info(f"Extracted video ID: {video_id} from URL: {url}")
            return video_id
        raise ValueError(f"Invalid YouTube URL: {url}")
        
    except Exception as e:
        logging.error(f"Error occurred while extracting video ID: {e}")
        raise CustomException(e, sys) # type: ignore
    

def get_transcript(url: str) -> str:
    try:
        video_id = extract_video_id(url)
        logging.info(f"Fetching transcript for video: {video_id}")

        # newer API — create instance first, then call fetch
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(
            video_id,
            languages=LANGUAGES
        )

        # join all text chunks
        full_transcript = " ".join(
            chunk.text for chunk in transcript_list
        )
        # note: chunk.text not chunk["text"]
        # newer version returns objects not dictionaries

        if len(full_transcript) > MAX_CHARS:
            full_transcript = full_transcript[:MAX_CHARS]
            logging.warning(f"Transcript trimmed to {MAX_CHARS} chars")

        logging.info(f"Transcript fetched: {len(full_transcript)} chars")
        return full_transcript

    except Exception as e:
        raise CustomException(e, sys) # type: ignore