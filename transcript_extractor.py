"""
transcript_extractor.py – Optional video transcript extraction
---------------------------------------------------------------
Downloads audio from video URLs via yt-dlp and transcribes with Whisper.
If dependencies are not installed, all functions gracefully return None.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Check if yt-dlp and whisper are importable."""
    try:
        import yt_dlp       # noqa: F401
        import whisper       # noqa: F401
        return True
    except ImportError:
        return False


def is_video_url(url: str) -> bool:
    """Heuristic check for whether a string looks like a video URL."""
    video_domains = [
        "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
        "twitch.tv", "bilibili.com",
    ]
    return any(d in url.lower() for d in video_domains)


# Keep private alias for backward compat
_is_video_url = is_video_url


def extract_transcript(
    url: str,
    output_dir: Path,
    whisper_model_size: str | None = None,
) -> Optional[Path]:
    """Download audio and transcribe a video URL.

    Parameters
    ----------
    url : str
        The video URL (YouTube, Vimeo, etc.)
    output_dir : Path
        Directory to save the transcript .txt file.
    whisper_model_size : str, optional
        Whisper model size. Defaults to ``config.WHISPER_MODEL_SIZE``.

    Returns
    -------
    Path or None
        Path to the transcript file, or None on failure.
    """
    if not is_available():
        logger.warning(
            "Transcript extraction skipped — yt-dlp and/or whisper not installed. "
            "Install with: pip install yt-dlp openai-whisper"
        )
        return None

    try:
        import yt_dlp
        import whisper

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download audio
        audio_path = output_dir / "audio_temp.mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(audio_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }],
            "quiet": True,
            "no_warnings": True,
        }

        logger.info("Downloading audio from %s", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get("title", "unknown")

        if not audio_path.exists():
            logger.warning("Audio download failed for %s", url)
            return None

        # Transcribe with Whisper
        model_size = whisper_model_size or config.WHISPER_MODEL_SIZE
        logger.info("Transcribing with Whisper (%s model)...", model_size)
        model = whisper.load_model(model_size)
        result = model.transcribe(str(audio_path))

        # Save transcript
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_"
                            for c in video_title)[:80]
        transcript_path = output_dir / f"{safe_title}.txt"

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"Transcript: {video_title}\n")
            f.write(f"Source: {url}\n")
            f.write("=" * 60 + "\n\n")

            for segment in result.get("segments", []):
                start = segment.get("start", 0)
                minutes, seconds = divmod(int(start), 60)
                text = segment.get("text", "").strip()
                f.write(f"[{minutes:02d}:{seconds:02d}] {text}\n")

        # Clean up temp audio
        if audio_path.exists():
            audio_path.unlink()

        logger.info("Transcript saved to %s (%d segments)",
                    transcript_path, len(result.get("segments", [])))
        return transcript_path

    except Exception as exc:
        logger.warning("Transcript extraction failed for %s: %s", url, exc)
        return None


def extract_transcripts_batch(
    urls: list[str],
    output_dir: Path,
) -> list[Path]:
    """Extract transcripts for multiple video URLs.

    Returns only the paths that succeeded (skips failures).
    """
    if not is_available():
        logger.warning(
            "Transcript extraction skipped — dependencies not installed."
        )
        return []

    results = []
    for url in urls:
        if _is_video_url(url):
            path = extract_transcript(url, output_dir)
            if path:
                results.append(path)
        else:
            logger.debug("Skipping non-video URL: %s", url)

    return results
