# fetchers/youtube_fetcher.py
# YouTube trending page URLs are dead for automated clients in 2026.
# Strategy: use yt-dlp's YouTube search with trending/viral search terms
# per category. This gets us currently viral videos without needing
# the trending page directly.

import subprocess
import json
import time
import sys
from datetime import datetime, timezone
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost

# Search queries designed to surface currently trending content per category.
# "today" and "right now" bias results toward recent viral content.
TRENDING_SEARCHES = {
    "tech": [
        "AI news today",
        "tech news this week",
        "OpenAI news",
        "artificial intelligence viral",
    ],
    "news": [
        "breaking news today",
        "world news right now",
        "news today US",
        "trending news this week",
    ],
    "memes": [
        "trending memes 2026",
        "viral memes this week",
        "funny viral video today",
    ],
    "crypto": [
        "bitcoin news today",
        "crypto news this week",
        "stock market news today",
    ],
    "entertainment": [
        "celebrity news today",
        "viral video today",
        "trending entertainment news",
        "movie trailer this week",
    ],
}

# yt-dlp sort order: relevance catches trending content best
YDL_BASE_ARGS = [
    sys.executable, "-m", "yt_dlp",
    "--dump-json",
    "--flat-playlist",
    "--no-warnings",
    "--quiet",
    "--extractor-args", "youtube:skip=dash,hls",
    "--match-filter", "duration > 60",   # skip shorts/ads
    "--playlist-end", "15",              # 15 per search query
]


class YouTubeFetcher(BaseFetcher):
    source_name = "youtube"

    def __init__(self):
        super().__init__()
        self._verify_ytdlp()

    def _verify_ytdlp(self):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                logger.info(f"yt-dlp version: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"yt-dlp check failed: {e}")

    def fetch(self) -> int:
        all_posts = []

        for category, queries in TRENDING_SEARCHES.items():
            for query in queries:
                posts = self._search_youtube(query, category)
                all_posts.extend(posts)
                time.sleep(2)   # space requests

        # Deduplicate by video ID before saving
        seen   = set()
        unique = []
        for p in all_posts:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)

        logger.info(f"YouTube: {len(unique)} unique videos across all categories")
        return self.save_batch(unique)

    def _search_youtube(self, query: str, category: str) -> list[RawPost]:
        """
        Use yt-dlp to search YouTube for a query.
        ytsearch{N}: prefix tells yt-dlp to search instead of fetch a URL.
        """
        search_url = f"ytsearch15:{query}"

        cmd = YDL_BASE_ARGS + [search_url]

        try:
            result = subprocess.run(
                cmd,
                capture_output = True,
                text           = True,
                timeout        = 45,
                encoding       = "utf-8",
                errors         = "replace",
            )

            if not result.stdout.strip():
                logger.debug(f"YouTube search empty for: {query}")
                return []

            return self._parse_output(result.stdout, category)

        except subprocess.TimeoutExpired:
            logger.warning(f"YouTube yt-dlp timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"YouTube yt-dlp error for '{query}': {e}")
            return []

    def _parse_output(self, output: str, category: str) -> list[RawPost]:
        posts = []
        now   = datetime.now(timezone.utc)

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip non-video entries
            if data.get("_type") in ("playlist", "channel"):
                continue

            title    = (data.get("title") or "").strip()
            video_id = (data.get("id")    or "").strip()

            if not title or not video_id:
                continue

            if not self.is_western(text=title):
                continue

            # Skip very short videos (Shorts are < 60s, already filtered by
            # --match-filter but double-check here)
            duration = data.get("duration") or 0
            if duration and duration < 60:
                continue

            view_count = data.get("view_count")  or 0
            like_count = data.get("like_count")  or 0
            channel    = (data.get("channel") or
                          data.get("uploader") or "")
            description = (data.get("description") or "")[:400]

            # Published timestamp
            upload_date  = data.get("upload_date") or ""
            published_at = now
            if upload_date and len(upload_date) == 8:
                try:
                    published_at = datetime(
                        int(upload_date[:4]),
                        int(upload_date[4:6]),
                        int(upload_date[6:8]),
                        tzinfo=timezone.utc,
                    )
                except ValueError:
                    published_at = now

            uid = self.make_post_id("youtube", video_id)
            posts.append(RawPost(
                id           = uid,
                source       = "youtube",
                category     = category,
                title        = title,
                body         = description,
                url          = f"https://youtube.com/watch?v={video_id}",
                author       = channel,
                upvotes      = min(view_count // 1000, 99999),
                comments     = like_count,
                published_at = published_at,
            ))

        logger.info(f"YouTube search '{category}': {len(posts)} videos")
        return posts