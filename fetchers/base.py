# fetchers/base.py
import hashlib
import time
import requests
from datetime import datetime, timezone
from loguru import logger
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COUNTRIES, EXCLUDE_REGIONS
from database.models import RawPost, get_session

# Configure logger
logger.add(
    "logs/agent.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
)

HEADERS = {
    "User-Agent": "ViralTrendAgent/1.0 (research project; contact@example.com)",
    "Accept-Language": "en-US,en;q=0.9",
}


class BaseFetcher:
    """
    Base class for all data source fetchers.
    Handles: HTTP requests with retries, deduplication,
    geo-filtering, and database persistence.
    """

    source_name: str = "base"
    category:    str = "general"

    def __init__(self):
        self.session = get_session()
        self.fetch_count = 0
        self.error_count = 0

    # ── HTTP ──────────────────────────────────────────────────────

    def get(self, url: str, params: dict = None,
            retries: int = 3, backoff: float = 2.0) -> Optional[requests.Response]:
        """GET with exponential backoff retry."""
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 429:
                    wait = backoff ** (attempt + 2)
                    logger.warning(f"{self.source_name} rate limited. Waiting {wait}s")
                    time.sleep(wait)
                elif resp.status_code in (403, 404):
                    logger.warning(f"{self.source_name} {resp.status_code} for {url}")
                    return None
                else:
                    logger.warning(f"{self.source_name} HTTP {resp.status_code} attempt {attempt+1}")
                    time.sleep(backoff ** attempt)
            except requests.exceptions.ConnectionError:
                logger.warning(f"{self.source_name} connection error attempt {attempt+1}")
                time.sleep(backoff ** attempt)
            except requests.exceptions.Timeout:
                logger.warning(f"{self.source_name} timeout attempt {attempt+1}")
                time.sleep(backoff ** attempt)
            except Exception as e:
                logger.error(f"{self.source_name} unexpected error: {e}")
                return None
        logger.error(f"{self.source_name} all {retries} retries failed for {url}")
        return None

    # ── Geo Filter ────────────────────────────────────────────────

    def is_western(self, text: str = "", geo: str = "", subreddit: str = "") -> bool:
        """
        Returns True if content is likely from/about Western countries.
        Filters out Indian/South Asian content aggressively.
        """
        # Explicit geo tag check
        if geo:
            if any(excluded in geo.upper() for excluded in EXCLUDE_REGIONS):
                return False
            if any(target in geo.upper() for target in TARGET_COUNTRIES):
                return True

        # Subreddit geo signals
        WESTERN_SUBS = {
            "worldnews", "news", "politics", "europe", "unitedkingdom",
            "AskUK", "AskAmericans", "canada", "australia", "technology",
            "artificial", "memes", "dankmemes", "entertainment",
            "CryptoCurrency", "bitcoin", "investing", "stocks",
            "popular", "all", "funny", "todayilearned", "science",
        }
        EXCLUDED_SUBS = {
            "india", "hinduism", "bollywood", "cricket", "UPSC",
            "desi", "pakistan", "bangladesh", "srilanka",
        }
        if subreddit:
            sub_lower = subreddit.lower()
            if any(ex in sub_lower for ex in EXCLUDED_SUBS):
                return False
            if any(ws in sub_lower for ws in WESTERN_SUBS):
                return True

        # Text keyword signals (fast pre-filter)
        EXCLUDE_KEYWORDS = [
            "india", "indian", "modi", "bjp", "bollywood", "rupee",
            "pakistan", "bangladesh", "cricket ipl", "bcci",
            "desi", "hindi", "tamil", "telugu", "bengali",
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in EXCLUDE_KEYWORDS):
            return False

        # Default: allow if no strong signal either way
        return True

    # ── Deduplication ─────────────────────────────────────────────

    def make_post_id(self, source: str, unique_str: str) -> str:
        """Generate a stable, unique post ID."""
        return f"{source}:{hashlib.md5(unique_str.encode()).hexdigest()[:16]}"

    def already_exists(self, post_id: str) -> bool:
        return self.session.query(RawPost).filter_by(id=post_id).first() is not None

    # ── Persistence ───────────────────────────────────────────────

    def save_post(self, post: RawPost) -> bool:
        """Save post to DB. Returns True if new, False if duplicate."""
        if self.already_exists(post.id):
            return False
        try:
            self.session.add(post)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            logger.error(f"{self.source_name} DB save error: {e}")
            return False

    def save_batch(self, posts: list[RawPost]) -> int:
        """Save a list of posts. Returns count of new posts saved."""
        new_count = 0
        for post in posts:
            if not self.already_exists(post.id):
                try:
                    self.session.add(post)
                    new_count += 1
                except Exception as e:
                    logger.error(f"Batch add error: {e}")
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"{self.source_name} batch commit error: {e}")
        return new_count

    # ── Interface (override in subclasses) ────────────────────────

    def fetch(self) -> int:
        """
        Main fetch method. Override in each subclass.
        Must return count of new posts saved.
        """
        raise NotImplementedError

    def run(self) -> int:
        """Called by scheduler. Wraps fetch() with logging."""
        logger.info(f"[{self.source_name}] Starting fetch...")
        start = time.time()
        try:
            count = self.fetch()
            elapsed = round(time.time() - start, 2)
            self.fetch_count += count
            logger.info(f"[{self.source_name}] Saved {count} new posts in {elapsed}s "
                        f"(total: {self.fetch_count})")
            return count
        except Exception as e:
            self.error_count += 1
            logger.error(f"[{self.source_name}] Fetch failed: {e}")
            return 0