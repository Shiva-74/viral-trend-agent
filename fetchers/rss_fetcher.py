# fetchers/rss_fetcher.py
import feedparser
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost
from config import RSS_FEEDS


class RSSFetcher(BaseFetcher):
    source_name = "rss"

    def fetch(self) -> int:
        total_new = 0
        for category, feed_urls in RSS_FEEDS.items():
            for url in feed_urls:
                posts = self._fetch_feed(url, category)
                saved = self.save_batch(posts)
                total_new += saved
                time.sleep(0.5)
        return total_new

    def _fetch_feed(self, url: str, category: str) -> list[RawPost]:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error(f"RSS parse error for {url}: {e}")
            return []

        posts = []
        for entry in feed.entries[:30]:
            title   = (getattr(entry, "title", "") or "").strip()
            summary = (getattr(entry, "summary", "") or "").strip()
            link    = (getattr(entry, "link",  "") or "").strip()

            if not title:
                continue

            # Geo filter
            if not self.is_western(text=f"{title} {summary}"):
                continue

            # Parse published date
            published_at = None
            if hasattr(entry, "published"):
                try:
                    published_at = parsedate_to_datetime(entry.published)
                    if published_at.tzinfo is None:
                        published_at = published_at.replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            # Determine source name from feed URL
            source_tag = self._source_tag(url)

            uid = self.make_post_id(f"rss_{source_tag}", link or title)
            post = RawPost(
                id           = uid,
                source       = f"rss_{source_tag}",
                category     = category,
                title        = title,
                body         = summary[:1000],
                url          = link,
                published_at = published_at,
            )
            posts.append(post)

        return posts

    def _source_tag(self, url: str) -> str:
        mapping = {
            "techcrunch": "techcrunch",
            "theverge":   "theverge",
            "arstechnica":"ars",
            "bbc":        "bbc",
            "reuters":    "reuters",
            "nytimes":    "nyt",
            "coindesk":   "coindesk",
            "cointelegraph": "cointelegraph",
            "variety":    "variety",
            "deadline":   "deadline",
        }
        for key, tag in mapping.items():
            if key in url.lower():
                return tag
        return "rss"