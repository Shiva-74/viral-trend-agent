# fetchers/google_trends_fetcher.py
# Google's old internal API endpoints are dead as of late 2025.
# We use Google Trends' official RSS feed + the trending page RSS,
# both of which are publicly accessible with no authentication.

import feedparser
import requests
import time
import json
from datetime import datetime, timezone
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost

# Google Trends officially exposes RSS feeds for trending searches
# These are the working endpoints in 2026
TRENDS_RSS_FEEDS = {
    "US": "https://trends.google.com/trending/rss?geo=US",
    "GB": "https://trends.google.com/trending/rss?geo=GB",
    "CA": "https://trends.google.com/trending/rss?geo=CA",
    "AU": "https://trends.google.com/trending/rss?geo=AU",
}

# Google News RSS — real-time news trends, no auth
GOOGLE_NEWS_RSS = {
    "top":           "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "tech":          "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGRqTVhZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
    "business":      "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
    "entertainment": "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNREpxYW5RU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
    "science":       "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRFp0Y1RjU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class GoogleTrendsFetcher(BaseFetcher):
    source_name = "google_trends"

    def fetch(self) -> int:
        total_new = 0

        # 1. Google Trends RSS (official trending searches per country)
        posts = self._fetch_trends_rss()
        total_new += self.save_batch(posts)
        time.sleep(2)

        # 2. Google News RSS (real-time news trending by topic)
        posts = self._fetch_google_news_rss()
        total_new += self.save_batch(posts)

        return total_new

    # ── Google Trends RSS ─────────────────────────────────────────────────────

    def _fetch_trends_rss(self) -> list[RawPost]:
        """
        Fetch from Google Trends' official RSS feeds.
        Each entry = a trending search query with traffic volume and related news.
        """
        all_posts = []

        for geo, url in TRENDS_RSS_FEEDS.items():
            posts = self._parse_trends_rss_feed(url, geo)
            all_posts.extend(posts)
            time.sleep(1.5)

        return all_posts

    def _parse_trends_rss_feed(self, url: str, geo: str) -> list[RawPost]:
        try:
            # feedparser handles the RSS parsing
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error(f"Google Trends RSS parse error ({geo}): {e}")
            return []

        if not feed.entries:
            logger.warning(f"Google Trends RSS: no entries for geo={geo}")
            return []

        posts  = []
        now    = datetime.now(timezone.utc)

        for entry in feed.entries[:30]:
            title = (getattr(entry, "title", "") or "").strip()
            if not title:
                continue

            if not self.is_western(text=title, geo=geo):
                continue

            # Google Trends RSS includes traffic data in ht:approx_traffic tag
            traffic_str = ""
            # Try to get traffic from custom namespace tags
            for tag in getattr(entry, "tags", []):
                if "traffic" in tag.get("term", "").lower():
                    traffic_str = tag.get("term", "")
                    break

            # Also check entry summary for traffic info
            summary = (getattr(entry, "summary", "") or "")
            traffic = self._extract_traffic(summary, traffic_str)

            # Extract related news snippets from description
            body = self._clean_html(summary)[:500]

            # Determine category from title
            category = self._guess_category(title)

            # Parse published date
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6],
                                           tzinfo=timezone.utc)
                except Exception:
                    published_at = now

            uid = self.make_post_id(
                f"gtrends_{geo.lower()}",
                f"{title}_{now.strftime('%Y%m%d%H')}"
            )
            posts.append(RawPost(
                id           = uid,
                source       = "google_trends",
                category     = category,
                title        = title,
                body         = body,
                geo          = geo,
                upvotes      = traffic,
                published_at = published_at or now,
            ))

        logger.info(f"Google Trends RSS ({geo}): {len(posts)} trending searches")
        return posts

    # ── Google News RSS ───────────────────────────────────────────────────────

    def _fetch_google_news_rss(self) -> list[RawPost]:
        """
        Google News RSS gives real-time news trending by topic category.
        This is the highest-quality free signal for what news is spreading.
        """
        all_posts = []

        for category, url in GOOGLE_NEWS_RSS.items():
            posts = self._parse_news_rss(url, category)
            all_posts.extend(posts)
            time.sleep(1)

        return all_posts

    def _parse_news_rss(self, url: str, category: str) -> list[RawPost]:
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            logger.error(f"Google News RSS parse error ({category}): {e}")
            return []

        posts = []
        now   = datetime.now(timezone.utc)

        for entry in feed.entries[:20]:
            title  = (getattr(entry, "title", "") or "").strip()
            source = (getattr(entry, "source", {}) or {})
            link   = (getattr(entry, "link",  "") or "").strip()

            if not title:
                continue

            # Google News titles often include source: "Title - Source Name"
            # Clean up the source suffix
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()

            if not self.is_western(text=title):
                continue

            summary = self._clean_html(
                getattr(entry, "summary", "") or ""
            )[:400]

            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published_at = datetime(*entry.published_parsed[:6],
                                           tzinfo=timezone.utc)
                except Exception:
                    published_at = now

            uid = self.make_post_id("gnews", link or title)
            posts.append(RawPost(
                id           = uid,
                source       = "google_trends",
                category     = category if category != "top" else self._guess_category(title),
                title        = title,
                body         = summary,
                url          = link,
                upvotes      = 300,   # Google News = high-signal, static proxy score
                published_at = published_at or now,
            ))

        return posts

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_traffic(self, summary: str, traffic_tag: str) -> int:
        """Extract search volume from Google Trends RSS data."""
        import re
        # Try ht:approx_traffic style values like "500K+"
        for text in [traffic_tag, summary]:
            match = re.search(r"([\d,]+[KMB]?\+?)\s*searches", text, re.I)
            if match:
                return self._parse_volume(match.group(1))
        return 200  # default proxy value

    def _parse_volume(self, vol_str: str) -> int:
        try:
            s = vol_str.upper().replace("+", "").replace(",", "").strip()
            if "B" in s:
                return int(float(s.replace("B", "")) * 1_000_000_000)
            elif "M" in s:
                return int(float(s.replace("M", "")) * 1_000_000)
            elif "K" in s:
                return int(float(s.replace("K", "")) * 1_000)
            return int(s) if s.isdigit() else 200
        except (ValueError, AttributeError):
            return 200

    @staticmethod
    def _clean_html(html: str) -> str:
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _guess_category(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ["ai", "tech", "software", "apple", "google",
                                  "microsoft", "openai", "gpu", "chip", "cyber"]):
            return "tech"
        if any(w in t for w in ["bitcoin", "crypto", "eth", "stock",
                                  "market", "fed", "inflation", "economy"]):
            return "crypto"
        if any(w in t for w in ["movie", "film", "celebrity", "award",
                                  "music", "album", "nba", "nfl", "actor"]):
            return "entertainment"
        if any(w in t for w in ["meme", "viral", "funny", "trend"]):
            return "memes"
        return "news"