# fetchers/nitter_fetcher.py
# Nitter is dead. This module now fetches from 3 better replacement sources:
#   1. Mastodon public trending (open API, no auth, real-time social signals)
#   2. Reddit r/OutOfTheLoop (people asking about viral things = perfect virality signal)
#   3. Trending topics from Bluesky public API

import time
import requests
from datetime import datetime, timezone
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost

# ── Mastodon ──────────────────────────────────────────────────────────────────
MASTODON_TRENDING_POSTS = "https://mastodon.social/api/v1/trends/statuses?limit=40"
MASTODON_TRENDING_TAGS  = "https://mastodon.social/api/v1/trends/tags?limit=20"
MASTODON_TRENDING_LINKS = "https://mastodon.social/api/v1/trends/links?limit=20"

# ── Bluesky ───────────────────────────────────────────────────────────────────
BLUESKY_TRENDING = "https://public.api.bsky.app/xrpc/app.bsky.unspecced.getTrendingTopics?limit=20"
BLUESKY_FEED     = "https://public.api.bsky.app/xrpc/app.bsky.feed.getTimeline?limit=30"
BLUESKY_SEARCH   = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"

# ── Reddit ────────────────────────────────────────────────────────────────────
OOTL_URL = "https://www.reddit.com/r/OutOfTheLoop/hot.json"
ASKREDDIT_TRENDING = "https://www.reddit.com/r/AskReddit/rising.json"


class NitterFetcher(BaseFetcher):
    """
    Social signals fetcher — replaces the defunct Nitter/Twitter scraper.
    Sources: Mastodon trending, Bluesky trending topics, Reddit social signal subs.
    """
    source_name = "social"

    def fetch(self) -> int:
        total_new = 0

        posts = self._fetch_mastodon_posts()
        total_new += self.save_batch(posts)
        time.sleep(1)

        posts = self._fetch_mastodon_links()
        total_new += self.save_batch(posts)
        time.sleep(1)

        posts = self._fetch_mastodon_tags()
        total_new += self.save_batch(posts)
        time.sleep(1)

        posts = self._fetch_bluesky_trending()
        total_new += self.save_batch(posts)
        time.sleep(1)

        posts = self._fetch_reddit_social()
        total_new += self.save_batch(posts)

        return total_new

    # ── Mastodon ──────────────────────────────────────────────────────────────

    def _fetch_mastodon_posts(self) -> list[RawPost]:
        resp = self.get(MASTODON_TRENDING_POSTS)
        if not resp:
            return []

        posts = []
        now   = datetime.now(timezone.utc)

        try:
            for status in resp.json():
                content = self._strip_html(status.get("content", ""))[:300].strip()

                if not content or len(content) < 15:
                    continue

                if not self.is_western(text=content):
                    continue

                lang = status.get("language", "en") or "en"
                if lang not in ("en", "en-US", "en-GB", None):
                    continue

                favourites = status.get("favourites_count", 0)
                reblogs    = status.get("reblogs_count", 0)
                replies    = status.get("replies_count", 0)
                total_eng  = favourites + reblogs * 2 + replies

                if total_eng < 5:
                    continue

                account  = status.get("account", {})
                username = account.get("acct", "")
                url      = status.get("url", "")
                sid      = status.get("id", "")

                uid = self.make_post_id("mastodon", sid or content[:40])
                posts.append(RawPost(
                    id=uid,
                    source="mastodon",
                    category=self._guess_category(content),
                    title=content[:200],
                    body="",
                    url=url,
                    author=username,
                    upvotes=favourites,
                    comments=replies,
                    published_at=now,
                ))
        except Exception as e:
            logger.error(f"Mastodon posts parse error: {e}")

        return posts

    def _fetch_mastodon_links(self) -> list[RawPost]:
        resp = self.get(MASTODON_TRENDING_LINKS)
        if not resp:
            return []

        posts = []
        now   = datetime.now(timezone.utc)

        try:
            for link in resp.json():
                title       = (link.get("title") or "").strip()
                description = (link.get("description") or "")[:300].strip()
                url         = (link.get("url") or "")
                uses        = link.get("history", [{}])[0].get("uses", "0")

                if not title:
                    continue

                if not self.is_western(text=f"{title} {description}"):
                    continue

                try:
                    uses_int = int(uses)
                except (ValueError, TypeError):
                    uses_int = 0

                if uses_int < 3:
                    continue

                uid = self.make_post_id("mastodon_link", url or title)
                posts.append(RawPost(
                    id=uid,
                    source="mastodon",
                    category=self._guess_category(title),
                    title=title,
                    body=description,
                    url=url,
                    upvotes=uses_int * 10,
                    published_at=now,
                ))
        except Exception as e:
            logger.error(f"Mastodon links parse error: {e}")

        return posts

    def _fetch_mastodon_tags(self) -> list[RawPost]:
        resp = self.get(MASTODON_TRENDING_TAGS)
        if not resp:
            return []

        posts = []
        now   = datetime.now(timezone.utc)

        try:
            for tag in resp.json():
                name = (tag.get("name") or "").strip()
                if not name or len(name) < 3:
                    continue

                history = tag.get("history", [])
                uses = sum(int(h.get("uses", 0)) for h in history[:2])
                accts = sum(int(h.get("accounts", 0)) for h in history[:2])

                if uses < 10:
                    continue

                title = f"#{name} trending"
                if not self.is_western(text=title):
                    continue

                uid = self.make_post_id("mastodon_tag", name)
                posts.append(RawPost(
                    id=uid,
                    source="mastodon",
                    category=self._guess_category(name),
                    title=title,
                    body=f"{uses} posts, {accts} accounts using #{name}",
                    url=f"https://mastodon.social/tags/{name}",
                    upvotes=uses,
                    published_at=now,
                ))
        except Exception as e:
            logger.error(f"Mastodon tags parse error: {e}")

        return posts

    # ── Bluesky ───────────────────────────────────────────────────────────────

    def _fetch_bluesky_trending(self) -> list[RawPost]:
        try:
            resp = requests.get(BLUESKY_TRENDING, timeout=15, headers={
                "User-Agent": "ViralTrendAgent/1.0",
                "Accept": "application/json",
            })
            if resp.status_code != 200:
                logger.warning(f"Bluesky trending: HTTP {resp.status_code}")
                return []

            data = resp.json()
            topics = data.get("topics", [])
        except Exception as e:
            logger.error(f"Bluesky trending fetch error: {e}")
            return []

        posts = []
        now = datetime.now(timezone.utc)

        for topic in topics[:25]:
            topic_text = (topic.get("topic") or topic.get("displayName") or topic.get("name") or "").strip()

            if not topic_text or len(topic_text) < 3:
                continue

            if not self.is_western(text=topic_text):
                continue

            description = (topic.get("description") or "")[:300]
            link = topic.get("link") or f"https://bsky.app/search?q={topic_text.replace(' ', '+')}"

            uid = self.make_post_id("bluesky", topic_text)
            posts.append(RawPost(
                id=uid,
                source="bluesky",
                category=self._guess_category(topic_text),
                title=f"{topic_text} trending on Bluesky",
                body=description,
                url=link,
                upvotes=200,
                published_at=now,
            ))

        return posts

    # ── Reddit Social ─────────────────────────────────────────────────────────

    def _fetch_reddit_social(self) -> list[RawPost]:
        """
        Use PullPush for OutOfTheLoop and AskReddit social signals
        instead of direct Reddit JSON (which now 403s without OAuth).
        """
        PULLPUSH_BASE = "https://api.pullpush.io/reddit/search/submission/"
        SOCIAL_SUBS = [
            ("OutOfTheLoop", "news"),
            ("AskReddit", "general"),
            ("NoStupidQuestions", "general"),
        ]

        posts = []
        for sub_name, cat in SOCIAL_SUBS:
            resp = self.get(PULLPUSH_BASE, params={
                "subreddit": sub_name,
                "sort": "score",
                "sort_type": "score",
                "size": 25,
                "score": ">50",
            })
            if not resp:
                continue

            try:
                items = resp.json().get("data", [])
            except:
                continue

            now = datetime.now(timezone.utc)
            for item in items:
                title = (item.get("title") or "").strip()
                post_id = item.get("id") or ""
                sub = item.get("subreddit") or sub_name

                if not title or not post_id:
                    continue

                if not self.is_western(text=title, subreddit=sub):
                    continue

                created = item.get("created_utc")
                pub_at = datetime.fromtimestamp(float(created), tz=timezone.utc) if created else now

                uid = self.make_post_id("social_reddit", post_id)
                posts.append(RawPost(
                    id=uid,
                    source="social",
                    category=cat,
                    title=title,
                    body=(item.get("selftext") or "")[:500],
                    url=f"https://reddit.com/r/{sub}/comments/{post_id}/",
                    author=item.get("author", ""),
                    upvotes=item.get("score", 0) or 0,
                    comments=item.get("num_comments", 0) or 0,
                    subreddit=sub,
                    published_at=pub_at,
                ))

            time.sleep(1)

        return posts

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_html(html: str) -> str:
        import re
        clean = re.sub(r"<br\s*/?>", " ", html)
        clean = re.sub(r"<[^>]+>", "", clean)
        clean = clean.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        clean = clean.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
        return clean.strip()

    def _guess_category(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["ai","tech","software","apple","google","microsoft","openai","gpu","gpt","llm"]):
            return "tech"
        if any(w in text for w in ["bitcoin","crypto","eth","defi","stock","market","usd","fed"]):
            return "crypto"
        if any(w in text for w in ["movie","film","celebrity","award","music","album","nba","nfl","sport"]):
            return "entertainment"
        if any(w in text for w in ["meme","funny","humor","lol","viral"]):
            return "memes"
        return "news"