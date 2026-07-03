# fetchers/reddit_fetcher.py
# Arctic Shift API — free, no auth, reliable Reddit data mirror.
# Base: https://arctic-shift.photon-reddit.com/api/posts/search
# sort=desc returns newest posts first (no sort_field parameter exists)

import time
from datetime import datetime, timezone, timedelta
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api/posts/search"

SUBREDDIT_CATEGORY = {
    "technology": "tech",        "artificial": "tech",
    "MachineLearning": "tech",   "programming": "tech",
    "OpenAI": "tech",            "cybersecurity": "tech",
    "worldnews": "news",         "news": "news",
    "politics": "news",          "europe": "news",
    "unitedkingdom": "news",     "ukpolitics": "news",
    "geopolitics": "news",       "OutOfTheLoop": "news",
    "memes": "memes",            "dankmemes": "memes",
    "funny": "memes",            "me_irl": "memes",
    "AdviceAnimals": "memes",    "shitposting": "memes",
    "CryptoCurrency": "crypto",  "bitcoin": "crypto",
    "ethereum": "crypto",        "investing": "crypto",
    "stocks": "crypto",          "wallstreetbets": "crypto",
    "finance": "crypto",
    "entertainment": "entertainment", "movies": "entertainment",
    "television": "entertainment",    "Music": "entertainment",
    "popculturechat": "entertainment","nba": "entertainment",
    "nfl": "entertainment",
    "AskReddit": "general",      "todayilearned": "general",
    "explainlikeimfive": "general",
}

SUBREDDITS_BY_CATEGORY = {
    "tech":          ["technology", "artificial", "MachineLearning",
                      "programming", "OpenAI", "cybersecurity"],
    "news":          ["worldnews", "news", "politics", "europe",
                      "unitedkingdom", "OutOfTheLoop", "geopolitics"],
    "memes":         ["memes", "dankmemes", "funny", "me_irl"],
    "crypto":        ["CryptoCurrency", "bitcoin", "wallstreetbets",
                      "investing", "stocks"],
    "entertainment": ["entertainment", "movies", "television",
                      "Music", "nba", "nfl"],
    "general":       ["AskReddit", "todayilearned"],
}


class RedditFetcher(BaseFetcher):
    source_name = "reddit"

    def fetch(self) -> int:
        total_new = 0
        # Unix timestamp for 24 hours ago
        after_ts = int(
            (datetime.now(timezone.utc) - timedelta(hours=24)).timestamp()
        )

        for category, subreddits in SUBREDDITS_BY_CATEGORY.items():
            for sub_name in subreddits:
                posts = self._query(
                    subreddit = sub_name,
                    category  = category,
                    after_ts  = after_ts,
                    sort      = "desc",
                    limit     = 40,
                )
                total_new += self.save_batch(posts)
                time.sleep(1.5)

        return total_new

    def _query(self, subreddit: str, category: str,
               after_ts: int, sort: str,
               limit: int) -> list[RawPost]:
        """Single Arctic Shift API query."""
        params = {
            "subreddit": subreddit,
            "after":     str(after_ts),
            "sort":      sort,
            "limit":     str(limit),
            "over_18":   "false",
        }

        resp = self.get(ARCTIC_BASE, params=params)
        if not resp:
            return []

        try:
            data = resp.json()
        except ValueError as e:
            logger.error(f"Arctic Shift JSON error for r/{subreddit}: {e}")
            return []

        error = data.get("error")
        if error:
            logger.warning(f"Arctic Shift error for r/{subreddit}: {error}")
            return []

        items = data.get("data", [])
        if not items:
            logger.debug(f"Arctic Shift: no results for r/{subreddit}")
            return []

        posts = []
        for item in items:
            post = self._item_to_post(item, category)
            if post:
                posts.append(post)

        # Sort client-side by score so highest-engagement posts
        # get processed first by the clustering engine
        posts.sort(key=lambda p: p.upvotes, reverse=True)

        logger.debug(
            f"Arctic Shift r/{subreddit}: {len(posts)} posts "
            f"(top score: {posts[0].upvotes if posts else 0})"
        )
        return posts

    def _item_to_post(self, item: dict,
                       category: str) -> RawPost | None:
        try:
            title     = (item.get("title") or "").strip()
            post_id   = item.get("id") or ""
            subreddit = item.get("subreddit") or ""
            upvotes   = item.get("score", 0) or 0
            comments  = item.get("num_comments", 0) or 0

            if not title or not post_id:
                return None

            # Skip NSFW
            if item.get("over_18", False):
                return None

            # No score filter — Arctic Shift returns very recent posts
            # which haven't accumulated votes yet. The virality scorer
            # handles quality ranking downstream using growth rate.

            # Geo filter
            if not self.is_western(text=title, subreddit=subreddit):
                return None

            body = (item.get("selftext") or "").strip()
            if body in ("[removed]", "[deleted]"):
                body = ""

            resolved_category = SUBREDDIT_CATEGORY.get(subreddit, category)

            created_utc  = item.get("created_utc")
            published_at = (
                datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                if created_utc else None
            )

            permalink = item.get("permalink", "")
            url = (
                f"https://reddit.com{permalink}"
                if permalink else
                f"https://reddit.com/r/{subreddit}/comments/{post_id}/"
            )

            upvote_ratio = float(item.get("upvote_ratio") or 1.0)

            uid = self.make_post_id("reddit", post_id)
            return RawPost(
                id           = uid,
                source       = "reddit",
                category     = resolved_category,
                title        = title,
                body         = body[:2000],
                url          = url,
                author       = item.get("author", "") or "",
                upvotes      = upvotes,
                comments     = comments,
                upvote_ratio = upvote_ratio,
                subreddit    = subreddit,
                published_at = published_at,
            )

        except Exception as e:
            logger.debug(f"Arctic Shift item parse error: {e}")
            return None