# fetchers/hn_fetcher.py
import time
from datetime import datetime, timezone
from loguru import logger
from fetchers.base import BaseFetcher
from database.models import RawPost

HN_BASE   = "https://hacker-news.firebaseio.com/v0"
HN_LISTS  = ["topstories", "newstories", "beststories"]


class HNFetcher(BaseFetcher):
    source_name = "hackernews"
    category    = "tech"

    def fetch(self) -> int:
        story_ids = self._get_story_ids()
        if not story_ids:
            return 0

        posts = []
        for story_id in story_ids:
            post = self._fetch_story(story_id)
            if post:
                posts.append(post)
            time.sleep(0.1)  # HN Firebase API is fast but be polite

        return self.save_batch(posts)

    def _get_story_ids(self) -> list[int]:
        seen = set()
        ids  = []
        for list_name in HN_LISTS:
            resp = self.get(f"{HN_BASE}/{list_name}.json")
            if resp:
                try:
                    for sid in resp.json()[:75]:  # top 75 per list
                        if sid not in seen:
                            seen.add(sid)
                            ids.append(sid)
                except (ValueError, TypeError) as e:
                    logger.error(f"HN list parse error ({list_name}): {e}")
        return ids

    def _fetch_story(self, story_id: int) -> RawPost | None:
        resp = self.get(f"{HN_BASE}/item/{story_id}.json")
        if not resp:
            return None

        try:
            d = resp.json()
        except ValueError:
            return None

        if not d or d.get("type") != "story":
            return None

        title = (d.get("title") or "").strip()
        if not title:
            return None

        score    = d.get("score", 0)
        comments = d.get("descendants", 0)

        # Skip low-signal stories
        if score < 10:
            return None

        # Geo filter on title
        if not self.is_western(text=title):
            return None

        created = d.get("time")
        published_at = (datetime.fromtimestamp(created, tz=timezone.utc)
                        if created else None)

        uid = self.make_post_id("hn", str(story_id))
        return RawPost(
            id           = uid,
            source       = "hackernews",
            category     = "tech",
            title        = title,
            body         = "",
            url          = d.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            author       = d.get("by", ""),
            upvotes      = score,
            comments     = comments,
            upvote_ratio = 1.0,
            published_at = published_at,
        )