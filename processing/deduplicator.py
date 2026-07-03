# processing/deduplicator.py
# Detects when the same story appears across multiple sources.
# Cross-platform confirmation = strongest virality signal in the system.
# A story on Reddit + HN + RSS simultaneously → near-certain viral.

import numpy as np
from loguru import logger
from datetime import datetime, timezone, timedelta
from sqlalchemy import and_

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models    import RawPost, get_session
from processing.embedder import Embedder

# Similarity threshold for considering two posts "the same story"
# 0.85 = very similar (same topic, possibly different angle)
# 0.92 = nearly identical (same story, different source)
SIMILARITY_THRESHOLD = 0.82

# Only look at posts from the last N hours for cross-platform matching
MATCH_WINDOW_HOURS = 3

# Minimum number of sources for cross-platform bonus
MIN_SOURCES_FOR_BONUS = 2


class Deduplicator:
    """
    Finds posts across different sources that are about the same story.
    Marks them with cross_platform=True and increments cross_platform_count.
    This count is used as a multiplier in the virality scoring engine.
    """

    def __init__(self):
        self.session = get_session()

    def run(self) -> int:
        """
        Main deduplication pass.
        Returns number of cross-platform matches found.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=MATCH_WINDOW_HOURS)

        # Get all embedded posts from the matching window
        posts = (
            self.session.query(RawPost)
            .filter(
                and_(
                    RawPost.is_embedded == True,
                    RawPost.fetched_at  >= cutoff,
                    RawPost.embedding   != "",
                    RawPost.embedding   != None,
                )
            )
            .all()
        )

        if len(posts) < 2:
            logger.info("Deduplicator: not enough posts for matching")
            return 0

        logger.info(f"Deduplicator: checking {len(posts)} posts for cross-platform matches")

        # Decode all embeddings into a matrix for fast comparison
        valid_posts  = []
        embeddings   = []

        for post in posts:
            emb = Embedder.decode_embedding(post.embedding)
            if emb is not None:
                valid_posts.append(post)
                embeddings.append(emb)

        if not embeddings:
            return 0

        emb_matrix = np.array(embeddings, dtype=np.float32)

        # Compute full similarity matrix
        # dot product of normalized vectors = cosine similarity
        sim_matrix = np.dot(emb_matrix, emb_matrix.T)

        match_count = 0
        processed   = set()

        for i, post_a in enumerate(valid_posts):
            if i in processed:
                continue

            # Find all posts similar to post_a from different sources
            similar_indices = np.where(sim_matrix[i] >= SIMILARITY_THRESHOLD)[0]

            # Group by source
            matched_sources = {post_a.source}
            matched_ids     = {post_a.id}

            for j in similar_indices:
                if j == i:
                    continue
                post_b = valid_posts[j]

                # Only count if from a DIFFERENT source
                if post_b.source != post_a.source:
                    matched_sources.add(post_b.source)
                    matched_ids.add(post_b.id)
                    processed.add(j)

            # If matched across 2+ sources — mark all as cross-platform
            if len(matched_sources) >= MIN_SOURCES_FOR_BONUS:
                source_count = len(matched_sources)
                for post_id in matched_ids:
                    # Find and update each matched post
                    matched_post = next(
                        (p for p in valid_posts if p.id == post_id), None
                    )
                    if matched_post:
                        matched_post.cross_platform       = True
                        matched_post.cross_platform_count = source_count

                match_count += 1
                logger.debug(
                    f"Cross-platform match: '{post_a.title[:50]}' "
                    f"across {matched_sources}"
                )

        try:
            self.session.commit()
            logger.info(
                f"Deduplicator: found {match_count} cross-platform stories "
                f"across {len(valid_posts)} posts"
            )
        except Exception as e:
            self.session.rollback()
            logger.error(f"Deduplicator commit error: {e}")

        return match_count


if __name__ == "__main__":
    d = Deduplicator()
    matches = d.run()
    print(f"\n✅ Found {matches} cross-platform matches")

    # Show examples
    session = get_session()
    examples = (session.query(RawPost)
                .filter(RawPost.cross_platform == True)
                .limit(5)
                .all())
    if examples:
        print("\nExample cross-platform stories:")
        for p in examples:
            print(f"  [{p.source}] {p.title[:70]} "
                  f"(across {p.cross_platform_count} sources)")