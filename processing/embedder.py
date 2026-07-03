# processing/embedder.py
# Converts post text into vector embeddings using sentence-transformers.
# Model: all-MiniLM-L6-v2 — fast, small (80MB), accurate enough for
# topic clustering. Runs entirely on CPU, no GPU needed.

import json
import time
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import and_
from datetime import datetime, timezone, timedelta
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models  import RawPost, get_session
from processing.cleaner import build_embedding_text, is_bot_post
from config import EMBEDDING_MODEL

# How many posts to embed in one batch
# Larger = faster but more RAM. 64 is safe for any machine.
BATCH_SIZE = 64

# Only embed posts from the last N hours
# No point embedding old posts that won't affect current trends
EMBED_WINDOW_HOURS = 48


class Embedder:
    """
    Loads the sentence transformer model once, then processes
    all un-embedded posts from the database in batches.
    """

    def __init__(self):
        self.session = get_session()
        self.model   = None   # lazy load — only download on first use

    def _load_model(self):
        """Lazy load the model on first call."""
        if self.model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            logger.info("First load downloads ~80MB — this takes 1-2 minutes once.")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded ✅")

    def get_unembedded_posts(self) -> list[RawPost]:
        """
        Fetch all posts that have not been embedded yet.
        No time window filter — process everything in the DB
        that's still waiting. Ordered newest first so fresh
        content gets embedded before older content.
        """
        posts = (
            self.session.query(RawPost)
            .filter(RawPost.is_embedded == False)
            .order_by(RawPost.fetched_at.desc())
            .limit(2000)   # cap per run to avoid memory issues
            .all()
        )
        return posts

    def embed_posts(self, posts: list[RawPost]) -> int:
        """
        Main method. Takes a list of RawPost objects,
        generates embeddings, saves back to DB.
        Returns count of successfully embedded posts.
        """
        if not posts:
            return 0

        self._load_model()

        # Filter out bot posts and build text for embedding
        valid_posts = []
        texts       = []

        for post in posts:
            # Skip bot posts
            if is_bot_post(post.title + " " + (post.body or "")):
                # Mark as embedded so we don't keep retrying
                post.is_embedded = True
                continue

            text = build_embedding_text(
                title  = post.title,
                body   = post.body or "",
                source = post.source,
            )

            if not text:
                post.is_embedded = True
                continue

            valid_posts.append(post)
            texts.append(text)

        if not texts:
            try:
                self.session.commit()
            except Exception:
                self.session.rollback()
            return 0

        logger.info(f"Embedding {len(texts)} posts in batches of {BATCH_SIZE}...")
        start = time.time()

        embedded_count = 0

        # Process in batches
        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[batch_start: batch_start + BATCH_SIZE]
            batch_posts = valid_posts[batch_start: batch_start + BATCH_SIZE]

            try:
                # encode() returns a numpy array of shape (batch_size, 384)
                embeddings = self.model.encode(
                    batch_texts,
                    batch_size      = BATCH_SIZE,
                    show_progress_bar = False,
                    normalize_embeddings = True,  # L2 normalize for cosine similarity
                )

                for post, embedding in zip(batch_posts, embeddings):
                    # Store as JSON string in the DB
                    post.embedding   = json.dumps(embedding.tolist())
                    post.is_embedded = True
                    embedded_count  += 1

                # Commit each batch so we don't lose everything if interrupted
                self.session.commit()
                logger.debug(
                    f"Embedded batch {batch_start // BATCH_SIZE + 1}/"
                    f"{(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}"
                )

            except Exception as e:
                logger.error(f"Embedding batch error: {e}")
                self.session.rollback()
                # Mark failed posts so we retry next cycle
                for post in batch_posts:
                    post.is_embedded = False

        elapsed = round(time.time() - start, 2)
        logger.info(
            f"Embedded {embedded_count}/{len(posts)} posts in {elapsed}s "
            f"({round(embedded_count/elapsed if elapsed > 0 else 0, 1)} posts/sec)"
        )
        return embedded_count

    def run(self) -> int:
        """Called by scheduler. Embeds all pending posts."""
        posts = self.get_unembedded_posts()
        if not posts:
            logger.info("Embedder: no new posts to embed")
            return 0
        logger.info(f"Embedder: {len(posts)} posts waiting for embedding")
        return self.embed_posts(posts)

    @staticmethod
    def decode_embedding(embedding_str: str) -> Optional[np.ndarray]:
        """
        Decode a stored embedding JSON string back to numpy array.
        Used by the clustering engine.
        """
        if not embedding_str:
            return None
        try:
            return np.array(json.loads(embedding_str), dtype=np.float32)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        Returns value between -1 and 1 (1 = identical topics).
        Used by the viral predictor.
        """
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


if __name__ == "__main__":
    embedder = Embedder()
    count    = embedder.run()
    print(f"\n✅ Embedded {count} posts")

    # Show a sample to verify it worked
    session = get_session()
    sample  = (session.query(RawPost)
               .filter(RawPost.is_embedded == True)
               .first())
    if sample:
        emb = Embedder.decode_embedding(sample.embedding)
        print(f"\nSample post: {sample.title[:60]}")
        print(f"Embedding shape: {emb.shape}")
        print(f"Embedding norm: {np.linalg.norm(emb):.4f} (should be ~1.0)")
        print(f"First 5 values: {emb[:5].tolist()}")