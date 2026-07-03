# intelligence/clusterer.py
# Groups embedded posts into topic clusters using HDBSCAN.
# Runs every 10 minutes. Maintains persistent cluster IDs across
# runs so growth rate can be tracked continuously.

import json
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timezone, timedelta
from loguru import logger
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import RawPost, Cluster, Snapshot, get_session
from processing.embedder import Embedder
from config import CLUSTER_WINDOW_HOURS, MIN_CLUSTER_SIZE

# How similar a new cluster centroid must be to an existing one
# to be considered the same ongoing topic (continuity tracking)
CONTINUITY_THRESHOLD = 0.75

# Maximum age of a cluster before it's marked inactive
CLUSTER_MAX_AGE_HOURS = 6


class ClusterEngine:
    """
    Core clustering engine.

    Every run:
    1. Fetches all embedded posts with valid embeddings
    2. Runs HDBSCAN to group them into topics
    3. Matches new clusters to existing ones (continuity)
    4. Labels each cluster with keywords
    5. Updates cluster post counts + category
    6. Saves a snapshot (for growth rate calculation)
    7. Updates post → cluster assignments
    """

    def __init__(self):
        self.session = get_session()

    def run(self) -> int:
        """Main entry point. Returns number of active clusters."""
        logger.info("Clusterer: starting clustering run...")
        start = datetime.now(timezone.utc)

        # 1. Load posts + embeddings
        posts, embeddings = self._load_embeddings()
        if len(posts) < MIN_CLUSTER_SIZE:
            logger.warning(
                f"Clusterer: only {len(posts)} embedded posts — "
                f"need at least {MIN_CLUSTER_SIZE} to cluster"
            )
            return 0

        # 2. Run HDBSCAN
        labels = self._run_hdbscan(embeddings)

        # 3. Build cluster data from labels
        raw_clusters = self._build_raw_clusters(posts, embeddings, labels)
        if not raw_clusters:
            logger.warning("Clusterer: no valid clusters formed")
            return 0

        # 4. Match to existing clusters (continuity) or create new ones
        active_clusters = self._reconcile_clusters(raw_clusters)

        # 5. Save snapshots for growth rate tracking
        self._save_snapshots(active_clusters)

        # 6. Update post → cluster assignments
        self._assign_posts_to_clusters(posts, labels, active_clusters)

        # 7. Mark old clusters inactive
        self._expire_old_clusters()

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        logger.info(
            f"Clusterer: {len(active_clusters)} active clusters "
            f"from {len(posts)} posts in {elapsed:.1f}s"
        )
        return len(active_clusters)

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_embeddings(self) -> tuple[list[RawPost], np.ndarray]:
        """
        Load embedded posts for clustering.
        Takes the most recent 3000 embedded posts with valid embeddings.
        No time-window filter — let post volume determine what's clustered.
        """
        posts = (
            self.session.query(RawPost)
            .filter(
                RawPost.is_embedded == True,
                RawPost.embedding   != "",
                RawPost.embedding   != None,
            )
            .order_by(RawPost.fetched_at.desc())
            .limit(3000)
            .all()
        )

        valid_posts = []
        embeddings  = []
        for post in posts:
            emb = Embedder.decode_embedding(post.embedding)
            if emb is not None:
                valid_posts.append(post)
                embeddings.append(emb)

        if not embeddings:
            logger.warning("Clusterer: no posts with valid embeddings found in DB")
            return [], np.array([])

        emb_matrix = np.array(embeddings, dtype=np.float32)
        # Re-normalize to ensure unit vectors (safety check)
        emb_matrix = normalize(emb_matrix, norm="l2")

        logger.info(f"Clusterer: loaded {len(valid_posts)} posts for clustering")
        return valid_posts, emb_matrix

    # ── HDBSCAN ───────────────────────────────────────────────────────────────

    def _run_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Run HDBSCAN clustering on the embedding matrix.

        Key parameters:
        - min_cluster_size: minimum posts to form a topic cluster
        - min_samples: controls how conservative clustering is
        - metric: euclidean on normalized vectors = cosine distance
        - cluster_selection_method: 'eom' finds natural cluster sizes
        """
        n_posts = len(embeddings)

        # Dynamically adjust min_cluster_size based on data volume
        if n_posts < 200:
            min_size    = 3
            min_samples = 2
        elif n_posts < 1000:
            min_size    = MIN_CLUSTER_SIZE
            min_samples = 3
        else:
            min_size    = max(MIN_CLUSTER_SIZE, n_posts // 200)
            min_samples = 4

        logger.info(
            f"Clusterer: HDBSCAN on {n_posts} posts "
            f"(min_size={min_size}, min_samples={min_samples})"
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size         = min_size,
            min_samples              = min_samples,
            metric                   = "euclidean",
            cluster_selection_method = "eom",
            prediction_data          = True,
        )
        clusterer.fit(embeddings)
        labels = clusterer.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        logger.info(
            f"Clusterer: formed {n_clusters} clusters, "
            f"{n_noise} noise points ({n_noise/len(labels)*100:.1f}%)"
        )
        return labels

    # ── Cluster Building ──────────────────────────────────────────────────────

    def _build_raw_clusters(
        self,
        posts: list[RawPost],
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> list[dict]:
        """
        Convert HDBSCAN label assignments into cluster dicts.
        Each dict contains the posts, centroid, category, and metadata
        needed for matching and labeling.
        """
        unique_labels = set(labels)
        unique_labels.discard(-1)   # -1 = noise, not a cluster

        raw_clusters = []
        for label in sorted(unique_labels):
            mask          = labels == label
            cluster_posts = [p for p, m in zip(posts, mask) if m]
            cluster_embs  = embeddings[mask]

            # Centroid = mean of all embeddings, re-normalized
            centroid = normalize(
                cluster_embs.mean(axis=0, keepdims=True), norm="l2"
            )[0]

            # Sources represented in this cluster
            sources = list({p.source for p in cluster_posts})

            # Category — majority vote across all posts in cluster
            categories = [p.category for p in cluster_posts if p.category]
            category   = (
                Counter(categories).most_common(1)[0][0]
                if categories else "general"
            )

            # Top posts by upvotes (for label generation + display)
            top_posts = sorted(
                cluster_posts, key=lambda p: p.upvotes, reverse=True
            )[:5]

            # Cross-platform count — max across posts in cluster
            cross_count = max(
                (p.cross_platform_count for p in cluster_posts), default=1
            )

            raw_clusters.append({
                "hdbscan_label": label,
                "posts":         cluster_posts,
                "centroid":      centroid,
                "post_count":    len(cluster_posts),
                "sources":       sources,
                "category":      category,
                "top_posts":     top_posts,
                "cross_count":   cross_count,
                "upvote_sum":    sum(p.upvotes  for p in cluster_posts),
                "comment_sum":   sum(p.comments for p in cluster_posts),
            })

        return raw_clusters

    # ── Cluster Continuity ────────────────────────────────────────────────────

    def _reconcile_clusters(self, raw_clusters: list[dict]) -> list[dict]:
        """
        Match new clusters to existing DB clusters by centroid similarity.
        - If match found: update existing cluster
        - If no match: create new cluster
        Returns list of dicts with 'db_cluster' (Cluster ORM object) added.
        """
        # Load all currently active clusters from DB
        existing = (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .all()
        )

        now        = datetime.now(timezone.utc)
        reconciled = []

        for raw in raw_clusters:
            new_centroid    = raw["centroid"]
            matched_cluster = None

            if existing:
                # Compare new centroid against all existing centroids
                existing_centroids = []
                for ec in existing:
                    c = Embedder.decode_embedding(ec.centroid)
                    if c is not None:
                        existing_centroids.append(c)
                    else:
                        existing_centroids.append(np.zeros(384))

                existing_matrix = np.array(existing_centroids, dtype=np.float32)
                sims = cosine_similarity(
                    new_centroid.reshape(1, -1),
                    existing_matrix
                )[0]

                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])

                if best_sim >= CONTINUITY_THRESHOLD:
                    matched_cluster = existing[best_idx]

            if matched_cluster:
                # Update existing cluster — preserve first_seen
                matched_cluster.post_count   = raw["post_count"]
                matched_cluster.sources      = ",".join(raw["sources"])
                matched_cluster.category     = raw["category"]
                matched_cluster.last_updated = now
                matched_cluster.centroid     = json.dumps(new_centroid.tolist())
                matched_cluster.top_posts    = json.dumps(
                    [p.id for p in raw["top_posts"]]
                )
                db_cluster = matched_cluster
                logger.debug(
                    f"Cluster continued: '{db_cluster.label}' "
                    f"({raw['post_count']} posts)"
                )
            else:
                # Create new cluster
                label = self._generate_label(raw["top_posts"])
                db_cluster = Cluster(
                    label        = label,
                    category     = raw["category"],
                    first_seen   = now,
                    last_updated = now,
                    post_count   = raw["post_count"],
                    sources      = ",".join(raw["sources"]),
                    top_posts    = json.dumps([p.id for p in raw["top_posts"]]),
                    centroid     = json.dumps(new_centroid.tolist()),
                    is_active    = True,
                )
                self.session.add(db_cluster)
                logger.debug(
                    f"New cluster: '{label}' "
                    f"({raw['post_count']} posts, category={raw['category']})"
                )

            raw["db_cluster"] = db_cluster
            reconciled.append(raw)

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Cluster reconcile commit error: {e}")

        return reconciled

    # ── Snapshots ─────────────────────────────────────────────────────────────

    def _save_snapshots(self, active_clusters: list[dict]) -> None:
        """
        Save a size/engagement snapshot for every active cluster.
        These snapshots are what the growth rate engine reads.
        """
        now = datetime.now(timezone.utc)
        for raw in active_clusters:
            db_cluster = raw.get("db_cluster")
            if not db_cluster or not db_cluster.id:
                continue

            snap = Snapshot(
                cluster_id  = db_cluster.id,
                post_count  = raw["post_count"],
                upvote_sum  = raw["upvote_sum"],
                comment_sum = raw["comment_sum"],
                recorded_at = now,
            )
            self.session.add(snap)

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Snapshot save error: {e}")

    # ── Post Assignment ───────────────────────────────────────────────────────

    def _assign_posts_to_clusters(
        self,
        posts: list[RawPost],
        labels: np.ndarray,
        active_clusters: list[dict],
    ) -> None:
        """Update each post's cluster_id field in the DB."""
        # Build mapping: hdbscan_label → db cluster id
        label_to_db_id = {}
        for raw in active_clusters:
            db_cluster = raw.get("db_cluster")
            if db_cluster and db_cluster.id:
                label_to_db_id[raw["hdbscan_label"]] = db_cluster.id

        for post, label in zip(posts, labels):
            post.cluster_id = label_to_db_id.get(int(label), -1)

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Post assignment commit error: {e}")

    # ── Cluster Expiry ────────────────────────────────────────────────────────

    def _expire_old_clusters(self) -> None:
        """Mark clusters inactive if not updated recently."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=CLUSTER_MAX_AGE_HOURS)
        old_clusters = (
            self.session.query(Cluster)
            .filter(
                Cluster.is_active    == True,
                Cluster.last_updated <= cutoff,
            )
            .all()
        )
        for c in old_clusters:
            c.is_active = False

        if old_clusters:
            logger.info(f"Clusterer: expired {len(old_clusters)} old clusters")

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Cluster expiry error: {e}")

    # ── Label Generation ──────────────────────────────────────────────────────

    def _generate_label(self, top_posts: list[RawPost]) -> str:
        """
        Generate a human-readable label for a cluster from its top posts.
        Uses frequency-based keyword extraction on post titles.
        Returns a 3-5 word phrase describing the topic.
        """
        from processing.cleaner import clean_text

        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "not", "no", "nor", "so", "yet", "both", "either", "neither",
            "that", "this", "these", "those", "it", "its", "as", "up",
            "about", "than", "then", "just", "more", "also", "over",
            "after", "before", "new", "says", "say", "said", "like",
            "what", "how", "why", "when", "who", "which", "where",
            "get", "got", "going", "people", "year", "years", "us",
            "you", "i", "he", "she", "they", "we", "my", "your",
            "his", "her", "their", "our", "can", "now", "time",
            "re", "ve", "ll", "one", "two", "three", "first", "last","trending", "viral", "thread", "megathread", "breaking",
            "today", "week", "month", "breaking", "watch", "video",
            "funny", "lol", "omg", "wtf", "wow", "breaking",
            "via", "amp", "rt", "dm", "pm", "update", "live",
        }

        all_words = []
        for post in top_posts[:8]:
            text  = clean_text(post.title)
            words = [
                w for w in text.split()
                if len(w) > 2 and w not in STOPWORDS
            ]
            all_words.extend(words)

        if not all_words:
            return "trending topic"

        word_counts = Counter(all_words)
        top_words   = [w for w, _ in word_counts.most_common(5)]

        if not top_words:
            return "trending topic"
	# Filter out hex-looking tokens and very long tokens
        top_words = [
            w for w in top_words
            if len(w) <= 20
            and not all(c in "0123456789abcdef" for c in w)
        ]

        if not top_words:
            return "trending topic"

        return " ".join(top_words[:4])
        return " ".join(top_words[:4])


if __name__ == "__main__":
    engine = ClusterEngine()
    count  = engine.run()
    print(f"\n✅ {count} active clusters")

    # Show what was clustered
    session  = get_session()
    clusters = (
        session.query(Cluster)
        .filter(Cluster.is_active == True)
        .order_by(Cluster.post_count.desc())
        .all()
    )
    print(f"\n{'Label':<35} {'Posts':>6} {'Category':<15} {'Sources'}")
    print("─" * 80)
    for c in clusters:
        sources = (c.sources or "").split(",")[:3]
        print(
            f"{c.label:<35} {c.post_count:>6} "
            f"{(c.category or 'general'):<15} "
            f"{', '.join(sources)}"
        )