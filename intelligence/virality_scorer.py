# intelligence/virality_scorer.py
# Computes a 0-100 virality score for every active cluster.
# Combines 5 signals with weighted formula.
# This score is what gets displayed in the Telegram bot.

import json
import math
from datetime import datetime, timezone, timedelta
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Cluster, Snapshot, RawPost, get_session
from config import (
    W_GROWTH_RATE, W_UPVOTE_VELOCITY, W_COMMENT_VELOCITY,
    W_CROSS_PLATFORM, W_RECENCY,
)

# Normalization ceiling values — scores above these are treated as max
# Tuned based on real-world Reddit/HN engagement patterns
NORM_GROWTH_RATE      = 300.0   # 300% growth = max growth score
NORM_UPVOTE_VELOCITY  = 5000.0  # 5000 upvotes/hour = max
NORM_COMMENT_VELOCITY = 500.0   # 500 comments/hour = max
NORM_CROSS_PLATFORM   = 5.0     # 5 sources = max cross-platform score

# Recency decay — clusters decay to 50% score after this many hours
RECENCY_HALF_LIFE_HOURS = 3.0


class ViralityScorer:
    """
    Scores every active cluster on a 0-100 scale.

    Formula:
        score = (
            W_GROWTH_RATE      × growth_score      +   # 35%
            W_UPVOTE_VELOCITY  × upvote_score      +   # 20%
            W_COMMENT_VELOCITY × comment_score     +   # 20%
            W_CROSS_PLATFORM   × cross_plat_score  +   # 15%
            W_RECENCY          × recency_score          # 10%
        ) × 100

    Each component is normalized to 0-1 before weighting.
    """

    def __init__(self):
        self.session = get_session()

    def run(self) -> list[dict]:
        """
        Score all active clusters.
        Returns sorted list of trend dicts (highest score first).
        """
        clusters = (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .all()
        )

        if not clusters:
            logger.info("ViralityScorer: no active clusters to score")
            return []

        scored = []
        for cluster in clusters:
            score_data = self._score_cluster(cluster)
            cluster.virality_score = score_data["total_score"]
            scored.append(score_data)

        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"ViralityScorer commit error: {e}")

        # Sort by score descending
        scored.sort(key=lambda x: x["total_score"], reverse=True)

        logger.info(
            f"ViralityScorer: scored {len(scored)} clusters. "
            f"Top score: {scored[0]['total_score']:.1f}/100 "
            f"('{scored[0]['label']}')"
            if scored else "ViralityScorer: no clusters scored"
        )

        return scored

    def _score_cluster(self, cluster: Cluster) -> dict:
        """Score a single cluster across all 5 dimensions."""

        # ── 1. Growth Rate Score ──────────────────────────────────────────────
        growth_rate  = max(0.0, cluster.growth_rate or 0.0)
        growth_score = min(growth_rate / NORM_GROWTH_RATE, 1.0)

        # ── 2. Upvote Velocity Score ──────────────────────────────────────────
        upvote_velocity  = self._get_upvote_velocity(cluster)
        upvote_score     = min(upvote_velocity / NORM_UPVOTE_VELOCITY, 1.0)

        # ── 3. Comment Velocity Score ─────────────────────────────────────────
        comment_velocity = self._get_comment_velocity(cluster)
        comment_score    = min(comment_velocity / NORM_COMMENT_VELOCITY, 1.0)

        # ── 4. Cross-Platform Score ───────────────────────────────────────────
        # Count distinct sources in this cluster
        sources      = [s for s in (cluster.sources or "").split(",") if s]
        source_count = len(set(sources))
        cross_score  = min(source_count / NORM_CROSS_PLATFORM, 1.0)

        # Bonus: if any post in cluster has cross_platform_count >= 3
        # that means the SAME STORY appeared on 3+ platforms → boost
        cross_bonus = self._get_cross_platform_bonus(cluster)
        cross_score  = min(cross_score + cross_bonus, 1.0)

        # ── 5. Recency Score ──────────────────────────────────────────────────
        recency_score = self._get_recency_score(cluster)

        # ── Weighted Total ────────────────────────────────────────────────────
        total = (
            W_GROWTH_RATE      * growth_score   +
            W_UPVOTE_VELOCITY  * upvote_score   +
            W_COMMENT_VELOCITY * comment_score  +
            W_CROSS_PLATFORM   * cross_score    +
            W_RECENCY          * recency_score
        ) * 100

        total = round(min(max(total, 0.0), 100.0), 1)

        return {
            "cluster_id":       cluster.id,
            "label":            cluster.label,
            "total_score":      total,
            "growth_score":     round(growth_score * 100, 1),
            "upvote_score":     round(upvote_score * 100, 1),
            "comment_score":    round(comment_score * 100, 1),
            "cross_score":      round(cross_score * 100, 1),
            "recency_score":    round(recency_score * 100, 1),
            "growth_rate":      round(cluster.growth_rate or 0, 1),
            "post_count":       cluster.post_count,
            "source_count":     source_count,
            "sources":          sources,
            "category":         cluster.category or "general",
            "first_seen":       cluster.first_seen,
            "last_updated":     cluster.last_updated,
        }

    # ── Signal Extractors ─────────────────────────────────────────────────────

    def _get_upvote_velocity(self, cluster: Cluster) -> float:
        """
        Upvotes per hour for this cluster.
        Computed from the two most recent snapshots.
        """
        snaps = (
            self.session.query(Snapshot)
            .filter(Snapshot.cluster_id == cluster.id)
            .order_by(Snapshot.recorded_at.desc())
            .limit(2)
            .all()
        )

        if len(snaps) < 2:
            # Only one snapshot — estimate from total upvotes
            if snaps:
                age_hours = max(
                    (datetime.now(timezone.utc) -
                     cluster.first_seen.replace(tzinfo=timezone.utc)
                     if cluster.first_seen.tzinfo is None
                     else cluster.first_seen).total_seconds() / 3600,
                    0.1
                )
                return snaps[0].upvote_sum / age_hours
            return 0.0

        latest, prev = snaps[0], snaps[1]
        time_diff = max(
            (latest.recorded_at - prev.recorded_at).total_seconds() / 3600,
            0.0167
        )
        upvote_delta = max(latest.upvote_sum - prev.upvote_sum, 0)
        return upvote_delta / time_diff

    def _get_comment_velocity(self, cluster: Cluster) -> float:
        """Comments per hour for this cluster."""
        snaps = (
            self.session.query(Snapshot)
            .filter(Snapshot.cluster_id == cluster.id)
            .order_by(Snapshot.recorded_at.desc())
            .limit(2)
            .all()
        )

        if len(snaps) < 2:
            if snaps:
                age_hours = max(
                    (datetime.now(timezone.utc) -
                     (cluster.first_seen.replace(tzinfo=timezone.utc)
                      if cluster.first_seen.tzinfo is None
                      else cluster.first_seen)).total_seconds() / 3600,
                    0.1
                )
                return snaps[0].comment_sum / age_hours
            return 0.0

        latest, prev = snaps[0], snaps[1]
        time_diff    = max(
            (latest.recorded_at - prev.recorded_at).total_seconds() / 3600,
            0.0167
        )
        comment_delta = max(latest.comment_sum - prev.comment_sum, 0)
        return comment_delta / time_diff

    def _get_cross_platform_bonus(self, cluster: Cluster) -> float:
        """
        Check if any post in this cluster was confirmed on 3+ platforms.
        Returns 0.2 bonus if yes, 0.0 otherwise.
        """
        top_post_ids = []
        try:
            top_post_ids = json.loads(cluster.top_posts or "[]")
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not top_post_ids:
            return 0.0

        confirmed = (
            self.session.query(RawPost)
            .filter(
                RawPost.id.in_(top_post_ids),
                RawPost.cross_platform_count >= 3,
            )
            .first()
        )
        return 0.2 if confirmed else 0.0

    def _get_recency_score(self, cluster: Cluster) -> float:
        """
        Exponential decay based on cluster age.
        New cluster (< 30 min): score = 1.0
        3 hours old: score = 0.5
        6 hours old: score = 0.25
        """
        if not cluster.first_seen:
            return 0.5

        first_seen = cluster.first_seen
        if first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=timezone.utc)

        age_hours = (
            datetime.now(timezone.utc) - first_seen
        ).total_seconds() / 3600

        # Exponential decay: score = 2^(-age / half_life)
        score = math.pow(2, -age_hours / RECENCY_HALF_LIFE_HOURS)
        return round(min(max(score, 0.0), 1.0), 4)

    def get_top_trends(self, n: int = 10,
                       category: str = None) -> list[dict]:
        """
        Get top N trends, optionally filtered by category.
        Always re-scores before returning so data is fresh.
        """
        scored = self.run()

        if category:
            scored = [s for s in scored
                      if s.get("category", "").lower() == category.lower()]

        return scored[:n]


if __name__ == "__main__":
    scorer = ViralityScorer()
    trends = scorer.get_top_trends(10)

    print(f"\n🔥 TOP TRENDS RIGHT NOW\n{'='*55}")
    for i, t in enumerate(trends, 1):
        bar = "█" * int(t["total_score"] / 5)
        print(f"\n#{i} {t['label'].upper()}")
        print(f"   {bar}")
        print(f"   📊 Virality:  {t['total_score']}/100")
        print(f"   📈 Growth:    +{t['growth_rate']}%")
        print(f"   💬 Posts:     {t['post_count']}")
        print(f"   🌐 Sources:   {t['source_count']} ({', '.join(t['sources'][:3])})")
        print(f"   🏷️  Category:  {t['category']}")