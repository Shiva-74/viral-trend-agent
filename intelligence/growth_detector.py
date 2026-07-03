# intelligence/growth_detector.py
# Calculates growth rates for every cluster by comparing
# current snapshot against snapshots from 30 minutes ago.
# Growth rate = the most important signal in the virality score.

from datetime import datetime, timezone, timedelta
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Cluster, Snapshot, get_session

# Compare against snapshot from this many minutes ago
GROWTH_LOOKBACK_MINUTES = 30

# Minimum posts needed for a growth rate to be meaningful
MIN_POSTS_FOR_GROWTH = 3


class GrowthDetector:
    """
    For each active cluster, computes:
    - growth_rate: % change in post count over last 30 minutes
    - upvote_velocity: upvotes per hour
    - comment_velocity: comments per hour
    Updates the Cluster.growth_rate field in DB.
    """

    def __init__(self):
        self.session = get_session()

    def run(self) -> dict[int, float]:
        """
        Calculate growth rates for all active clusters.
        Returns dict of {cluster_id: growth_rate_percent}
        """
        now     = datetime.now(timezone.utc)
        cutoff  = now - timedelta(minutes=GROWTH_LOOKBACK_MINUTES)

        active_clusters = (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .all()
        )

        if not active_clusters:
            logger.info("GrowthDetector: no active clusters")
            return {}

        growth_rates = {}
        updated      = 0

        for cluster in active_clusters:
            result = self._calculate_growth(cluster, now, cutoff)

            cluster.growth_rate  = result["growth_rate"]
            cluster.last_updated = now
            growth_rates[cluster.id] = result["growth_rate"]

            if result["growth_rate"] > 50:
                logger.info(
                    f"📈 Growing cluster '{cluster.label}': "
                    f"+{result['growth_rate']:.0f}% | "
                    f"{cluster.post_count} posts"
                )
            updated += 1

        try:
            self.session.commit()
            logger.info(
                f"GrowthDetector: updated {updated} clusters. "
                f"Top growth: {max(growth_rates.values(), default=0):.0f}%"
            )
        except Exception as e:
            self.session.rollback()
            logger.error(f"GrowthDetector commit error: {e}")

        return growth_rates

    def _calculate_growth(
        self,
        cluster: Cluster,
        now: datetime,
        lookback_cutoff: datetime,
    ) -> dict:
        """
        Calculate all growth metrics for a single cluster.

        Growth rate formula:
            rate = (current_count - past_count) / max(past_count, 1) * 100

        If the cluster is brand new (no past snapshot), we assign
        a moderate growth rate based on its current size relative
        to the average cluster size.
        """
        # Get the most recent snapshot
        latest_snap = (
            self.session.query(Snapshot)
            .filter(Snapshot.cluster_id == cluster.id)
            .order_by(Snapshot.recorded_at.desc())
            .first()
        )

        # Get the snapshot closest to GROWTH_LOOKBACK_MINUTES ago
        past_snap = (
            self.session.query(Snapshot)
            .filter(
                Snapshot.cluster_id  == cluster.id,
                Snapshot.recorded_at <= lookback_cutoff,
            )
            .order_by(Snapshot.recorded_at.desc())
            .first()
        )

        if not latest_snap:
            return {"growth_rate": 0.0, "upvote_velocity": 0.0}

        current_count = latest_snap.post_count
        current_votes = latest_snap.upvote_sum
        current_comms = latest_snap.comment_sum

        if past_snap:
            past_count = max(past_snap.post_count, 1)
            growth_rate = ((current_count - past_count) / past_count) * 100

            # Calculate velocities (per hour)
            time_diff_hours = max(
                (latest_snap.recorded_at - past_snap.recorded_at).total_seconds() / 3600,
                0.0167  # minimum 1 minute to avoid division by zero
            )
            upvote_velocity  = (current_votes - past_snap.upvote_sum)  / time_diff_hours
            comment_velocity = (current_comms - past_snap.comment_sum) / time_diff_hours

        else:
            # New cluster — no past data yet
            # Assign a baseline growth rate based on post count
            # Small new cluster (< 10 posts): moderate signal
            # Large new cluster (50+ posts): strong signal
            if current_count >= 50:
                growth_rate = 100.0
            elif current_count >= 20:
                growth_rate = 50.0
            elif current_count >= 10:
                growth_rate = 25.0
            else:
                growth_rate = 10.0

            upvote_velocity  = current_votes / 0.5  # assume 30min old
            comment_velocity = current_comms / 0.5

        # Clamp growth rate to reasonable range
        # Very negative means cluster is dying (old content)
        # Cap at 1000% to avoid score inflation
        growth_rate = max(-100.0, min(1000.0, growth_rate))

        return {
            "growth_rate":      growth_rate,
            "upvote_velocity":  max(0.0, upvote_velocity  if past_snap else upvote_velocity),
            "comment_velocity": max(0.0, comment_velocity if past_snap else comment_velocity),
        }

    def get_fastest_growing(self, top_n: int = 10) -> list[Cluster]:
        """Return the top N fastest growing active clusters."""
        return (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .order_by(Cluster.growth_rate.desc())
            .limit(top_n)
            .all()
        )


if __name__ == "__main__":
    detector = GrowthDetector()
    rates    = detector.run()

    print(f"\n✅ Growth rates calculated for {len(rates)} clusters")
    print("\nFastest growing right now:")
    top = detector.get_fastest_growing(10)
    for c in top:
        bar = "█" * min(int(c.growth_rate / 10), 30)
        print(f"  {bar} +{c.growth_rate:.0f}% | "
              f"{c.post_count:3d} posts | '{c.label}'")