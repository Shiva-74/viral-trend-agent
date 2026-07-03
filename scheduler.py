# scheduler.py — Main entry point. Run this to start the agent.
import asyncio
import time
import signal
import sys
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from dotenv import load_dotenv

load_dotenv()

from database.models                import init_db
from fetchers.reddit_fetcher        import RedditFetcher
from fetchers.hn_fetcher            import HNFetcher
from fetchers.rss_fetcher           import RSSFetcher
from fetchers.google_trends_fetcher import GoogleTrendsFetcher
from fetchers.youtube_fetcher       import YouTubeFetcher
from fetchers.nitter_fetcher        import NitterFetcher
from processing.embedder            import Embedder
from processing.deduplicator        import Deduplicator
from intelligence.clusterer         import ClusterEngine
from intelligence.growth_detector   import GrowthDetector
from intelligence.virality_scorer   import ViralityScorer
from intelligence.emotion_detector  import EmotionDetector
from bot.telegram_bot               import ViralBot
from config import (
    REDDIT_INTERVAL, HN_INTERVAL, RSS_INTERVAL,
    GOOGLE_TRENDS_INTERVAL, YOUTUBE_INTERVAL, NITTER_INTERVAL,
)

logger.add(
    "logs/scheduler.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# ── Instantiate all workers once ──────────────────────────────────────────────
reddit_fetcher   = RedditFetcher()
hn_fetcher       = HNFetcher()
rss_fetcher      = RSSFetcher()
gt_fetcher       = GoogleTrendsFetcher()
yt_fetcher       = YouTubeFetcher()
nitter_fetcher   = NitterFetcher()
embedder         = Embedder()
deduplicator     = Deduplicator()
cluster_engine   = ClusterEngine()
growth_detector  = GrowthDetector()
virality_scorer  = ViralityScorer()
emotion_detector = EmotionDetector()
viral_bot        = ViralBot()


# ── Pipeline & Jobs ───────────────────────────────────────────────────────────

def processing_pipeline():
    """
    Full intelligence pipeline — runs every 10 minutes.
    Order: embed → deduplicate → cluster → growth → score → emotion
    """
    logger.info("=" * 50)
    logger.info("Pipeline starting...")

    # 1. Embed new posts
    embedded = embedder.run()

    # 2. Cross-platform deduplication
    if embedded > 0:
        deduplicator.run()

    # 3. Cluster into topics
    cluster_count = cluster_engine.run()

    # 4. Calculate growth rates
    if cluster_count > 0:
        growth_detector.run()

    # 5. Score every cluster
    if cluster_count > 0:
        trends = virality_scorer.run()
        if trends:
            logger.info(
                f"🔥 Top trend: '{trends[0]['label']}' "
                f"— {trends[0]['total_score']}/100"
            )

    # 6. Emotion detection
    if cluster_count > 0:
        emotion_detector.run()

    logger.info(
        f"Pipeline complete — "
        f"{embedded} embedded | "
        f"{cluster_count} clusters | scored | emotions tagged"
    )
    logger.info("=" * 50)


def run_alert_check():
    """Run the alert broadcaster — called every 10 minutes."""
    if viral_bot.app is None:
        return
    try:
        asyncio.run(viral_bot.broadcast_alerts(viral_bot.app))
    except Exception as e:
        logger.error(f"Alert check error: {e}")


def print_db_status():
    """Log a live status summary every 5 minutes."""
    try:
        from database.models import get_session, RawPost, Cluster, Snapshot
        s        = get_session()
        total    = s.query(RawPost).count()
        embedded = s.query(RawPost).filter(RawPost.is_embedded == True).count()
        cross    = s.query(RawPost).filter(RawPost.cross_platform == True).count()
        clusters = s.query(Cluster).filter(Cluster.is_active == True).count()
        snaps    = s.query(Snapshot).count()
        s.close()

        logger.info(
            f"📊 Status — "
            f"Posts: {total} | "
            f"Embedded: {embedded} | "
            f"Cross-platform: {cross} | "
            f"Active clusters: {clusters} | "
            f"Snapshots: {snaps}"
        )
    except Exception as e:
        logger.error(f"Status check error: {e}")


def on_job_error(event):
    logger.error(f"Scheduler job crashed: {event.job_id} — {event.exception}")


def on_job_done(event):
    logger.debug(f"Job done: {event.job_id}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def start_scheduler():
    init_db()

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_listener(on_job_error, EVENT_JOB_ERROR)
    scheduler.add_listener(on_job_done,  EVENT_JOB_EXECUTED)

    # ── Fetch jobs ────────────────────────────────────────────────────────────
    scheduler.add_job(
        reddit_fetcher.run, "interval",
        seconds=REDDIT_INTERVAL, id="reddit", max_instances=1
    )
    scheduler.add_job(
        hn_fetcher.run, "interval",
        seconds=HN_INTERVAL, id="hn", max_instances=1
    )
    scheduler.add_job(
        rss_fetcher.run, "interval",
        seconds=RSS_INTERVAL, id="rss", max_instances=1
    )
    scheduler.add_job(
        gt_fetcher.run, "interval",
        seconds=GOOGLE_TRENDS_INTERVAL, id="gtrends", max_instances=1
    )
    scheduler.add_job(
        yt_fetcher.run, "interval",
        seconds=YOUTUBE_INTERVAL, id="youtube", max_instances=1
    )
    scheduler.add_job(
        nitter_fetcher.run, "interval",
        seconds=NITTER_INTERVAL, id="nitter", max_instances=1
    )

    # ── Intelligence pipeline — every 10 minutes ──────────────────────────────
    scheduler.add_job(
        processing_pipeline, "interval",
        seconds=600, id="processing", max_instances=1
    )

    # ── Alert broadcaster — every 10 minutes ──────────────────────────────────
    scheduler.add_job(
        run_alert_check, "interval",
        seconds=600, id="alerts", max_instances=1
    )

    # ── Status logger — every 5 minutes ──────────────────────────────────────
    scheduler.add_job(
        print_db_status, "interval",
        seconds=300, id="status", max_instances=1
    )

    scheduler.start()
    logger.info("=" * 60)
    logger.info("  Viral Agent — all systems active")
    logger.info("  Fetchers:  Reddit | HN | RSS | GTrends | YouTube | Social")
    logger.info("  Pipeline:  Embedder | Deduplicator | Clusterer | Growth")
    logger.info("  Alerts:    Telegram broadcaster active")
    logger.info("=" * 60)

    # ── Run everything immediately on startup ─────────────────────────────────
    logger.info("Running initial fetch from all sources...")
    for fetcher in [
        reddit_fetcher, hn_fetcher, rss_fetcher,
        gt_fetcher, yt_fetcher, nitter_fetcher
    ]:
        try:
            fetcher.run()
        except Exception as e:
            logger.error(f"Initial fetch error ({fetcher.source_name}): {e}")

    logger.info("Running initial intelligence pipeline...")
    try:
        processing_pipeline()
    except Exception as e:
        logger.error(f"Initial pipeline error: {e}")

    # Initial status
    print_db_status()

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def shutdown(signum, frame):
        logger.info("Shutting down Viral Agent...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Keep alive ────────────────────────────────────────────────────────────
    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        shutdown(None, None)


if __name__ == "__main__":
    start_scheduler()