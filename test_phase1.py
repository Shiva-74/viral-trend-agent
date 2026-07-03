# test_phase1.py — Run to verify all fetchers work before scheduler
import sys
from dotenv import load_dotenv
load_dotenv()

from database.models import init_db, get_session, RawPost
from fetchers.reddit_fetcher        import RedditFetcher
from fetchers.hn_fetcher            import HNFetcher
from fetchers.rss_fetcher           import RSSFetcher
from fetchers.google_trends_fetcher import GoogleTrendsFetcher
from fetchers.youtube_fetcher       import YouTubeFetcher
from fetchers.nitter_fetcher        import NitterFetcher

def test_all():
    print("\n🔧 Initialising database...")
    init_db()

    fetchers = [
        ("Reddit",        RedditFetcher()),
        ("Hacker News",   HNFetcher()),
        ("RSS Feeds",     RSSFetcher()),
        ("Google Trends", GoogleTrendsFetcher()),
        ("YouTube",       YouTubeFetcher()),
        ("Nitter",        NitterFetcher()),
    ]

    results = {}
    for name, fetcher in fetchers:
        print(f"\n⏳ Testing {name}...")
        try:
            count = fetcher.run()
            results[name] = ("✅", count)
            print(f"  → Saved {count} new posts")
        except Exception as e:
            results[name] = ("❌", str(e))
            print(f"  → FAILED: {e}")

    # Summary
    session = get_session()
    total   = session.query(RawPost).count()
    print("\n" + "=" * 50)
    print("  PHASE 1 TEST RESULTS")
    print("=" * 50)
    for name, (status, val) in results.items():
        print(f"  {status}  {name:<20} → {val}")
    print(f"\n  📦 Total posts in database: {total}")
    print("=" * 50)

    all_ok = all(status == "✅" for status, _ in results.values())
    if all_ok:
        print("\n✅ Phase 1 complete. Run: python scheduler.py\n")
    else:
        print("\n⚠️  Some fetchers failed. Check logs/agent.log for details.\n")

if __name__ == "__main__":
    test_all()