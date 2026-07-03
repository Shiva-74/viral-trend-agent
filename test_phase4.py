# test_phase4.py
from dotenv import load_dotenv
load_dotenv()

from database.models             import init_db, get_session, Cluster
from intelligence.clusterer      import ClusterEngine
from intelligence.growth_detector import GrowthDetector
from intelligence.virality_scorer import ViralityScorer


def test_scoring():
    print("\n── Virality Scorer ──────────────────────────")

    # Make sure clusters exist
    engine  = ClusterEngine()
    n       = engine.run()
    print(f"  Active clusters: {n}")

    if n == 0:
        print("  ❌ No clusters — cannot score")
        return False

    # Run growth detector first
    detector = GrowthDetector()
    detector.run()

    # Score everything
    scorer = ViralityScorer()
    trends = scorer.get_top_trends(10)

    if not trends:
        print("  ❌ No trends scored")
        return False

    print(f"\n  🔥 TOP {len(trends)} TRENDS RIGHT NOW")
    print(f"  {'─'*53}")

    for i, t in enumerate(trends, 1):
        bar = "█" * int(t["total_score"] / 5)
        print(f"\n  #{i} {t['label'].upper()}")
        print(f"     Virality:  {t['total_score']:5.1f}/100  {bar}")
        print(f"     Growth:    +{t['growth_rate']}%")
        print(f"     Posts:     {t['post_count']}")
        print(f"     Sources:   {', '.join(t['sources'][:4])}")
        print(f"     Category:  {t['category']}")
        print(f"     Breakdown: growth={t['growth_score']} "
              f"upvote={t['upvote_score']} "
              f"cross={t['cross_score']} "
              f"recency={t['recency_score']}")

    # Verify scores are in valid range
    all_valid = all(0 <= t["total_score"] <= 100 for t in trends)
    sorted_ok = all(
        trends[i]["total_score"] >= trends[i+1]["total_score"]
        for i in range(len(trends)-1)
    )

    print(f"\n  Score range valid (0-100): {'✅' if all_valid else '❌'}")
    print(f"  Sorted descending:         {'✅' if sorted_ok else '❌'}")

    # Test category filter
    categories = list({t["category"] for t in trends})
    if categories:
        cat   = categories[0]
        filt  = scorer.get_top_trends(10, category=cat)
        print(f"  Category filter '{cat}': {len(filt)} results ✅")

    return all_valid and sorted_ok and len(trends) > 0


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 4 TEST")
    print("=" * 55)
    init_db()

    result = test_scoring()

    print("\n" + "=" * 55)
    print(f"  Virality Scorer: {'✅' if result else '❌'}")
    if result:
        print("\n  ✅ Phase 4 complete. Ready for Phase 5.")
    else:
        print("\n  ❌ Check output above.")
    print("=" * 55)