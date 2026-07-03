# test_phase3.py
from dotenv import load_dotenv
load_dotenv()

from database.models              import init_db, get_session, Cluster, Snapshot, RawPost
from intelligence.clusterer       import ClusterEngine
from intelligence.growth_detector import GrowthDetector


def test_clustering():
    print("\n── Clustering Engine ────────────────────────")
    engine = ClusterEngine()
    count  = engine.run()
    print(f"  Active clusters formed: {count}")

    session  = get_session()
    clusters = (session.query(Cluster)
                .filter(Cluster.is_active == True)
                .order_by(Cluster.post_count.desc())
                .limit(10)
                .all())

    print(f"\n  Top 10 clusters by post count:")
    for c in clusters:
        sources = c.sources or ""
        src_list = sources.split(",") if sources else []
        src_display = ", ".join(src_list[:3])
        print(f"    [{c.post_count:3d} posts] '{c.label}' "
              f"| sources: {src_display}")

    snapshots = session.query(Snapshot).count()
    print(f"\n  Snapshots in DB: {snapshots}")

    # Verify posts got assigned to clusters
    assigned = (session.query(RawPost)
                .filter(RawPost.cluster_id != -1)
                .count())
    total_embedded = (session.query(RawPost)
                      .filter(RawPost.is_embedded == True)
                      .count())
    print(f"  Posts assigned to clusters: {assigned}/{total_embedded}")

    return count > 0


def test_growth_detector():
    print("\n── Growth Rate Detector ─────────────────────")
    detector = GrowthDetector()

    # Run clustering twice with a short gap to generate growth data
    print("  Running clustering pass 1...")
    engine = ClusterEngine()
    engine.run()

    import time
    print("  Waiting 5 seconds...")
    time.sleep(5)

    print("  Running clustering pass 2...")
    engine.run()

    print("  Calculating growth rates...")
    rates = detector.run()
    print(f"  Growth rates calculated: {len(rates)} clusters")

    top = detector.get_fastest_growing(10)
    if top:
        print(f"\n  📈 Top growing clusters right now:")
        for c in top:
            bar    = "█" * min(int(abs(c.growth_rate) / 10), 25)
            sign   = "+" if c.growth_rate >= 0 else ""
            print(f"    {bar:<25} {sign}{c.growth_rate:6.0f}% | "
                  f"{c.post_count:3d} posts | '{c.label}'")

    session   = get_session()
    snapshots = session.query(Snapshot).count()
    print(f"\n  Total snapshots in DB: {snapshots}")
    print(f"  (snapshots accumulate over time — growth rates improve)")

    return len(rates) > 0


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 3 TEST")
    print("=" * 55)
    init_db()

    r1 = test_clustering()
    r2 = test_growth_detector()

    print("\n" + "=" * 55)
    print(f"  Clustering:     {'✅' if r1 else '❌'}")
    print(f"  Growth Rates:   {'✅' if r2 else '❌'}")

    if r1 and r2:
        print("\n  ✅ Phase 3 complete. Ready for Phase 4.")
    else:
        print("\n  ❌ Check output above.")
    print("=" * 55)