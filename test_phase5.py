# test_phase5.py
from dotenv import load_dotenv
load_dotenv()

from database.models               import init_db, get_session, Cluster
from intelligence.clusterer        import ClusterEngine
from intelligence.growth_detector  import GrowthDetector
from intelligence.virality_scorer  import ViralityScorer
from intelligence.emotion_detector import EmotionDetector, EMOTION_EMOJI


def test_emotion_detection():
    print("\n── Emotion Detection ────────────────────────")

    # Ensure clusters exist and are scored
    engine = ClusterEngine()
    engine.run()
    GrowthDetector().run()
    ViralityScorer().run()

    # Run emotion detection
    detector  = EmotionDetector()
    processed = detector.run()
    print(f"  Clusters tagged: {processed}")

    # Show results
    session  = get_session()
    clusters = (
        session.query(Cluster)
        .filter(Cluster.is_active == True)
        .order_by(Cluster.virality_score.desc())
        .all()
    )

    print(f"\n  {'#':<3} {'Label':<32} {'Emotion':<10} {'Conf':>5} {'Score':>8}")
    print(f"  {'─'*65}")
    for i, c in enumerate(clusters, 1):
        emoji   = EMOTION_EMOJI.get(c.emotion or "neutral", "📰")
        emotion = (c.emotion or "neutral").capitalize()
        print(
            f"  #{i:<2} {c.label:<32} "
            f"{emoji} {emotion:<8} "
            f"{(c.emotion_score or 0):>5.2f} "
            f"{(c.virality_score or 0):>7.1f}/100"
        )

    # Test single text classification
    print(f"\n  Single text classification test:")
    test_texts = [
        "War breaks out — thousands flee as bombs fall on capital city",
        "New iPhone just announced and it looks absolutely incredible",
        "Politician caught lying about corruption — people are furious",
        "Scientists discover cure for cancer in breakthrough study",
    ]
    for text in test_texts:
        emotion, conf = detector.get_dominant_emotion(text)
        emoji         = EMOTION_EMOJI.get(emotion, "📰")
        print(f"  {emoji} [{emotion:<8} {conf:.2f}] {text[:55]}")

    # Verify emotions are being stored
    tagged = session.query(Cluster).filter(
        Cluster.is_active == True,
        Cluster.emotion   != "",
        Cluster.emotion   != None,
    ).count()

    print(f"\n  Clusters with emotion tags: {tagged}/{len(clusters)}")
    return tagged > 0 and processed > 0


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 5 TEST")
    print("=" * 55)
    init_db()

    result = test_emotion_detection()

    print("\n" + "=" * 55)
    print(f"  Emotion Detection: {'✅' if result else '❌'}")
    if result:
        print("\n  ✅ Phase 5 complete. Ready for Phase 6.")
    else:
        print("\n  ❌ Check output above.")
    print("=" * 55)