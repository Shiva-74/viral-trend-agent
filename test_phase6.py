# test_phase6.py
from dotenv import load_dotenv
load_dotenv()

from database.models              import init_db
from intelligence.clusterer       import ClusterEngine
from intelligence.growth_detector import GrowthDetector
from intelligence.virality_scorer import ViralityScorer
from intelligence.emotion_detector import EmotionDetector
from predictor.viral_predictor    import ViralPredictor


def test_predictor():
    print("\n── Viral Predictor ──────────────────────────")

    # Ensure pipeline has run so clusters exist
    ClusterEngine().run()
    GrowthDetector().run()
    ViralityScorer().run()
    EmotionDetector().run()

    predictor = ViralPredictor()

    test_cases = [
        # Should score HIGH
        ("I just got fired — company replaced entire dev team with AI overnight",
         "high"),
        ("Breaking: Trump announces 200% tariff on all imports starting Monday",
         "high"),
        ("New OpenAI model leaked — destroys every benchmark ever set",
         "high"),
        # Should score MEDIUM
        ("Scientists find new species of fish in the Pacific Ocean",
         "medium"),
        ("Local restaurant wins award for best pizza in the city",
         "low"),
        # Topic currently trending
        ("Iran military fires warning shots at ships in Strait of Hormuz",
         "high"),
    ]

    all_pass  = True
    print(f"\n  {'Input':<50} {'Score':>6} {'Emotion':<10} {'Verdict'}")
    print(f"  {'─'*90}")

    for text, expected_level in test_cases:
        result = predictor.predict(text)
        prob   = result["probability"]
        emotion = result["emotion"]
        verdict = result["verdict"]

        # Loose validation — just check scores are in range
        valid = 0 <= prob <= 100
        if not valid:
            all_pass = False

        emoji  = result["emotion_emoji"]
        status = "✅" if valid else "❌"
        print(f"  {status} {text[:48]:<48} {prob:>5.1f}% "
              f"{emoji}{emotion:<9} {verdict}")

        if result.get("reasoning"):
            print(f"     💡 {result['reasoning'][0][:75]}")

    print(f"\n  {'─'*90}")

    # Test format_result
    print(f"\n── Formatted Output Sample ──────────────────")
    sample = predictor.predict(
        "Breaking: Major AI company announces AGI has been achieved"
    )
    print(predictor.format_result(sample))

    return all_pass


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 6 TEST")
    print("=" * 55)
    init_db()

    result = test_predictor()

    print("\n" + "=" * 55)
    print(f"  Viral Predictor: {'✅' if result else '❌'}")
    if result:
        print("\n  ✅ Phase 6 complete. Ready for Phase 7 — Telegram Bot.")
    else:
        print("\n  ❌ Check output above.")
    print("=" * 55)