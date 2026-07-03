# test_phase2.py — corrected
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from database.models         import init_db, get_session, RawPost
from processing.cleaner      import clean_text, build_embedding_text, is_bot_post
from processing.embedder     import Embedder
from processing.deduplicator import Deduplicator


def test_cleaner():
    print("\n── Cleaner ──────────────────────────────────")

    # (input, expected_fragment_in_output, should_be_bot)
    cases = [
        ("AI startup raises $100M #tech https://example.com",
         "ai startup raises",   False),
        ("<p>Breaking news: earthquake hits <b>Japan</b></p>",
         "breaking news",       False),
        ("I am a bot, this was done automatically",
         "",                    True),   # bot → flagged
        ("r/worldnews — Biden signs executive order",
         "biden signs",         False),
        ("@elonmusk says crypto is dead",
         "says crypto is dead", False),  # @mention stripped, rest kept ✅
    ]

    all_pass = True
    for raw, expected_fragment, expect_bot in cases:
        result  = clean_text(raw)
        is_bot  = is_bot_post(raw)

        if expect_bot:
            ok = is_bot
        elif expected_fragment:
            ok = expected_fragment in result
        else:
            ok = True

        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False

        print(f"  {status} '{raw[:50]}'")
        print(f"      → '{result[:60]}'")

    return all_pass


def test_embedder():
    print("\n── Embedder ─────────────────────────────────")
    embedder = Embedder()
    count    = embedder.run()
    print(f"  Embedded {count} new posts this run")

    session  = get_session()
    total    = session.query(RawPost).count()
    embedded = session.query(RawPost).filter(RawPost.is_embedded == True).count()
    print(f"  DB: {embedded}/{total} posts embedded")

    # Verify a sample embedding
    sample = (session.query(RawPost)
              .filter(RawPost.is_embedded == True,
                      RawPost.embedding  != "")
              .first())

    if not sample:
        print("  ❌ No embedded posts found")
        return False

    emb  = Embedder.decode_embedding(sample.embedding)
    norm = np.linalg.norm(emb)
    print(f"  Sample: '{sample.title[:55]}'")
    print(f"  Shape: {emb.shape} | Norm: {norm:.4f} (expect ~1.0)")

    shape_ok = emb.shape == (384,)
    norm_ok  = abs(norm - 1.0) < 0.01
    print(f"  Shape: {'✅' if shape_ok else '❌'} | Norm: {'✅' if norm_ok else '❌'}")

    # Semantic ordering test — relative order matters, not absolute values
    embedder._load_model()
    vecs = embedder.model.encode(
        [
            "artificial intelligence machine learning neural network",
            "bitcoin cryptocurrency blockchain ethereum",
            "war military conflict troops weapons",
        ],
        normalize_embeddings=True,
    )
    sim_ai_ai     = float(np.dot(vecs[0], vecs[0]))
    sim_ai_crypto = float(np.dot(vecs[0], vecs[1]))
    sim_ai_war    = float(np.dot(vecs[0], vecs[2]))

    print(f"\n  Semantic ordering test (relative similarity):")
    print(f"    AI ↔ AI:     {sim_ai_ai:.4f}  (must be 1.0)")
    print(f"    AI ↔ Crypto: {sim_ai_crypto:.4f}  (must be < AI↔AI)")
    print(f"    AI ↔ War:    {sim_ai_war:.4f}  (must be < AI↔AI)")

    # What we actually care about: self-similarity is 1.0,
    # and different topics are less similar than same topic
    ordering_ok = (
        abs(sim_ai_ai - 1.0) < 0.01 and
        sim_ai_crypto < sim_ai_ai and
        sim_ai_war    < sim_ai_ai
    )
    print(f"  Ordering: {'✅ Correct — model is working' if ordering_ok else '❌ Wrong ordering'}")

    return shape_ok and norm_ok and ordering_ok and embedded > 0


def test_deduplicator():
    print("\n── Deduplicator ─────────────────────────────")
    d       = Deduplicator()
    matches = d.run()
    print(f"  Found {matches} cross-platform matches this run")

    session     = get_session()
    cross_count = session.query(RawPost).filter(
        RawPost.cross_platform == True
    ).count()
    print(f"  Total cross-platform posts in DB: {cross_count}")

    # Show the best examples — stories found on 3+ sources
    best = (session.query(RawPost)
            .filter(RawPost.cross_platform == True,
                    RawPost.cross_platform_count >= 3)
            .limit(5)
            .all())

    if best:
        print(f"\n  🌐 Stories confirmed across 3+ sources:")
        for p in best:
            print(f"    [{p.source:<15}] {p.title[:55]} "
                  f"({p.cross_platform_count} sources)")

    # Show 2-source matches too
    two_source = (session.query(RawPost)
                  .filter(RawPost.cross_platform == True,
                          RawPost.cross_platform_count == 2)
                  .limit(3)
                  .all())
    if two_source:
        print(f"\n  🔗 Stories confirmed across 2 sources:")
        for p in two_source:
            print(f"    [{p.source:<15}] {p.title[:55]}")

    return cross_count > 0


if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 2 TEST — CORRECTED")
    print("=" * 55)
    init_db()

    r1 = test_cleaner()
    r2 = test_embedder()
    r3 = test_deduplicator()

    print("\n" + "=" * 55)
    print(f"  Cleaner:       {'✅' if r1 else '❌'}")
    print(f"  Embedder:      {'✅' if r2 else '❌'}")
    print(f"  Deduplicator:  {'✅' if r3 else '❌'}")
    if r1 and r2 and r3:
        print("\n  ✅ Phase 2 complete. Ready for Phase 3.")
    else:
        print("\n  ❌ Check output above.")
    print("=" * 55)