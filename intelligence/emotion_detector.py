# intelligence/emotion_detector.py
# Detects the dominant emotion driving each trend cluster.
# Model: j-hartmann/emotion-english-distilroberta-base
# Runs locally on CPU, no API needed.
# Emotions: joy, fear, anger, sadness, surprise, disgust, neutral

import json
from datetime import datetime, timezone
from loguru import logger
from collections import Counter
from transformers import pipeline

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models    import Cluster, RawPost, get_session
from processing.cleaner import clean_for_emotion
from config             import EMOTION_MODEL

# How many top posts to sample per cluster for emotion detection
POSTS_PER_CLUSTER = 8

# Minimum confidence to accept an emotion label
MIN_CONFIDENCE = 0.25

# Emotion → spread pattern mapping
# Used by the Telegram bot to explain why something is spreading
EMOTION_SPREAD_PATTERN = {
    "fear":     "⚠️  Spreads fast — fear drives urgent sharing",
    "anger":    "😡 High comment velocity — controversy drives engagement",
    "joy":      "🎉 Sustained spread — positive content gets reshared",
    "surprise": "😲 Sharp spike — shock value drives initial burst",
    "sadness":  "😢 Slower spread — empathy-driven shares",
    "disgust":  "🤢 Controversy-driven — outrage sharing",
    "neutral":  "📰 Informational spread — news-driven",
}

# Emoji per emotion for display
EMOTION_EMOJI = {
    "fear":     "😨",
    "anger":    "😡",
    "joy":      "🚀",
    "surprise": "😲",
    "sadness":  "😢",
    "disgust":  "🤢",
    "neutral":  "📰",
}


class EmotionDetector:
    """
    For each active cluster, samples the top N posts,
    runs emotion classification on each title, and assigns
    the dominant emotion to the cluster.

    Uses a lazy-loaded pipeline — model only loads on first call.
    """

    def __init__(self):
        self.session  = get_session()
        self._pipe    = None   # lazy load

    def _load_model(self):
        """Load the emotion classification pipeline on first use."""
        if self._pipe is None:
            logger.info(f"Loading emotion model: {EMOTION_MODEL}")
            logger.info("First load downloads ~300MB — one time only.")
            self._pipe = pipeline(
                "text-classification",
                model      = EMOTION_MODEL,
                top_k      = None,       # return all emotion scores
                truncation = True,
                max_length = 128,
            )
            logger.info("Emotion model loaded ✅")

    def run(self) -> int:
        """
        Detect emotions for all active clusters.
        Returns count of clusters processed.
        """
        clusters = (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .all()
        )

        if not clusters:
            logger.info("EmotionDetector: no active clusters")
            return 0

        self._load_model()

        processed = 0
        for cluster in clusters:
            emotion, score = self._detect_cluster_emotion(cluster)
            cluster.emotion       = emotion
            cluster.emotion_score = score
            processed += 1

            logger.debug(
                f"Emotion [{emotion} {score:.2f}] "
                f"'{cluster.label}'"
            )

        try:
            self.session.commit()
            logger.info(
                f"EmotionDetector: tagged {processed} clusters"
            )
        except Exception as e:
            self.session.rollback()
            logger.error(f"EmotionDetector commit error: {e}")

        return processed

    def _detect_cluster_emotion(
        self, cluster: Cluster
    ) -> tuple[str, float]:
        """
        Detect dominant emotion for a single cluster.

        Strategy:
        1. Load top N posts from this cluster
        2. Run emotion classifier on each title
        3. Aggregate scores across all posts
        4. Return the emotion with highest aggregate confidence
        """
        # Get top post IDs for this cluster
        top_post_ids = []
        try:
            top_post_ids = json.loads(cluster.top_posts or "[]")
        except (json.JSONDecodeError, TypeError):
            pass

        # Also get posts by cluster_id as fallback
        if not top_post_ids:
            posts = (
                self.session.query(RawPost)
                .filter(RawPost.cluster_id == cluster.id)
                .order_by(RawPost.upvotes.desc())
                .limit(POSTS_PER_CLUSTER)
                .all()
            )
        else:
            posts = (
                self.session.query(RawPost)
                .filter(RawPost.id.in_(top_post_ids))
                .order_by(RawPost.upvotes.desc())
                .limit(POSTS_PER_CLUSTER)
                .all()
            )

        if not posts:
            return "neutral", 0.5

        # Build texts for classification
        texts = []
        for post in posts:
            text = clean_for_emotion(post.title)
            if text and len(text) > 5:
                texts.append(text[:512])

        if not texts:
            return "neutral", 0.5

        # Run emotion classifier on all texts at once
        try:
            results = self._pipe(texts, batch_size=8)
        except Exception as e:
            logger.error(f"Emotion classification error: {e}")
            return "neutral", 0.5

        # Aggregate emotion scores across all posts
        # Each result is a list of {label, score} dicts
        aggregated = Counter()
        for result in results:
            for item in result:
                label = item["label"].lower()
                score = item["score"]
                if score >= MIN_CONFIDENCE:
                    aggregated[label] += score

        if not aggregated:
            return "neutral", 0.5

        # Dominant emotion = highest aggregate score
        dominant_emotion = aggregated.most_common(1)[0][0]
        total_score      = aggregated[dominant_emotion]
        # Normalize to 0-1 by dividing by number of texts
        normalized_score = min(total_score / len(texts), 1.0)

        return dominant_emotion, round(normalized_score, 3)

    def classify_text(self, text: str) -> dict:
        """
        Classify a single piece of text.
        Returns dict of {emotion: score} for all emotions.
        Used by the viral predictor.
        """
        self._load_model()
        text = clean_for_emotion(text)
        if not text:
            return {"neutral": 1.0}

        try:
            results = self._pipe([text[:512]])[0]
            return {
                item["label"].lower(): round(item["score"], 3)
                for item in results
            }
        except Exception as e:
            logger.error(f"classify_text error: {e}")
            return {"neutral": 1.0}

    def get_dominant_emotion(self, text: str) -> tuple[str, float]:
        """
        Returns (emotion_name, confidence) for a single text.
        Convenience wrapper used by the predictor.
        """
        scores = self.classify_text(text)
        if not scores:
            return "neutral", 0.5
        best = max(scores, key=scores.get)
        return best, scores[best]

    @staticmethod
    def format_emotion(emotion: str, score: float) -> str:
        """Format emotion for display in Telegram messages."""
        emoji   = EMOTION_EMOJI.get(emotion, "📰")
        pattern = EMOTION_SPREAD_PATTERN.get(emotion, "")
        return f"{emoji} {emotion.capitalize()} ({score:.0%}) — {pattern}"


if __name__ == "__main__":
    detector = EmotionDetector()
    count    = detector.run()
    print(f"\n✅ Tagged {count} clusters with emotions\n")

    # Show results
    from database.models import get_session, Cluster
    session  = get_session()
    clusters = (
        session.query(Cluster)
        .filter(Cluster.is_active == True)
        .order_by(Cluster.virality_score.desc())
        .all()
    )

    print(f"{'Label':<35} {'Emotion':<10} {'Score':>6} {'Virality':>9}")
    print("─" * 65)
    for c in clusters:
        emoji = EMOTION_EMOJI.get(c.emotion or "neutral", "📰")
        print(
            f"{c.label:<35} "
            f"{emoji} {(c.emotion or 'neutral'):<8} "
            f"{c.emotion_score or 0:>6.2f} "
            f"{c.virality_score or 0:>8.1f}/100"
        )