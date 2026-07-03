# predictor/viral_predictor.py
# Predicts whether a piece of content will go viral.
# Input: any text (post title, meme description, topic, headline)
# Output: probability score, emotion, platform prediction, reasoning

import re
import numpy as np
from datetime import datetime, timezone
from loguru import logger

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models               import Cluster, RawPost, get_session
from processing.embedder           import Embedder
from processing.cleaner            import clean_text, build_embedding_text
from intelligence.emotion_detector import EmotionDetector, EMOTION_EMOJI
from intelligence.virality_scorer  import ViralityScorer

# ── Linguistic Virality Markers ───────────────────────────────────────────────
# Patterns that consistently appear in viral content.
# Each pattern contributes a score bonus.

VIRAL_PATTERNS = [
    # Numbers in titles drive clicks
    (r"\b\d+\b",                          0.05, "Contains numbers"),
    # First-person story = high shareability
    (r"\b(i |my |i'm |i've |i was )\b",   0.08, "First-person narrative"),
    # Questions drive engagement
    (r"\?",                                0.06, "Contains question"),
    # Superlatives signal extreme content
    (r"\b(biggest|worst|best|first|last|"
     r"never|always|only|most|least|"
     r"greatest|record|historic)\b",       0.07, "Superlative language"),
    # Urgency words = breaking/real-time
    (r"\b(breaking|just|now|alert|urgent|"
     r"developing|live|official)\b",       0.08, "Urgency signal"),
    # Controversy drivers
    (r"\b(ban|banned|fired|arrested|"
     r"exposed|leaked|scandal|caught|"
     r"resign|resigns|removed)\b",         0.07, "Controversy signal"),
    # Money/economy = high interest
    (r"\$[\d,]+[kmb]?|\b\d+[\s]?(billion|"
     r"million|trillion)\b",               0.05, "Financial figures"),
    # Death/disaster = high fear spread
    (r"\b(dies|dead|killed|death|crash|"
     r"explosion|disaster|attack)\b",      0.06, "High-impact event"),
    # Tech virality
    (r"\b(ai|gpt|openai|chatgpt|claude|"
     r"llm|robot|elon|musk)\b",            0.05, "Tech viral topic"),
    # Political virality
    (r"\b(trump|biden|congress|senate|"
     r"president|white house|supreme "
     r"court)\b",                          0.04, "Political topic"),
]

# ── Platform Spread Prediction ────────────────────────────────────────────────
# Which platforms a piece of content is likely to spread on,
# based on its category and emotion

PLATFORM_SPREAD = {
    ("tech",          "fear"):     ["Reddit/r/technology", "Hacker News", "Twitter/X"],
    ("tech",          "surprise"): ["Twitter/X", "Reddit/r/technology", "YouTube"],
    ("tech",          "joy"):      ["Hacker News", "Reddit/r/technology", "LinkedIn"],
    ("news",          "fear"):     ["Reddit/r/worldnews", "Twitter/X", "Facebook"],
    ("news",          "anger"):    ["Twitter/X", "Reddit/r/politics", "Facebook"],
    ("news",          "surprise"): ["Twitter/X", "Reddit/r/news", "Google Trends"],
    ("crypto",        "fear"):     ["Reddit/r/CryptoCurrency", "Twitter/X", "Telegram"],
    ("crypto",        "joy"):      ["Reddit/r/CryptoCurrency", "Twitter/X", "YouTube"],
    ("memes",         "joy"):      ["Reddit/r/memes", "Instagram", "TikTok"],
    ("memes",         "anger"):    ["Twitter/X", "Reddit/r/memes", "TikTok"],
    ("entertainment", "surprise"): ["Twitter/X", "Instagram", "YouTube"],
    ("entertainment", "joy"):      ["Instagram", "TikTok", "YouTube"],
    ("entertainment", "sadness"):  ["Twitter/X", "Facebook", "Instagram"],
}

DEFAULT_PLATFORMS = ["Twitter/X", "Reddit", "Google Trends"]

# ── Time of Day Scoring ───────────────────────────────────────────────────────
# Virality peaks at specific hours (UTC). Posting at off-peak = lower spread.

PEAK_HOURS_UTC = {
    range(13, 17): 1.0,    # 9am-1pm EST — peak US morning
    range(17, 22): 0.9,    # 1pm-6pm EST — afternoon
    range(22, 24): 0.7,    # evening
    range(0,  4):  0.6,    # late night
    range(4,  9):  0.5,    # overnight
    range(9,  13): 0.8,    # early morning
}


class ViralPredictor:
    """
    Predicts viral probability for any input text.

    Scoring components:
    1. Semantic similarity to currently trending clusters (35%)
    2. Linguistic virality markers (25%)
    3. Emotion match to currently dominant emotions (20%)
    4. Time of day factor (10%)
    5. Cross-platform bonus if similar content confirmed viral (10%)
    """

    def __init__(self):
        self.session  = get_session()
        self.embedder = None   # lazy load
        self.emotion_detector = EmotionDetector()

    def _load_embedder(self):
        if self.embedder is None:
            self.embedder = Embedder()
            self.embedder._load_model()

    def predict(self, text: str) -> dict:
        """
        Main prediction method.

        Args:
            text: Any text — post title, meme description, topic, headline

        Returns:
            dict with probability, emotion, platforms, reasoning, and details
        """
        if not text or not text.strip():
            return self._empty_result("No text provided")

        text = text.strip()
        logger.info(f"Predictor: analyzing '{text[:60]}...'")

        self._load_embedder()

        # ── Component Scores ──────────────────────────────────────────────────

        # 1. Semantic similarity to trending clusters
        similarity_score, similar_clusters = self._semantic_similarity(text)

        # 2. Linguistic markers
        linguistic_score, triggered_patterns = self._linguistic_score(text)

        # 3. Emotion match
        emotion, emotion_conf, emotion_score = self._emotion_match(text)

        # 4. Time of day
        time_score = self._time_of_day_score()

        # 5. Cross-platform bonus
        cross_bonus = self._cross_platform_bonus(similar_clusters)

        # ── Weighted Total ────────────────────────────────────────────────────
        raw_score = (
            0.35 * similarity_score +
            0.25 * linguistic_score +
            0.20 * emotion_score    +
            0.10 * time_score       +
            0.10 * cross_bonus
        )

        # Convert to 0-100 percentage
        probability = round(min(max(raw_score * 100, 1.0), 99.0), 1)

        # ── Determine category from similar clusters ───────────────────────────
        category = self._guess_category(text, similar_clusters)

        # ── Platform prediction ───────────────────────────────────────────────
        platforms = self._predict_platforms(category, emotion)

        # ── Peak time estimate ────────────────────────────────────────────────
        peak_estimate = self._estimate_peak(probability, emotion)

        # ── Build reasoning ───────────────────────────────────────────────────
        reasoning = self._build_reasoning(
            probability, triggered_patterns, emotion,
            emotion_conf, similar_clusters, cross_bonus > 0
        )

        result = {
            "probability":       probability,
            "verdict":           self._verdict(probability),
            "emotion":           emotion,
            "emotion_emoji":     EMOTION_EMOJI.get(emotion, "📰"),
            "emotion_confidence": round(emotion_conf * 100, 1),
            "category":          category,
            "platforms":         platforms,
            "peak_estimate":     peak_estimate,
            "reasoning":         reasoning,
            "similar_trends":    [c["label"] for c in similar_clusters[:3]],
            "score_breakdown": {
                "similarity":  round(similarity_score * 100, 1),
                "linguistic":  round(linguistic_score * 100, 1),
                "emotion":     round(emotion_score * 100, 1),
                "time":        round(time_score * 100, 1),
                "cross_bonus": round(cross_bonus * 100, 1),
            },
            "input_text": text[:200],
        }

        logger.info(
            f"Predictor: {probability}% viral — "
            f"{emotion} — '{text[:50]}'"
        )
        return result

    # ── Component Scorers ─────────────────────────────────────────────────────

    def _semantic_similarity(self, text: str) -> tuple[float, list]:
        """
        Compare input text embedding against all active cluster centroids.
        High similarity to a currently trending cluster = high score.
        Returns (score 0-1, list of similar cluster dicts)
        """
        embedding_text = build_embedding_text(text)
        if not embedding_text:
            return 0.0, []

        try:
            input_emb = self.embedder.model.encode(
                [embedding_text],
                normalize_embeddings=True
            )[0]
        except Exception as e:
            logger.error(f"Predictor embedding error: {e}")
            return 0.0, []

        # Load active clusters with centroids and virality scores
        clusters = (
            self.session.query(Cluster)
            .filter(
                Cluster.is_active  == True,
                Cluster.centroid   != "",
                Cluster.centroid   != None,
            )
            .all()
        )

        if not clusters:
            return 0.0, []

        similar_clusters = []
        best_sim         = 0.0

        for cluster in clusters:
            centroid = Embedder.decode_embedding(cluster.centroid)
            if centroid is None:
                continue

            sim = float(np.dot(input_emb, centroid))
            sim = max(0.0, sim)

            if sim > 0.3:
                similar_clusters.append({
                    "label":         cluster.label,
                    "similarity":    round(sim, 3),
                    "virality":      cluster.virality_score or 0,
                    "growth_rate":   cluster.growth_rate or 0,
                    "category":      cluster.category or "general",
                    "cross_count":   1,
                })

            best_sim = max(best_sim, sim)

        # Sort by similarity
        similar_clusters.sort(key=lambda x: x["similarity"], reverse=True)

        # Score: how similar AND how viral is the closest match
        if similar_clusters:
            top    = similar_clusters[0]
            score  = top["similarity"] * 0.6 + (top["virality"] / 100) * 0.4
        else:
            score  = 0.0

        return min(score, 1.0), similar_clusters[:5]

    def _linguistic_score(self, text: str) -> tuple[float, list[str]]:
        """
        Check text against viral linguistic patterns.
        Returns (score 0-1, list of triggered pattern descriptions)
        """
        text_lower       = text.lower()
        total_bonus      = 0.0
        triggered        = []

        for pattern, bonus, description in VIRAL_PATTERNS:
            if re.search(pattern, text_lower):
                total_bonus += bonus
                triggered.append(description)

        # Title length sweet spot: 60-100 chars performs best
        length = len(text)
        if 60 <= length <= 100:
            total_bonus += 0.05
            triggered.append("Optimal title length")
        elif length < 20:
            total_bonus -= 0.05

        return min(total_bonus, 1.0), triggered

    def _emotion_match(self, text: str) -> tuple[str, float, float]:
        """
        Detect emotion in input text and compare to currently
        dominant emotions across trending clusters.
        Returns (emotion, confidence, score 0-1)
        """
        # Classify the input text
        emotion, conf = self.emotion_detector.get_dominant_emotion(text)

        # Check if this emotion is currently dominant in trending clusters
        clusters = (
            self.session.query(Cluster)
            .filter(Cluster.is_active == True)
            .all()
        )

        if not clusters:
            # No trending data — use emotion base rates
            base_rates = {
                "fear": 0.8, "anger": 0.75, "surprise": 0.7,
                "joy": 0.6, "sadness": 0.5, "disgust": 0.5, "neutral": 0.3
            }
            score = conf * base_rates.get(emotion, 0.4)
            return emotion, conf, score

        # Count how many trending clusters share this emotion
        total    = len(clusters)
        matching = sum(1 for c in clusters if c.emotion == emotion)
        dominance = matching / total if total > 0 else 0

        # Combined score: emotion confidence × current dominance
        score = conf * 0.6 + dominance * 0.4
        return emotion, conf, min(score, 1.0)

    def _time_of_day_score(self) -> float:
        """Score based on current time — peak hours = higher spread potential."""
        current_hour = datetime.now(timezone.utc).hour
        for hour_range, score in PEAK_HOURS_UTC.items():
            if current_hour in hour_range:
                return score
        return 0.6

    def _cross_platform_bonus(self, similar_clusters: list) -> float:
        """
        Bonus if similar content has already been confirmed
        viral across multiple platforms.
        """
        if not similar_clusters:
            return 0.0

        # Check if the top similar cluster has cross-platform posts
        top_cluster_label = similar_clusters[0].get("label", "")
        cluster = (
            self.session.query(Cluster)
            .filter(Cluster.label == top_cluster_label)
            .first()
        )

        if not cluster:
            return 0.0

        sources = [s for s in (cluster.sources or "").split(",") if s]
        if len(set(sources)) >= 3:
            return 0.8   # confirmed on 3+ platforms = strong bonus
        elif len(set(sources)) >= 2:
            return 0.4
        return 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _verdict(self, probability: float) -> str:
        if probability >= 80:
            return "🔥 Very Likely Viral"
        elif probability >= 60:
            return "📈 Likely to Trend"
        elif probability >= 40:
            return "⚡ Has Viral Potential"
        elif probability >= 20:
            return "📊 Below Average Chance"
        else:
            return "❄️  Unlikely to Spread"

    def _guess_category(self, text: str, similar_clusters: list) -> str:
        """Guess content category from similar clusters or text keywords."""
        if similar_clusters:
            return similar_clusters[0].get("category", "general")

        text_lower = text.lower()
        if any(w in text_lower for w in ["ai", "tech", "software", "openai", "gpt"]):
            return "tech"
        if any(w in text_lower for w in ["bitcoin", "crypto", "stock", "market"]):
            return "crypto"
        if any(w in text_lower for w in ["movie", "celebrity", "music", "nba", "nfl"]):
            return "entertainment"
        if any(w in text_lower for w in ["meme", "funny", "viral"]):
            return "memes"
        return "news"

    def _predict_platforms(self, category: str, emotion: str) -> list[str]:
        """Predict which platforms content will spread on."""
        key = (category, emotion)
        return PLATFORM_SPREAD.get(key, DEFAULT_PLATFORMS)

    def _estimate_peak(self, probability: float, emotion: str) -> str:
        """Estimate when content will peak based on probability and emotion."""
        if probability >= 75:
            if emotion in ("fear", "anger"):
                return "30–90 minutes"
            elif emotion == "surprise":
                return "1–3 hours"
            else:
                return "2–6 hours"
        elif probability >= 50:
            return "3–8 hours"
        elif probability >= 30:
            return "6–24 hours (if at all)"
        else:
            return "Unlikely to peak"

    def _build_reasoning(
        self,
        probability: float,
        triggered_patterns: list,
        emotion: str,
        emotion_conf: float,
        similar_clusters: list,
        has_cross_platform: bool,
    ) -> list[str]:
        """Build human-readable explanation of the prediction."""
        reasons = []

        if similar_clusters:
            top = similar_clusters[0]
            reasons.append(
                f"Similar to currently trending '{top['label']}' "
                f"(+{top['growth_rate']:.0f}% growth, "
                f"{top['virality']:.0f}/100 virality)"
            )

        if triggered_patterns:
            reasons.append(
                f"Viral language detected: {', '.join(triggered_patterns[:4])}"
            )

        if emotion != "neutral" and emotion_conf > 0.5:
            reasons.append(
                f"{EMOTION_EMOJI.get(emotion, '')} Strong {emotion} signal "
                f"({emotion_conf:.0%} confidence) — "
                f"emotional content spreads faster"
            )

        if has_cross_platform:
            reasons.append(
                "⚡ Related content confirmed viral across 3+ platforms"
            )

        if not reasons:
            if probability > 40:
                reasons.append("Moderate viral signals detected")
            else:
                reasons.append("Weak viral signals — content may not spread widely")

        return reasons

    def _empty_result(self, reason: str) -> dict:
        return {
            "probability": 0,
            "verdict":     "❌ Cannot analyze",
            "error":       reason,
        }

    def format_result(self, result: dict) -> str:
        """
        Format prediction result as a clean Telegram message.
        """
        if "error" in result:
            return f"❌ {result['error']}"

        prob    = result["probability"]
        verdict = result["verdict"]
        emotion = result["emotion_emoji"] + " " + result["emotion"].capitalize()
        lines   = [
            f"🎯 *VIRAL PREDICTION*",
            f"",
            f"📝 *Input:* {result['input_text'][:100]}",
            f"",
            f"📊 *Viral Probability: {prob}%*",
            f"{'█' * int(prob/5)}{'░' * (20 - int(prob/5))} {prob}%",
            f"*{verdict}*",
            f"",
            f"{emotion} ({result['emotion_confidence']}% confidence)",
            f"⏱️  Peak estimate: {result['peak_estimate']}",
            f"",
        ]

        if result.get("similar_trends"):
            lines.append("🔗 *Similar to trending:*")
            for trend in result["similar_trends"]:
                lines.append(f"  • {trend}")
            lines.append("")

        if result.get("reasoning"):
            lines.append("💡 *Why:*")
            for reason in result["reasoning"]:
                lines.append(f"  • {reason}")
            lines.append("")

        if result.get("platforms"):
            platforms = " → ".join(result["platforms"][:3])
            lines.append(f"🌐 *Spread path:* {platforms}")

        breakdown = result.get("score_breakdown", {})
        if breakdown:
            lines.append(f"")
            lines.append(
                f"📈 _Similarity:{breakdown.get('similarity',0):.0f} "
                f"Linguistic:{breakdown.get('linguistic',0):.0f} "
                f"Emotion:{breakdown.get('emotion',0):.0f} "
                f"Timing:{breakdown.get('time',0):.0f}_"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    predictor = ViralPredictor()

    test_cases = [
        "I just got fired by an AI — my company replaced entire dev team overnight",
        "New OpenAI model just leaked and it destroys everything else",
        "Trump announces 200% tariff on all Chinese goods starting tomorrow",
        "Scientists discover water on Mars in breakthrough finding",
        "Local cat learns to open fridge, owner documents everything",
    ]

    print("🎯 VIRAL PREDICTOR TEST\n" + "=" * 55)
    for text in test_cases:
        result = predictor.predict(text)
        print(f"\n📝 {text[:65]}")
        print(f"   📊 {result['probability']}% — {result['verdict']}")
        print(f"   {result['emotion_emoji']} {result['emotion'].capitalize()} "
              f"({result['emotion_confidence']}%)")
        print(f"   ⏱️  Peak: {result['peak_estimate']}")
        if result.get("reasoning"):
            print(f"   💡 {result['reasoning'][0][:70]}")