# processing/cleaner.py
# Cleans raw post text before embedding.
# Goal: remove noise while preserving semantic meaning.
# We do NOT remove stopwords here — sentence transformers work
# better with natural language than bag-of-words style stripping.

import re
import html
from loguru import logger


# Patterns compiled once at module load for performance
_URL_RE         = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_REDDIT_TAG_RE  = re.compile(r"\/?r\/\w+|\/?u\/\w+")
_MENTION_RE     = re.compile(r"@\w+")
_HASHTAG_RE     = re.compile(r"#(\w+)")          # keep word, strip #
_EMOJI_RE       = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE,
)
_SPECIAL_CHARS_RE = re.compile(r"[^\w\s\-\'\"\.\,\!\?\:\;]")
_WHITESPACE_RE    = re.compile(r"\s+")
_REPEATED_PUNCT   = re.compile(r"([!?.]){3,}")   # !!! → !

# Known bot/spam signals — posts containing these are flagged
BOT_SIGNALS = [
    "i am a bot",
    "this action was performed automatically",
    "contact the moderators",
    "if you have questions or concerns",
    "*i am a bot*",
    "beep boop",
]

# Minimum meaningful post length after cleaning
MIN_CLEAN_LENGTH = 10


def clean_text(text: str, preserve_case: bool = False) -> str:
    """
    Full cleaning pipeline for a single piece of text.
    Returns cleaned string, or empty string if text is too short/spam.

    Args:
        text:          Raw text from any source
        preserve_case: If True, keep original casing (for emotion detection).
                       If False, lowercase (for embeddings).
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Decode HTML entities (&amp; → &, &lt; → <, etc.)
    text = html.unescape(text)

    # 2. Strip HTML tags (Mastodon posts come with HTML)
    text = _HTML_TAG_RE.sub(" ", text)

    # 3. Remove URLs entirely
    text = _URL_RE.sub(" ", text)

    # 4. Remove Reddit-specific noise (r/subreddit, u/username)
    text = _REDDIT_TAG_RE.sub(" ", text)

    # 5. Remove @mentions
    text = _MENTION_RE.sub(" ", text)

    # 6. Strip # from hashtags but keep the word (viral → viral)
    text = _HASHTAG_RE.sub(r"\1 ", text)

    # 7. Remove emojis
    text = _EMOJI_RE.sub(" ", text)

    # 8. Collapse repeated punctuation
    text = _REPEATED_PUNCT.sub(r"\1", text)

    # 9. Remove remaining special characters (keep basic punctuation)
    text = _SPECIAL_CHARS_RE.sub(" ", text)

    # 10. Normalize whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # 11. Optionally lowercase
    if not preserve_case:
        text = text.lower()

    # 12. Final length check
    if len(text) < MIN_CLEAN_LENGTH:
        return ""

    return text


def is_bot_post(text: str) -> bool:
    """Returns True if the text contains known bot/spam signals."""
    text_lower = text.lower()
    return any(signal in text_lower for signal in BOT_SIGNALS)


def build_embedding_text(title: str, body: str = "",
                          source: str = "") -> str:
    """
    Combines title + body into a single string for embedding.
    Title is weighted by repeating it — the title carries more
    semantic signal than the body for trend detection purposes.

    Args:
        title:  Post title (most important)
        body:   Post body/summary (supporting context)
        source: Source name (reddit/hn/rss etc) — used for weighting
    """
    title_clean = clean_text(title)
    body_clean  = clean_text(body)

    if not title_clean:
        return ""

    # Title repeated twice = higher weight in the embedding space
    # Body truncated to 200 chars — enough context without diluting signal
    parts = [title_clean, title_clean]
    if body_clean:
        parts.append(body_clean[:200])

    return " ".join(parts).strip()


def clean_for_emotion(text: str) -> str:
    """
    Lighter cleaning pass for emotion detection.
    Preserves case and some punctuation since emotion models
    use these signals (ALL CAPS = emphasis, ! = excitement, etc.)
    """
    if not text or not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _REDDIT_TAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text[:512]   # emotion models have token limits


if __name__ == "__main__":
    # Quick sanity test
    test_cases = [
        "Check out this amazing AI tool! https://example.com #AI #tech",
        "<p>Breaking: Major earthquake hits <b>California</b></p>",
        "r/worldnews — Biden signs new executive order on immigration",
        "@elonmusk just tweeted something wild about crypto 🚀🚀🚀",
        "I am a bot, and this action was performed automatically.",
        "   ",
    ]
    print("Cleaner test:")
    for t in test_cases:
        result = clean_text(t)
        print(f"  IN:  {t[:60]}")
        print(f"  OUT: {result[:60]}")
        print()