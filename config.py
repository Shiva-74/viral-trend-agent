# config.py — All settings in one place. Change values here only.

# ── Geography Filter ──────────────────────────────────────────────
# We only want US, UK, Europe trends. These settings enforce that.
TARGET_COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "NL", "SE", "NO"]
GOOGLE_TRENDS_GEO = "US"   # Primary geo for Google Trends
EXCLUDE_REGIONS = ["IN", "PK", "BD", "LK", "NP"]  # Explicitly exclude

# ── Reddit Sources ────────────────────────────────────────────────
REDDIT_ENDPOINTS = [
    "https://www.reddit.com/r/all/rising.json",
    "https://www.reddit.com/r/popular.json",
    "https://www.reddit.com/r/technology/hot.json",
    "https://www.reddit.com/r/worldnews/hot.json",
    "https://www.reddit.com/r/memes/rising.json",
    "https://www.reddit.com/r/CryptoCurrency/rising.json",
    "https://www.reddit.com/r/artificial/hot.json",
    "https://www.reddit.com/r/entertainment/hot.json",
    "https://www.reddit.com/r/europe/hot.json",
    "https://www.reddit.com/r/unitedkingdom/hot.json",
    "https://www.reddit.com/r/politics/rising.json",
]

# ── News RSS Feeds (Western only) ────────────────────────────────
RSS_FEEDS = {
    "tech":    ["https://feeds.feedburner.com/TechCrunch",
                "https://www.theverge.com/rss/index.xml",
                "https://feeds.arstechnica.com/arstechnica/index"],
    "news":    ["http://feeds.bbci.co.uk/news/rss.xml",
                "https://feeds.reuters.com/reuters/topNews",
                "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"],
    "crypto":  ["https://coindesk.com/arc/outboundfeeds/rss/",
                "https://cointelegraph.com/rss"],
    "entertain":["https://variety.com/feed/",
                "https://deadline.com/feed/"],
}

# ── Fetch Intervals (seconds) ─────────────────────────────────────
REDDIT_INTERVAL       = 300   # 5 minutes
HN_INTERVAL           = 300   # 5 minutes
RSS_INTERVAL          = 300   # 5 minutes
GOOGLE_TRENDS_INTERVAL= 1800  # 30 minutes
YOUTUBE_INTERVAL      = 1800  # 30 minutes
NITTER_INTERVAL       = 600   # 10 minutes

# ── Telegram Alert Settings ───────────────────────────────────────
DEFAULT_ALERT_THRESHOLD   = 80   # notify when virality crosses this
UPDATE_PUSH_INTERVAL      = 600  # 10 minutes (your choice)
TOP_TRENDS_COUNT          = 5    # how many trends per /trending command

# ── Virality Score Weights ────────────────────────────────────────
W_GROWTH_RATE     = 0.35
W_UPVOTE_VELOCITY = 0.20
W_COMMENT_VELOCITY= 0.20
W_CROSS_PLATFORM  = 0.15
W_RECENCY         = 0.10

# ── Model Settings ────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMOTION_MODEL    = "j-hartmann/emotion-english-distilroberta-base"
CLUSTER_WINDOW_HOURS = 3   # cluster posts from last N hours
MIN_CLUSTER_SIZE = 5       # minimum posts to form a topic cluster

# ── Database ──────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///database/viral_agent.db"

# ── Logging ───────────────────────────────────────────────────────
LOG_FILE = "logs/agent.log"
LOG_LEVEL = "INFO"