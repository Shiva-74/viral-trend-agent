# 🔥 Viral Trend Intelligence Agent

An autonomous AI agent that monitors, clusters, scores, and predicts viral trends in real time across Reddit, Hacker News, Google Trends, YouTube, Mastodon, and Bluesky — delivered through a Telegram bot and Streamlit dashboard.

---

## What It Does

Most "trending" tools just show you the most upvoted posts. This system detects **what is growing fastest right now** — catching emerging narratives minutes after they appear, before they hit mainstream.

The agent runs a full intelligence pipeline every 10 minutes:

1. **Collects** posts from 6 sources simultaneously
2. **Embeds** every post title into a 384-dimensional semantic vector
3. **Clusters** posts into topic groups using HDBSCAN
4. **Tracks** cluster growth over time using snapshots
5. **Scores** each trend on a 0–100 virality scale
6. **Tags** the dominant emotion driving each trend
7. **Alerts** you on Telegram when something spikes

---

## Demo

```
🔥 TOP TRENDS RIGHT NOW

#1 TRUMP TARIFF ANNOUNCEMENT
████████████████░░░░ 82/100
📈 Growth: +340%  😡 Anger
💬 487 posts  🌐 5 sources
Sources: reddit, google_trends, rss_bbc, hackernews, mastodon

#2 OPENAI NEW MODEL LEAKED
██████████████░░░░░░ 71/100
📈 Growth: +180%  😲 Surprise
💬 312 posts  🌐 4 sources
Sources: reddit, hackernews, rss_theverge, bluesky
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                           │
│  Reddit (Arctic Shift) │ Hacker News │ Google Trends RSS   │
│  YouTube (yt-dlp)      │ Mastodon    │ Bluesky             │
│  RSS (BBC/Reuters/NYT/Verge/Ars/TechCrunch)                │
└────────────────────────────┬────────────────────────────────┘
                             │ every 5–30 min
┌────────────────────────────▼────────────────────────────────┐
│                    PROCESSING PIPELINE                      │
│  Text Cleaner → Sentence Embeddings → Cross-Platform Dedup │
└────────────────────────────┬────────────────────────────────┘
                             │ every 10 min
┌────────────────────────────▼────────────────────────────────┐
│                  INTELLIGENCE ENGINE                        │
│  HDBSCAN Clustering → Growth Rate → Virality Scorer        │
│  Emotion Detection  → Viral Predictor                      │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     OUTPUT LAYER                            │
│          Telegram Bot          Streamlit Dashboard          │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### Data Collection
- **6 sources** polled simultaneously on independent schedules
- **Geography filter** — US, UK, Canada, Australia, Europe only
- **Deduplication** — same post from multiple sources merged, not duplicated
- **Cross-platform detection** — story appearing on 3+ platforms gets a virality bonus

### Intelligence Pipeline
- **Sentence embeddings** using `all-MiniLM-L6-v2` (runs on CPU, no GPU needed)
- **HDBSCAN clustering** with automatic topic count — no need to specify K
- **Cluster continuity tracking** — same topic gets the same ID across runs, enabling true growth rate measurement
- **Virality formula:**

```
Score = (0.35 × growth_rate)
      + (0.20 × upvote_velocity)
      + (0.20 × comment_velocity)
      + (0.15 × cross_platform_score)
      + (0.10 × recency_decay)
```

- **Emotion detection** using `j-hartmann/emotion-english-distilroberta-base` — Fear, Anger, Joy, Surprise, Sadness, Disgust, Neutral

### Viral Predictor
Paste any text and get:
- Viral probability (0–100%)
- Dominant emotion + confidence
- Which platforms it will spread on
- Estimated time to peak
- Why it will/won't spread (reasoning breakdown)

### Telegram Bot Commands
| Command | Description |
|---|---|
| `/trending` | Top 5 trends with category filter keyboard |
| `/top10` | Full top 10 in detail |
| `/predict [text]` | Viral probability for any content |
| `/explain [keyword]` | Deep dive on a specific trend |
| `/alert set [score]` | Get notified when virality crosses threshold |
| `/alert set [score] [emotion] [category]` | Filtered alerts |
| `/stats` | System statistics and source breakdown |
| `/sources` | Active data sources and schedules |

### Streamlit Dashboard
- **Live Trends** — auto-refreshing cards with score breakdowns
- **Trend Timeline** — Plotly charts showing virality over time
- **Viral Predictor** — interactive predictor UI with visual breakdown
- **Statistics** — source distribution, category pie chart, cluster table

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Scheduler | APScheduler |
| Reddit | Arctic Shift API (no auth) |
| Hacker News | Official Firebase API |
| Google Trends | RSS feed (official) + Google News RSS |
| YouTube | yt-dlp search |
| Social | Mastodon API + Bluesky public API |
| News RSS | feedparser (BBC, Reuters, NYT, Verge, Ars, TechCrunch) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Clustering | HDBSCAN |
| Emotion | HuggingFace `distilroberta-base` |
| Database | SQLite (SQLAlchemy ORM) |
| Telegram Bot | python-telegram-bot v21 |
| Dashboard | Streamlit + Plotly |
| OCR (predictor) | pytesseract |

**Total API cost: $0** — every data source is free and unauthenticated.

---

## Project Structure

```
viral_agent/
├── fetchers/
│   ├── base.py                  # Base fetcher with retry, geo-filter, dedup
│   ├── reddit_fetcher.py        # Arctic Shift API
│   ├── hn_fetcher.py            # Hacker News Firebase API
│   ├── rss_fetcher.py           # BBC/Reuters/NYT/Verge/Ars/TechCrunch
│   ├── google_trends_fetcher.py # Google Trends RSS + Google News RSS
│   ├── youtube_fetcher.py       # yt-dlp search
│   └── nitter_fetcher.py        # Mastodon + Bluesky + Reddit social subs
│
├── processing/
│   ├── cleaner.py               # Text cleaning pipeline
│   ├── embedder.py              # Sentence transformer embeddings
│   └── deduplicator.py          # Cross-platform story matching
│
├── intelligence/
│   ├── clusterer.py             # HDBSCAN clustering + continuity tracking
│   ├── growth_detector.py       # Growth rate from snapshots
│   ├── virality_scorer.py       # 0-100 virality scoring engine
│   └── emotion_detector.py      # Emotion classification
│
├── predictor/
│   └── viral_predictor.py       # Viral probability predictor
│
├── bot/
│   └── telegram_bot.py          # Telegram bot + alert broadcaster
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
├── database/
│   └── models.py                # SQLAlchemy models (RawPost, Cluster, Snapshot, UserAlert)
│
├── logs/                        # Rotating log files
├── scheduler.py                 # Main entry point
├── config.py                    # All settings in one place
├── requirements.txt
└── .env                         # Telegram token (not committed)
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/yourusername/viral_agent
cd viral_agent
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install yt-dlp
```

### 3. Configure environment

Create `.env` in the project root:

```
TELEGRAM_BOT_TOKEN=your_token_from_botfather
```

To get a Telegram bot token:
1. Open Telegram and search `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the token into `.env`

### 4. Initialise the database

```bash
python database/models.py
```

### 5. Verify setup

```bash
python verify_setup.py
python test_phase1.py
```

### 6. Run

**Terminal 1 — Scheduler (data collection + intelligence pipeline):**
```bash
python scheduler.py
```

**Terminal 2 — Telegram Bot:**
```bash
python bot/telegram_bot.py
```

**Terminal 3 — Dashboard:**
```bash
streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`

---

## How the Virality Score Works

The score is **not** based on total upvotes. A post with 50,000 upvotes from yesterday scores lower than a cluster that went from 10 posts to 200 posts in the last 30 minutes.

**Growth rate is the most important signal (35% weight).** Everything else is supporting evidence.

| Signal | Weight | What it measures |
|---|---|---|
| Growth rate | 35% | % increase in cluster size over last 30 min |
| Upvote velocity | 20% | Upvotes per hour across cluster |
| Comment velocity | 20% | Comments per hour across cluster |
| Cross-platform | 15% | How many different sources confirmed the story |
| Recency | 10% | Exponential decay — older trends score lower |

Cross-platform scoring:
- 1 source only → 0.1 (heavily penalized — could be platform-specific noise)
- 2 sources → 0.4
- 3+ sources → scales to 1.0 + bonus if same story confirmed identical across platforms

---

## How Clustering Works

Every 10 minutes:

1. All embedded posts from the last 24 hours are loaded
2. HDBSCAN groups them by semantic similarity — no need to specify how many clusters
3. Each cluster's centroid is compared against existing clusters from the previous run
4. If similarity > 0.75, it's the **same topic continuing** (cluster ID preserved)
5. If no match, it's a **new emerging topic** (new cluster created)
6. Cluster size is snapshotted — this is what enables growth rate tracking

This continuity system is what separates real trend detection from just showing popular posts.

---

## Configuration

All settings are in `config.py`:

```python
# Geography — only western content
TARGET_COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "NL", "SE", "NO"]

# Fetch intervals (seconds)
REDDIT_INTERVAL        = 300   # 5 minutes
HN_INTERVAL            = 300   # 5 minutes
RSS_INTERVAL           = 300   # 5 minutes
GOOGLE_TRENDS_INTERVAL = 1800  # 30 minutes
YOUTUBE_INTERVAL       = 1800  # 30 minutes
NITTER_INTERVAL        = 600   # 10 minutes

# Virality score weights
W_GROWTH_RATE      = 0.35
W_UPVOTE_VELOCITY  = 0.20
W_COMMENT_VELOCITY = 0.20
W_CROSS_PLATFORM   = 0.15
W_RECENCY          = 0.10

# Clustering
CLUSTER_WINDOW_HOURS = 24
MIN_CLUSTER_SIZE     = 5
```

---

## Telegram Alert Examples

```
🚨 VIRAL ALERT

🔥 IRAN MILITARY STRAIT HORMUZ

📊 Virality: 91/100 (your alert: 80)
📈 Growth: +340% in 20 minutes
💬 Posts: 487
😨 Emotion: Fear
🌐 Sources: reddit, rss_bbc, google_trends

Use /explain iran for more detail.
```

```
🎯 VIRAL PREDICTION

📝 Input: New OpenAI model just leaked and destroys every benchmark

📊 Viral Probability: 84%
████████████████░░░░ 84%
🔥 Very Likely Viral

😲 Surprise (79% confidence)
⏱️  Peak estimate: 1–3 hours

🔗 Similar to trending:
  • openai model announcement
  • ai benchmark leak

💡 Why:
  • Similar to currently trending 'openai gpt release' (+180% growth, 71/100)
  • Viral language detected: Tech viral topic, Urgency signal, Superlative language
  • ⚡ Related content confirmed viral across 3+ platforms

🌐 Spread path: Twitter/X → Reddit/r/technology → Hacker News
```

---

## Limitations

- **Reddit data** comes via Arctic Shift which has a ~15 minute delay from live Reddit
- **YouTube** uses search queries rather than the official trending page (blocked for automated clients)
- **Nitter** instances are mostly dead in 2026 — replaced by Mastodon + Bluesky
- **Emotion detection** works best on full sentences; short cluster labels tend to score Neutral
- **Cluster quality** improves significantly after 2–3 hours of continuous data collection
- Growth rates are most meaningful after multiple pipeline runs have accumulated snapshots

---

## Requirements

```
APScheduler==3.10.4
requests==2.31.0
feedparser==6.0.11
beautifulsoup4==4.12.3
lxml==5.1.0
sentence-transformers==2.7.0
hdbscan==0.8.33
scikit-learn==1.4.2
numpy==1.26.4
transformers==4.40.0
torch==2.3.0
pytesseract==0.3.10
Pillow==10.3.0
SQLAlchemy==2.0.29
python-telegram-bot==21.3
streamlit==1.35.0
python-dotenv==1.0.1
loguru==0.7.2
plotly>=5.0.0
yt-dlp
```

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built as a personal AI agent project exploring real-time trend intelligence, semantic clustering, and multi-source data fusion.
