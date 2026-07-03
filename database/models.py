# database/models.py
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, Text, Boolean, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_URL

Base = declarative_base()


class RawPost(Base):
    """Every piece of content collected from any source."""
    __tablename__ = "raw_posts"

    id               = Column(String, primary_key=True)  # source:unique_id
    source           = Column(String, nullable=False)    # reddit/hn/rss/youtube/nitter/google
    category         = Column(String)                    # tech/news/memes/crypto/entertainment
    title            = Column(Text, nullable=False)
    body             = Column(Text, default="")
    url              = Column(String, default="")
    author           = Column(String, default="")
    upvotes          = Column(Integer, default=0)
    comments         = Column(Integer, default=0)
    upvote_ratio     = Column(Float, default=1.0)
    subreddit        = Column(String, default="")
    geo              = Column(String, default="")        # country/region tag
    fetched_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    published_at     = Column(DateTime, nullable=True)
    is_cleaned       = Column(Boolean, default=False)
    is_embedded      = Column(Boolean, default=False)
    embedding        = Column(Text, default="")          # JSON-serialized vector
    cluster_id       = Column(Integer, default=-1)
    cross_platform   = Column(Boolean, default=False)
    cross_platform_count = Column(Integer, default=1)

    __table_args__ = (
        Index("ix_source",     "source"),
        Index("ix_category",   "category"),
        Index("ix_fetched_at", "fetched_at"),
        Index("ix_cluster_id", "cluster_id"),
    )


class Cluster(Base):
    """A topic cluster — group of posts about the same thing."""
    __tablename__ = "clusters"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    label            = Column(String, default="")
    category         = Column(String, default="general")        # ← ADD THIS LINE
    first_seen       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_updated     = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    post_count       = Column(Integer, default=0)
    sources          = Column(Text, default="")
    top_posts        = Column(Text, default="")
    virality_score   = Column(Float, default=0.0)
    growth_rate      = Column(Float, default=0.0)
    emotion          = Column(String, default="")
    emotion_score    = Column(Float, default=0.0)
    is_active        = Column(Boolean, default=True)
    centroid         = Column(Text, default="")


class Snapshot(Base):
    """
    Cluster size logged every 10 minutes.
    This is what enables growth rate calculation.
    """
    __tablename__ = "snapshots"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id       = Column(Integer, nullable=False)
    post_count       = Column(Integer, default=0)
    virality_score   = Column(Float, default=0.0)
    upvote_sum       = Column(Integer, default=0)
    comment_sum      = Column(Integer, default=0)
    recorded_at      = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_snap_cluster",  "cluster_id"),
        Index("ix_snap_recorded", "recorded_at"),
    )


class UserAlert(Base):
    """Telegram user alert preferences."""
    __tablename__ = "user_alerts"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    telegram_chat_id = Column(String, nullable=False)
    alert_type       = Column(String, default="score")   # score / emotion / category
    threshold        = Column(Float, default=80.0)
    emotion_filter   = Column(String, default="")        # fear/anger/hype etc
    category_filter  = Column(String, default="")        # tech/news/memes etc
    is_active        = Column(Boolean, default=True)
    created_at       = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_triggered   = Column(DateTime, nullable=True)


def get_engine():
    return create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("✅ Database initialised")


if __name__ == "__main__":
    init_db()