# dashboard/app.py
# Streamlit dashboard — visual interface for the viral trend agent.
# Run with: streamlit run dashboard/app.py

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models               import get_session, Cluster, RawPost, Snapshot
from intelligence.virality_scorer  import ViralityScorer
from intelligence.emotion_detector import EMOTION_EMOJI
from predictor.viral_predictor     import ViralPredictor

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Viral Trend Intelligence",
    page_icon  = "🔥",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .trend-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .score-high   { color: #ff4757; font-weight: bold; font-size: 1.4em; }
    .score-medium { color: #ffa502; font-weight: bold; font-size: 1.4em; }
    .score-low    { color: #2ed573; font-weight: bold; font-size: 1.4em; }
    .metric-label { color: #a4b0be; font-size: 0.85em; }
    .metric-value { color: #ffffff; font-size: 1.1em; font-weight: 600; }
    .stMetric > div { background: #1a1a2e; border-radius: 8px; padding: 8px; }
    div[data-testid="stSidebarContent"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)


# ── Cached Data Loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)   # refresh every 60 seconds
def load_trends(category: str = None, limit: int = 20) -> list[dict]:
    scorer = ViralityScorer()
    return scorer.get_top_trends(limit, category=category if category != "All" else None)

@st.cache_data(ttl=60)
def load_db_stats() -> dict:
    session  = get_session()
    total    = session.query(RawPost).count()
    embedded = session.query(RawPost).filter(RawPost.is_embedded == True).count()
    cross    = session.query(RawPost).filter(RawPost.cross_platform == True).count()
    clusters = session.query(Cluster).filter(Cluster.is_active == True).count()
    snaps    = session.query(Snapshot).count()

    # Source breakdown
    from sqlalchemy import func
    source_counts = (
        session.query(RawPost.source, func.count(RawPost.id))
        .group_by(RawPost.source)
        .order_by(func.count(RawPost.id).desc())
        .limit(10)
        .all()
    )
    session.close()
    return {
        "total":        total,
        "embedded":     embedded,
        "cross":        cross,
        "clusters":     clusters,
        "snapshots":    snaps,
        "sources":      [(s, c) for s, c in source_counts],
    }

@st.cache_data(ttl=120)
def load_cluster_history(cluster_id: int) -> pd.DataFrame:
    session = get_session()
    snaps   = (
        session.query(Snapshot)
        .filter(Snapshot.cluster_id == cluster_id)
        .order_by(Snapshot.recorded_at.asc())
        .all()
    )
    session.close()
    if not snaps:
        return pd.DataFrame()
    return pd.DataFrame([{
        "time":       s.recorded_at,
        "post_count": s.post_count,
        "upvotes":    s.upvote_sum,
        "comments":   s.comment_sum,
    } for s in snaps])

@st.cache_data(ttl=60)
def load_sample_posts(cluster_id: int, limit: int = 6) -> list[dict]:
    session = get_session()
    cluster = session.query(Cluster).filter(Cluster.id == cluster_id).first()
    if not cluster:
        return []
    top_ids = []
    try:
        top_ids = json.loads(cluster.top_posts or "[]")
    except Exception:
        pass
    posts = (
        session.query(RawPost)
        .filter(RawPost.id.in_(top_ids))
        .order_by(RawPost.upvotes.desc())
        .limit(limit)
        .all()
    ) if top_ids else []
    session.close()
    return [{
        "title":   p.title,
        "source":  p.source,
        "upvotes": p.upvotes,
        "url":     p.url,
    } for p in posts]


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-medium"
    return "score-low"

def time_ago(dt) -> str:
    if not dt:
        return "unknown"
    if hasattr(dt, "tzinfo") and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = datetime.now(timezone.utc) - dt
    mins = int(diff.total_seconds() / 60)
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    return f"{hours}h ago" if hours < 24 else f"{hours//24}d ago"

def virality_bar(score: float) -> str:
    filled = int(score / 5)
    empty  = 20 - filled
    return "█" * filled + "░" * empty


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image(
        "https://img.icons8.com/emoji/96/fire.png", width=60
    )
    st.sidebar.title("Viral Agent")
    st.sidebar.caption("Real-time trend intelligence")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        ["🔥 Live Trends", "📈 Trend Timeline",
         "🎯 Viral Predictor", "📊 Statistics"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.caption(
        f"Last refresh: {datetime.now().strftime('%H:%M:%S')}"
    )
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    return page


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_live_trends():
    st.title("🔥 Live Viral Trends")
    st.caption("Updated every 60 seconds · Powered by Reddit, HN, Google Trends, YouTube, Mastodon, Bluesky")

    # Category filter
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        category = st.selectbox(
            "Category",
            ["All", "tech", "news", "memes", "crypto", "entertainment", "general"],
            label_visibility="collapsed",
        )
    with col2:
        limit = st.selectbox("Show", [5, 10, 15, 20], index=1,
                             label_visibility="collapsed")
    with col3:
        st.write("")  # spacer

    trends = load_trends(category, limit)

    if not trends:
        st.info("⏳ No trends detected yet. The pipeline needs a few minutes to collect and cluster data.")
        return

    # Summary metrics row
    top = trends[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Virality Score", f"{top['total_score']:.0f}/100")
    m2.metric("Top Growth Rate",    f"+{top['growth_rate']:.0f}%")
    m3.metric("Active Trends",      len(trends))
    m4.metric("Top Sources",        top["source_count"])

    st.divider()

    # Trend cards
    for i, trend in enumerate(trends, 1):
        emoji   = EMOTION_EMOJI.get(trend.get("emotion", "neutral"), "📰")
        sources = list(set(trend.get("sources", [])))
        score   = trend["total_score"]

        with st.expander(
            f"#{i} {trend['label'].upper()} — {score:.0f}/100",
            expanded=(i <= 3)
        ):
            c1, c2 = st.columns([3, 2])

            with c1:
                # Virality bar
                st.markdown(
                    f"`{virality_bar(score)}` **{score:.1f}/100**"
                )

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("📈 Growth",  f"+{trend['growth_rate']:.0f}%")
                col_b.metric("💬 Posts",   trend["post_count"])
                col_c.metric("🌐 Sources", trend["source_count"])

                st.markdown(
                    f"{emoji} **Emotion:** {(trend.get('emotion') or 'neutral').capitalize()}  |  "
                    f"🏷️ **Category:** {trend['category']}  |  "
                    f"⏱️ **First seen:** {time_ago(trend.get('first_seen'))}"
                )

                # Sources list
                st.caption("📡 " + " · ".join(sources[:6]))

            with c2:
                # Score breakdown radar-style bar chart
                breakdown = trend.get("score_breakdown", {})
                if breakdown:
                    fig = go.Figure(go.Bar(
                        x = list(breakdown.values()),
                        y = ["Growth", "Upvotes", "Comments", "Cross-Platform", "Recency"],
                        orientation = "h",
                        marker_color = ["#ff4757", "#ffa502", "#2ed573", "#1e90ff", "#a29bfe"],
                        text = [f"{v:.0f}" for v in breakdown.values()],
                        textposition = "inside",
                    ))
                    fig.update_layout(
                        height      = 180,
                        margin      = dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor = "rgba(0,0,0,0)",
                        plot_bgcolor  = "rgba(0,0,0,0)",
                        xaxis = dict(range=[0, 100], showgrid=False,
                                     tickfont=dict(color="white")),
                        yaxis = dict(tickfont=dict(color="white")),
                        font  = dict(color="white"),
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"breakdown_{i}")

            # Sample posts
            posts = load_sample_posts(trend["cluster_id"])
            if posts:
                st.markdown("**📌 Sample posts:**")
                for p in posts[:4]:
                    source_tag = f"`{p['source']}`"
                    upvote_tag = f"⬆️ {p['upvotes']}" if p['upvotes'] > 0 else ""
                    if p.get("url"):
                        st.markdown(
                            f"- {source_tag} {upvote_tag} "
                            f"[{p['title'][:80]}]({p['url']})"
                        )
                    else:
                        st.markdown(
                            f"- {source_tag} {upvote_tag} {p['title'][:80]}"
                        )


def page_trend_timeline():
    st.title("📈 Trend Timeline")
    st.caption("Track how virality scores change over time")

    session  = get_session()
    clusters = (
        session.query(Cluster)
        .filter(Cluster.is_active == True)
        .order_by(Cluster.virality_score.desc())
        .limit(20)
        .all()
    )
    session.close()

    if not clusters:
        st.info("⏳ No active clusters yet.")
        return

    # Cluster selector
    cluster_names = {c.id: f"{c.label} ({c.virality_score:.0f}/100)" for c in clusters}
    selected_ids  = st.multiselect(
        "Select trends to compare",
        options   = list(cluster_names.keys()),
        default   = list(cluster_names.keys())[:3],
        format_func = lambda x: cluster_names[x],
    )

    if not selected_ids:
        st.info("Select at least one trend above.")
        return

    # Build combined dataframe
    all_data = []
    for cid in selected_ids:
        df = load_cluster_history(cid)
        if not df.empty:
            label = cluster_names[cid]
            df["trend"] = label
            all_data.append(df)

    if not all_data:
        st.info("Not enough snapshot data yet. More data accumulates every 10 minutes.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Post count over time
    fig1 = px.line(
        combined, x="time", y="post_count", color="trend",
        title    = "Post Count Over Time",
        template = "plotly_dark",
        labels   = {"post_count": "Posts", "time": "Time", "trend": "Trend"},
    )
    fig1.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

    # Upvotes over time
    fig2 = px.line(
        combined, x="time", y="upvotes", color="trend",
        title    = "Cumulative Upvotes Over Time",
        template = "plotly_dark",
        labels   = {"upvotes": "Total Upvotes", "time": "Time", "trend": "Trend"},
    )
    fig2.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    # Current snapshot table
    st.subheader("Current Snapshot")
    rows = []
    for c in clusters:
        if c.id in selected_ids:
            emoji = EMOTION_EMOJI.get(c.emotion or "neutral", "📰")
            rows.append({
                "Trend":     c.label,
                "Score":     f"{c.virality_score:.1f}",
                "Growth":    f"+{c.growth_rate:.0f}%",
                "Posts":     c.post_count,
                "Emotion":   f"{emoji} {c.emotion or 'neutral'}",
                "Category":  c.category or "general",
                "First Seen": time_ago(c.first_seen),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)


def page_viral_predictor():
    st.title("🎯 Viral Predictor")
    st.caption("Paste any text, headline, or topic — get a viral probability score")

    input_text = st.text_area(
        "Enter content to analyze",
        placeholder=(
            "Examples:\n"
            "• I just got fired — company replaced entire dev team with AI\n"
            "• Breaking: Major earthquake hits California, 7.8 magnitude\n"
            "• New OpenAI model just leaked and it destroys every benchmark\n"
            "• Trump announces 200% tariff on all European goods"
        ),
        height = 120,
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze = st.button("🔮 Analyze", type="primary", use_container_width=True)

    if analyze and input_text.strip():
        with st.spinner("Analyzing viral potential..."):
            predictor = ViralPredictor()
            result    = predictor.predict(input_text.strip())

        if "error" in result:
            st.error(result["error"])
            return

        prob    = result["probability"]
        verdict = result["verdict"]
        emotion = result["emotion"]
        emoji   = result["emotion_emoji"]

        # Main score display
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Viral Probability", f"{prob}%")
        c2.metric("Emotion",           f"{emoji} {emotion.capitalize()}")
        c3.metric("Peak Estimate",     result["peak_estimate"])

        # Probability bar
        color = "#ff4757" if prob >= 70 else "#ffa502" if prob >= 40 else "#2ed573"
        st.markdown(f"""
        <div style="background:#1a1a2e;border-radius:8px;padding:16px;margin:8px 0">
            <div style="font-size:1.2em;margin-bottom:8px">{verdict}</div>
            <div style="background:#0d1117;border-radius:4px;height:24px;overflow:hidden">
                <div style="background:{color};width:{prob}%;height:100%;
                            transition:width 0.5s ease;border-radius:4px"></div>
            </div>
            <div style="color:#a4b0be;margin-top:4px;font-size:0.85em">{prob}% viral probability</div>
        </div>
        """, unsafe_allow_html=True)

        # Score breakdown
        breakdown = result.get("score_breakdown", {})
        if breakdown:
            st.subheader("Score Breakdown")
            labels = ["Similarity", "Linguistic", "Emotion", "Timing", "Cross-Platform"]
            values = [
                breakdown.get("similarity",  0),
                breakdown.get("linguistic",  0),
                breakdown.get("emotion",     0),
                breakdown.get("time",        0),
                breakdown.get("cross_bonus", 0),
            ]
            fig = go.Figure(go.Bar(
                x            = labels,
                y            = values,
                marker_color = ["#ff4757", "#ffa502", "#2ed573", "#1e90ff", "#a29bfe"],
                text         = [f"{v:.0f}" for v in values],
                textposition = "outside",
            ))
            fig.update_layout(
                height        = 280,
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                yaxis         = dict(range=[0, 105], showgrid=True,
                                     gridcolor="#1a1a2e",
                                     tickfont=dict(color="white")),
                xaxis         = dict(tickfont=dict(color="white")),
                font          = dict(color="white"),
                margin        = dict(t=30, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Reasoning
        col_a, col_b = st.columns(2)
        with col_a:
            if result.get("reasoning"):
                st.subheader("💡 Why")
                for reason in result["reasoning"]:
                    st.markdown(f"• {reason}")

            if result.get("similar_trends"):
                st.subheader("🔗 Similar Trending Topics")
                for t in result["similar_trends"]:
                    st.markdown(f"• `{t}`")

        with col_b:
            if result.get("platforms"):
                st.subheader("🌐 Spread Path")
                platforms = result["platforms"]
                for j, p in enumerate(platforms):
                    arrow = " →" if j < len(platforms) - 1 else ""
                    st.markdown(f"**{j+1}.** {p}{arrow}")

            st.subheader("📊 Emotion Analysis")
            st.markdown(
                f"{emoji} **{emotion.capitalize()}**  \n"
                f"Confidence: {result['emotion_confidence']}%"
            )

    elif analyze:
        st.warning("Please enter some text to analyze.")


def page_statistics():
    st.title("📊 System Statistics")

    stats = load_db_stats()

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Posts",       f"{stats['total']:,}")
    c2.metric("Embedded",          f"{stats['embedded']:,}")
    c3.metric("Cross-Platform",    f"{stats['cross']:,}")
    c4.metric("Active Clusters",   stats["clusters"])
    c5.metric("Snapshots",         f"{stats['snapshots']:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Posts by Source")
        if stats["sources"]:
            sources_df = pd.DataFrame(
                stats["sources"], columns=["Source", "Count"]
            )
            fig = px.bar(
                sources_df, x="Count", y="Source",
                orientation  = "h",
                template     = "plotly_dark",
                color        = "Count",
                color_continuous_scale = "Viridis",
            )
            fig.update_layout(
                height        = 380,
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                showlegend    = False,
                coloraxis_showscale = False,
                yaxis         = dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Active Clusters by Category")
        session  = get_session()
        from sqlalchemy import func
        cat_counts = (
            session.query(Cluster.category, func.count(Cluster.id))
            .filter(Cluster.is_active == True)
            .group_by(Cluster.category)
            .all()
        )
        session.close()

        if cat_counts:
            cat_df = pd.DataFrame(cat_counts, columns=["Category", "Count"])
            fig2   = px.pie(
                cat_df, values="Count", names="Category",
                template = "plotly_dark",
                color_discrete_sequence = px.colors.qualitative.Set3,
            )
            fig2.update_layout(
                height        = 380,
                paper_bgcolor = "rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No cluster data yet.")

    # Active clusters table
    st.subheader("All Active Clusters")
    session  = get_session()
    clusters = (
        session.query(Cluster)
        .filter(Cluster.is_active == True)
        .order_by(Cluster.virality_score.desc())
        .all()
    )
    session.close()

    if clusters:
        rows = []
        for c in clusters:
            emoji = EMOTION_EMOJI.get(c.emotion or "neutral", "📰")
            rows.append({
                "Label":      c.label,
                "Score":      f"{c.virality_score:.1f}",
                "Growth":     f"+{c.growth_rate:.0f}%",
                "Posts":      c.post_count,
                "Category":   c.category or "general",
                "Emotion":    f"{emoji} {c.emotion or 'neutral'}",
                "Sources":    c.sources or "",
                "First Seen": time_ago(c.first_seen),
            })
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width = True,
            hide_index          = True,
        )
    else:
        st.info("No active clusters yet.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🔥 Live Trends":
        page_live_trends()
    elif page == "📈 Trend Timeline":
        page_trend_timeline()
    elif page == "🎯 Viral Predictor":
        page_viral_predictor()
    elif page == "📊 Statistics":
        page_statistics()

    # Auto-refresh every 60 seconds on Live Trends page
    if page == "🔥 Live Trends":
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()