# bot/telegram_bot.py
# Telegram bot — the main interface for the viral trend agent.
# Commands: /trending, /top10, /predict, /alert, /explain, /stats, /sources

import os
import asyncio
from datetime import datetime, timezone
from loguru import logger
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, ContextTypes, filters,
)
from telegram.constants import ParseMode

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models               import get_session, Cluster, RawPost, UserAlert
from intelligence.virality_scorer  import ViralityScorer
from intelligence.emotion_detector import EMOTION_EMOJI, EMOTION_SPREAD_PATTERN
from predictor.viral_predictor     import ViralPredictor
from config                        import TOP_TRENDS_COUNT, DEFAULT_ALERT_THRESHOLD

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Category filter keyboard
CATEGORY_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🤖 Tech",          callback_data="cat_tech"),
        InlineKeyboardButton("🌍 News",          callback_data="cat_news"),
        InlineKeyboardButton("😂 Memes",         callback_data="cat_memes"),
    ],
    [
        InlineKeyboardButton("₿  Crypto",        callback_data="cat_crypto"),
        InlineKeyboardButton("🎬 Entertainment", callback_data="cat_entertainment"),
        InlineKeyboardButton("🔥 All",           callback_data="cat_all"),
    ],
])


class ViralBot:

    def __init__(self):
        self.scorer    = ViralityScorer()
        self.predictor = ViralPredictor()
        self.app       = None

    # ── Formatters ────────────────────────────────────────────────────────────

    def _format_trend(self, trend: dict, rank: int,
                       detailed: bool = False) -> str:
        """Format a single trend for display."""
        emoji   = EMOTION_EMOJI.get(trend.get("emotion", "neutral"), "📰")
        score   = trend["total_score"]
        bar     = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        sources = trend.get("sources", [])
        src_str = ", ".join(list(set(sources))[:3])

        lines = [
            f"*#{rank} {trend['label'].upper()}*",
            f"`{bar}` {score:.0f}/100",
            f"📈 Growth: +{trend['growth_rate']:.0f}%  "
            f"{emoji} {(trend.get('emotion') or 'neutral').capitalize()}",
            f"💬 {trend['post_count']} posts  "
            f"🌐 {len(set(sources))} sources",
        ]

        if detailed:
            lines += [
                f"📡 *Sources:* {src_str}",
                f"🏷️  *Category:* {trend['category']}",
                f"⏱️  *First seen:* "
                f"{self._time_ago(trend.get('first_seen'))}",
            ]

        return "\n".join(lines)

    def _format_trends_list(self, trends: list[dict],
                             title: str = "🔥 TOP TRENDS") -> str:
        """Format a list of trends into a full message."""
        if not trends:
            return (
                "📭 *No trends detected yet.*\n\n"
                "The system is still collecting data. "
                "Check back in a few minutes."
            )

        now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
        lines = [f"*{title}*", f"_Updated: {now}_", ""]

        for i, trend in enumerate(trends, 1):
            lines.append(self._format_trend(trend, i))
            lines.append("")   # spacer

        lines.append("_Use /top10 for full list · /predict to test content_")
        return "\n".join(lines)

    @staticmethod
    def _time_ago(dt) -> str:
        if not dt:
            return "unknown"
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = datetime.now(timezone.utc) - dt
        mins = int(diff.total_seconds() / 60)
        if mins < 60:
            return f"{mins}m ago"
        hours = mins // 60
        if hours < 24:
            return f"{hours}h ago"
        return f"{hours // 24}d ago"

    # ── Command Handlers ──────────────────────────────────────────────────────

    async def cmd_start(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
        """Welcome message."""
        text = (
            "🔥 *Viral Trend Intelligence Agent*\n\n"
            "I track what's going viral across Reddit, "
            "Google Trends, Hacker News, YouTube, Mastodon "
            "and Bluesky — in real time.\n\n"
            "*Commands:*\n"
            "/trending — Top 5 viral trends right now\n"
            "/top10 — Full top 10 with details\n"
            "/predict — Will your content go viral?\n"
            "/alert — Set up viral spike alerts\n"
            "/explain — Deep dive on a trend\n"
            "/stats — System statistics\n"
            "/sources — Active data sources\n\n"
            "_Data updates every 10 minutes._"
        )
        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_trending(self, update: Update,
                           context: ContextTypes.DEFAULT_TYPE):
        """Show top 5 trends with category filter keyboard."""
        await update.message.reply_text(
            "🔍 *Select a category or view all:*",
            reply_markup   = CATEGORY_KEYBOARD,
            parse_mode     = ParseMode.MARKDOWN,
        )

    async def cmd_top10(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
        """Show top 10 trends in detail."""
        msg = await update.message.reply_text("⏳ Fetching top trends...")
        trends = self.scorer.get_top_trends(10)

        text = self._format_trends_list(trends, "🔥 TOP 10 TRENDS")
        await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN)

    async def cmd_predict(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE):
        """Viral predictor — analyze user-provided text."""
        # If text was provided with the command
        if context.args:
            input_text = " ".join(context.args)
            await self._run_prediction(update, input_text)
        else:
            await update.message.reply_text(
                "🎯 *Viral Predictor*\n\n"
                "Send me any text, topic, headline or meme description "
                "and I'll predict its viral potential.\n\n"
                "Usage:\n"
                "`/predict Your text here`\n\n"
                "Or just send any message after this and I'll analyze it.",
                parse_mode=ParseMode.MARKDOWN,
            )
            # Set context flag so next message is treated as prediction input
            context.user_data["awaiting_prediction"] = True

    async def _run_prediction(self, update: Update, text: str):
        """Run the predictor and send formatted result."""
        msg = await update.message.reply_text(
            f"🔮 Analyzing: _{text[:60]}_...",
            parse_mode=ParseMode.MARKDOWN,
        )
        result    = self.predictor.predict(text)
        formatted = self.predictor.format_result(result)
        await msg.edit_text(formatted, parse_mode=ParseMode.MARKDOWN)

    async def cmd_alert(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
        """Manage virality alerts."""
        chat_id = str(update.effective_chat.id)
        args    = context.args

        if not args:
            # Show current alerts and options
            session  = get_session()
            alerts   = (session.query(UserAlert)
                        .filter_by(telegram_chat_id=chat_id, is_active=True)
                        .all())

            lines = ["🔔 *Alert Settings*\n"]
            if alerts:
                for a in alerts:
                    lines.append(
                        f"• Score ≥ {a.threshold:.0f}"
                        + (f" | Emotion: {a.emotion_filter}" if a.emotion_filter else "")
                        + (f" | Category: {a.category_filter}" if a.category_filter else "")
                    )
            else:
                lines.append("No active alerts.")

            lines += [
                "",
                "*Set an alert:*",
                "`/alert set 80` — notify when score ≥ 80",
                "`/alert set 70 fear` — notify on fear trends ≥ 70",
                "`/alert set 75 anger tech` — fear + category filter",
                "`/alert off` — disable all alerts",
            ]
            await update.message.reply_text(
                "\n".join(lines), parse_mode=ParseMode.MARKDOWN
            )
            return

        if args[0] == "off":
            session = get_session()
            (session.query(UserAlert)
             .filter_by(telegram_chat_id=chat_id, is_active=True)
             .update({"is_active": False}))
            session.commit()
            await update.message.reply_text("🔕 All alerts disabled.")
            return

        if args[0] == "set":
            try:
                threshold = float(args[1]) if len(args) > 1 else DEFAULT_ALERT_THRESHOLD
                emotion   = args[2] if len(args) > 2 else ""
                category  = args[3] if len(args) > 3 else ""

                session = get_session()
                alert   = UserAlert(
                    telegram_chat_id = chat_id,
                    threshold        = threshold,
                    emotion_filter   = emotion,
                    category_filter  = category,
                    is_active        = True,
                )
                session.add(alert)
                session.commit()

                conf = f"✅ Alert set: notify when virality ≥ {threshold:.0f}"
                if emotion:
                    conf += f" + emotion={emotion}"
                if category:
                    conf += f" + category={category}"
                await update.message.reply_text(conf)

            except (IndexError, ValueError):
                await update.message.reply_text(
                    "❌ Usage: `/alert set 80` or `/alert set 75 fear tech`",
                    parse_mode=ParseMode.MARKDOWN,
                )

    async def cmd_explain(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE):
        """Deep dive on a specific trend."""
        if not context.args:
            await update.message.reply_text(
                "Usage: `/explain keyword`\nExample: `/explain iran`",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        keyword = " ".join(context.args).lower()
        session = get_session()

        # Find matching cluster
        clusters = (
            session.query(Cluster)
            .filter(
                Cluster.is_active == True,
                Cluster.label.contains(keyword),
            )
            .order_by(Cluster.virality_score.desc())
            .first()
        )

        if not clusters:
            await update.message.reply_text(
                f"🔍 No active trend matching '{keyword}'.\n"
                f"Use /trending to see current trends."
            )
            return

        c       = clusters
        emoji   = EMOTION_EMOJI.get(c.emotion or "neutral", "📰")
        sources = list(set((c.sources or "").split(",")))

        # Get sample posts
        import json
        top_ids = []
        try:
            top_ids = json.loads(c.top_posts or "[]")
        except Exception:
            pass

        sample_posts = (
            session.query(RawPost)
            .filter(RawPost.id.in_(top_ids))
            .order_by(RawPost.upvotes.desc())
            .limit(5)
            .all()
        ) if top_ids else []

        lines = [
            f"🔍 *TREND DEEP DIVE*",
            f"",
            f"*{c.label.upper()}*",
            f"",
            f"📊 Virality: *{c.virality_score:.1f}/100*",
            f"📈 Growth: +{c.growth_rate:.0f}%",
            f"💬 Posts: {c.post_count}",
            f"{emoji} Emotion: {(c.emotion or 'neutral').capitalize()} "
            f"({c.emotion_score or 0:.0%} confidence)",
            f"🏷️  Category: {c.category or 'general'}",
            f"🌐 Sources: {', '.join(sources[:5])}",
            f"⏱️  First seen: {self._time_ago(c.first_seen)}",
            f"",
        ]

        if sample_posts:
            lines.append("📌 *Sample posts:*")
            for p in sample_posts[:4]:
                lines.append(f"• _{p.title[:80]}_")

        spread = EMOTION_SPREAD_PATTERN.get(c.emotion or "neutral", "")
        if spread:
            lines += ["", f"💡 *Why it spreads:* {spread}"]

        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_stats(self, update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
        """System statistics."""
        session = get_session()
        total   = session.query(RawPost).count()
        embedded = session.query(RawPost).filter(
            RawPost.is_embedded == True
        ).count()
        cross   = session.query(RawPost).filter(
            RawPost.cross_platform == True
        ).count()
        clusters = session.query(Cluster).filter(
            Cluster.is_active == True
        ).count()

        # Source breakdown
        from sqlalchemy import func
        source_counts = (
            session.query(RawPost.source, func.count(RawPost.id))
            .group_by(RawPost.source)
            .order_by(func.count(RawPost.id).desc())
            .limit(8)
            .all()
        )

        lines = [
            "📊 *System Statistics*",
            "",
            f"📦 Total posts collected: *{total:,}*",
            f"🧠 Posts embedded: *{embedded:,}*",
            f"🌐 Cross-platform stories: *{cross:,}*",
            f"🔥 Active trend clusters: *{clusters}*",
            "",
            "*Posts by source:*",
        ]
        for source, count in source_counts:
            bar = "█" * min(int(count / (total / 20 + 1)), 15)
            lines.append(f"`{source:<20}` {bar} {count:,}")

        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )

    async def cmd_sources(self, update: Update,
                          context: ContextTypes.DEFAULT_TYPE):
        """Show active data sources."""
        text = (
            "📡 *Active Data Sources*\n\n"
            "✅ *Reddit* (via PullPush) — 5 min\n"
            "✅ *Hacker News* — 5 min\n"
            "✅ *RSS Feeds* (BBC/Reuters/NYT/Verge/Ars) — 5 min\n"
            "✅ *Google Trends* (US/GB/CA/AU) — 30 min\n"
            "✅ *YouTube* (search trending) — 30 min\n"
            "✅ *Mastodon* (trending posts/tags/links) — 10 min\n"
            "✅ *Bluesky* (trending topics) — 10 min\n\n"
            "🌍 *Geography filter:* US, UK, Canada, Australia, Europe\n"
            "🔄 *Pipeline runs:* every 10 minutes\n"
            "🧠 *Clustering:* HDBSCAN topic grouping\n"
            "📊 *Scoring:* Growth rate × Engagement × Cross-platform"
        )
        await update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN
        )

    # ── Callback Handler (category buttons) ──────────────────────────────────

    async def handle_callback(self, update: Update,
                              context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard button presses."""
        query    = update.callback_query
        await query.answer()
        data     = query.data

        if data.startswith("cat_"):
            category = data.replace("cat_", "")
            if category == "all":
                trends = self.scorer.get_top_trends(TOP_TRENDS_COUNT)
                title  = "🔥 TOP TRENDS — ALL CATEGORIES"
            else:
                trends = self.scorer.get_top_trends(
                    TOP_TRENDS_COUNT, category=category
                )
                cat_emoji = {
                    "tech": "🤖", "news": "🌍", "memes": "😂",
                    "crypto": "₿", "entertainment": "🎬",
                }.get(category, "🔥")
                title = f"{cat_emoji} TOP TRENDS — {category.upper()}"

            text = self._format_trends_list(trends, title)
            await query.edit_message_text(
                text, parse_mode=ParseMode.MARKDOWN
            )

    # ── Message Handler (for /predict flow) ──────────────────────────────────

    async def handle_message(self, update: Update,
                             context: ContextTypes.DEFAULT_TYPE):
        """Handle free-text messages for prediction."""
        if context.user_data.get("awaiting_prediction"):
            context.user_data["awaiting_prediction"] = False
            await self._run_prediction(update, update.message.text)
        else:
            # Default: treat any message as a prediction request
            await self._run_prediction(update, update.message.text)

    # ── Alert Broadcaster ─────────────────────────────────────────────────────

    async def broadcast_alerts(self, app: Application):
        """
        Check all active alerts and send notifications.
        Called every 10 minutes by the scheduler.
        """
        session = get_session()
        alerts  = session.query(UserAlert).filter_by(is_active=True).all()
        if not alerts:
            return

        trends = self.scorer.get_top_trends(20)
        if not trends:
            return

        now = datetime.now(timezone.utc)

        for alert in alerts:
            for trend in trends:
                # Check threshold
                if trend["total_score"] < alert.threshold:
                    continue

                # Check emotion filter
                if (alert.emotion_filter and
                        trend.get("emotion") != alert.emotion_filter):
                    continue

                # Check category filter
                if (alert.category_filter and
                        trend.get("category") != alert.category_filter):
                    continue

                # Check if we already alerted this user about this trend recently
                if alert.last_triggered:
                    triggered = alert.last_triggered
                    if triggered.tzinfo is None:
                        triggered = triggered.replace(tzinfo=timezone.utc)
                    mins_since = (now - triggered).total_seconds() / 60
                    if mins_since < 30:   # don't spam — max 1 alert per 30min
                        continue

                # Send alert
                emoji   = EMOTION_EMOJI.get(trend.get("emotion", ""), "📰")
                sources = list(set(trend.get("sources", [])))[:3]
                msg = (
                    f"🚨 *VIRAL ALERT*\n\n"
                    f"🔥 *{trend['label'].upper()}*\n\n"
                    f"📊 Virality: *{trend['total_score']:.0f}/100* "
                    f"(your alert: {alert.threshold:.0f})\n"
                    f"📈 Growth: +{trend['growth_rate']:.0f}%\n"
                    f"💬 Posts: {trend['post_count']}\n"
                    f"{emoji} Emotion: {(trend.get('emotion') or 'neutral').capitalize()}\n"
                    f"🌐 Sources: {', '.join(sources)}\n\n"
                    f"_Use /explain {trend['label'].split()[0]} for details_"
                )

                try:
                    await app.bot.send_message(
                        chat_id    = alert.telegram_chat_id,
                        text       = msg,
                        parse_mode = ParseMode.MARKDOWN,
                    )
                    alert.last_triggered = now
                    logger.info(
                        f"Alert sent to {alert.telegram_chat_id}: "
                        f"'{trend['label']}' ({trend['total_score']:.0f})"
                    )
                except Exception as e:
                    logger.error(f"Alert send error: {e}")

        try:
            session.commit()
        except Exception:
            session.rollback()

    # ── Scheduled Push (every 10 min) ─────────────────────────────────────────

    async def push_trends_update(self, app: Application):
        """
        Proactive push of top trends to users who have alerts set.
        Only sends if there are meaningful trends (score > 50).
        """
        trends = self.scorer.get_top_trends(5)
        if not trends or trends[0]["total_score"] < 50:
            return

        session = get_session()
        # Get unique chat IDs that have any active alert
        chat_ids = list(set(
            a.telegram_chat_id for a in
            session.query(UserAlert).filter_by(is_active=True).all()
        ))

        if not chat_ids:
            return

        text = self._format_trends_list(trends, "🔥 TREND UPDATE")
        for chat_id in chat_ids:
            try:
                await app.bot.send_message(
                    chat_id    = chat_id,
                    text       = text,
                    parse_mode = ParseMode.MARKDOWN,
                )
            except Exception as e:
                logger.error(f"Push update error for {chat_id}: {e}")

    # ── App Builder ───────────────────────────────────────────────────────────

    def build_app(self) -> Application:
        """Build and configure the Telegram application."""
        if not BOT_TOKEN:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not found in .env — "
                "get it from @BotFather on Telegram"
            )

        app = Application.builder().token(BOT_TOKEN).build()

        # Register command handlers
        app.add_handler(CommandHandler("start",    self.cmd_start))
        app.add_handler(CommandHandler("trending", self.cmd_trending))
        app.add_handler(CommandHandler("top10",    self.cmd_top10))
        app.add_handler(CommandHandler("predict",  self.cmd_predict))
        app.add_handler(CommandHandler("alert",    self.cmd_alert))
        app.add_handler(CommandHandler("explain",  self.cmd_explain))
        app.add_handler(CommandHandler("stats",    self.cmd_stats))
        app.add_handler(CommandHandler("sources",  self.cmd_sources))

        # Inline keyboard callbacks
        app.add_handler(CallbackQueryHandler(self.handle_callback))

        # Free text → prediction
        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_message,
        ))

        self.app = app
        return app


def run_bot():
    """Start the bot in polling mode."""
    bot = ViralBot()
    app = bot.build_app()

    logger.info("Telegram bot starting...")
    app.run_polling(
        allowed_updates = Update.ALL_TYPES,
        drop_pending_updates = True,
    )


if __name__ == "__main__":
    run_bot()