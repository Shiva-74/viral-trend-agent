# verify_setup.py — Run this to confirm everything installed correctly

print("Checking all imports...")

try:
    import requests; print("✅ requests")
    import feedparser; print("✅ feedparser")
    from pytrends.request import TrendReq; print("✅ pytrends")
    from bs4 import BeautifulSoup; print("✅ beautifulsoup4")
    from sentence_transformers import SentenceTransformer; print("✅ sentence-transformers")
    import hdbscan; print("✅ hdbscan")
    from transformers import pipeline; print("✅ transformers")
    from telegram import Bot; print("✅ python-telegram-bot")
    import streamlit; print("✅ streamlit")
    from dotenv import load_dotenv; print("✅ python-dotenv")
    from loguru import logger; print("✅ loguru")
    import sqlalchemy; print("✅ sqlalchemy")
    from apscheduler.schedulers.background import BackgroundScheduler; print("✅ APScheduler")
    print("\n✅ ALL IMPORTS SUCCESSFUL. Phase 0 complete.")
except ImportError as e:
    print(f"\n❌ FAILED: {e}")
    print("Run: pip install -r requirements.txt")