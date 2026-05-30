"""
zoro/sentiment.py — FinBERT sentiment scorer (Phase 2 Step 3)

Replaces VADER with ProsusAI/finbert — a BERT model fine-tuned on
financial phrases. Same interface, better signal.

VADER:   rule-based, 2014, generic English
FinBERT: transformer, trained on financial news/SEC filings, knows
         that "bearish divergence" is negative and "beat estimates" is positive

Output
------
get_sentiment_score() → float in [-1.0, +1.0]
    +1.0 = strongly bullish
     0.0 = neutral
    -1.0 = strongly bearish

First call downloads ~440 MB model (once, cached at ~/.cache/huggingface).
Subsequent calls are fast (~0.1s per headline on CPU).
"""
from __future__ import annotations

import time
import urllib.request
from typing import Optional

# ── Module-level cache ────────────────────────────────────────────────────────
_pipeline   = None          # transformers sentiment pipeline
_last_score: float = 0.0    # last computed score (returned on feed failure)
_last_fetch: float = 0.0    # unix timestamp of last fetch
_CACHE_TTL  = 300           # re-fetch headlines every 5 minutes


# ── RSS feeds (crypto-focused, no auth required) ──────────────────────────────
_FEEDS = [
    "https://feeds.feedburner.com/CoinDesk",
    "https://cointelegraph.com/rss",
    "https://cryptonews.com/news/feed/",
]


def _load_pipeline():
    """Load FinBERT pipeline once and cache it."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline
        print("[INFO] Loading FinBERT model (first run: ~30s download)...")
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,          # return all 3 labels with scores
            device=-1,           # CPU; change to 0 for GPU
            truncation=True,
            max_length=512,
        )
        print("[INFO] FinBERT loaded ✅")
        return _pipeline
    except ImportError:
        print("[WARN] sentiment: transformers not installed — "
              "run: pip install transformers torch")
        return None
    except Exception as e:
        print(f"[WARN] sentiment: FinBERT load failed ({e})")
        return None


def _fetch_headlines(n: int = 8) -> list[str]:
    """Fetch latest crypto headlines from RSS feeds."""
    headlines = []
    for url in _FEEDS:
        try:
            # Use feedparser if available, else basic urllib
            try:
                import feedparser
                feed = feedparser.parse(url)
                for entry in feed.entries[:n]:
                    title = getattr(entry, "title", "")
                    if title:
                        headlines.append(title)
            except ImportError:
                # Fallback: raw urllib (less reliable but no extra dep)
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    html = r.read().decode("utf-8", errors="ignore")
                import re
                titles = re.findall(r"<title>([^<]{10,200})</title>", html)
                headlines.extend(titles[1:n+1])   # skip feed title

        except Exception:
            continue

        if len(headlines) >= n:
            break

    return headlines[:n]


def _finbert_score(headlines: list[str]) -> float:
    """
    Run FinBERT on a list of headlines.
    Returns mean score in [-1, +1].
    positive label → +score
    negative label → -score
    neutral  label → 0
    """
    pipe = _load_pipeline()
    if pipe is None or not headlines:
        return 0.0

    scores = []
    try:
        results = pipe(headlines)
        for result in results:
            # result is a list of dicts: [{label, score}, ...]
            label_scores = {r["label"].lower(): r["score"] for r in result}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            # net score: positive pulls +, negative pulls -
            net = pos - neg
            scores.append(net)
    except Exception as e:
        print(f"[WARN] sentiment: FinBERT inference failed ({e})")
        return 0.0

    return float(sum(scores) / len(scores)) if scores else 0.0


def get_sentiment_score(force_refresh: bool = False) -> float:
    """
    Return a sentiment score in [-1.0, +1.0] based on latest crypto headlines.

    Uses a 5-minute cache so it doesn't re-fetch on every 60s scan.

    Parameters
    ----------
    force_refresh : bypass cache and fetch fresh headlines

    Returns
    -------
    float in [-1.0, +1.0]
        > +0.15  → bullish  (+pts to LONG confidence)
        < -0.15  → bearish  (+pts to SHORT confidence)
        else     → neutral  (no effect)
    """
    global _last_score, _last_fetch

    now = time.time()
    if not force_refresh and (now - _last_fetch) < _CACHE_TTL:
        return _last_score

    headlines = _fetch_headlines(n=8)
    if not headlines:
        print("[WARN] sentiment: no headlines fetched — using neutral 0.0")
        return 0.0

    score = _finbert_score(headlines)
    _last_score = float(max(-1.0, min(1.0, score)))
    _last_fetch = now

    label = "BULLISH" if score > 0.15 else "BEARISH" if score < -0.15 else "NEUTRAL"
    print(f"[INFO] FinBERT sentiment: {label} ({score:+.3f})  "
          f"[{len(headlines)} headlines]")

    return _last_score
