"""News aggregation for crypto assets via NewsAPI.

Adapted from AITrading's NewsFeed. Finnhub's `/company-news` endpoint is
equity-specific (keyed by stock ticker) and has no real crypto equivalent
usable here, so it's dropped entirely -- NewsAPI's general `/v2/everything`
search (which AITrading already uses as its own fallback path) becomes the
primary and only source, querying by the asset's plain name (e.g.
"Bitcoin") rather than its trading-pair symbol, since a general news search
matches names, not tickers.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class SentimentScore(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class NewsItem:
    ticker: str
    headline: str
    summary: str
    source: str
    url: str
    published: datetime
    sentiment: SentimentScore = SentimentScore.NEUTRAL
    relevance_score: float = 0.0
    category: str = ""


class NewsFeed:
    def __init__(self, config: dict):
        self.config = config
        self.newsapi_key = os.getenv("NEWSAPI_API_KEY", "")
        # Per-ticker in-memory TTL cache (2026-08-16, real incident: this project's
        # 15-min scan interval across 22 assets means every asset gets re-searched up
        # to 96x/day with no caching at all -- 2,112 requests/day against a free-tier
        # NewsAPI key capped at 100/day, confirmed live as 376+ "429 Too Many Requests"
        # failures in a single day. Real crypto news doesn't meaningfully change
        # inside a few hours, so a cache this coarse costs essentially nothing in
        # freshness. research.news_cache_hours (default 6) keeps 22 assets' worth of
        # fetches at ~88/day (22 * 24/6) -- comfortably under a 100/day free key with
        # no shared-quota risk against AITrading's own key ever again.
        self._cache: dict[str, tuple[datetime, list[NewsItem]]] = {}

    async def get_asset_news(self, ticker: str, asset_name: str, days: int = 7) -> list[NewsItem]:
        """ticker is this project's canonical symbol (e.g. "BTC/USD"), used only to tag
        the returned NewsItems; asset_name (e.g. "Bitcoin") is what's actually searched,
        since NewsAPI matches article titles/content, not trading-pair symbols."""
        if not self.newsapi_key:
            return []
        cache_hours = self.config.get("research", {}).get("news_cache_hours", 6)
        cached = self._cache.get(ticker)
        if cached:
            fetched_at, items = cached
            if datetime.now() - fetched_at < timedelta(hours=cache_hours):
                return items
        items = await self._newsapi_search(ticker, asset_name, days)
        self._cache[ticker] = (datetime.now(), items)
        return items

    async def _newsapi_search(self, ticker: str, query: str, days: int = 7) -> list[NewsItem]:
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "searchIn": "title",
                        "from": from_date,
                        "sortBy": "relevancy",
                        "language": "en",
                        "pageSize": 20,
                        "apiKey": self.newsapi_key,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("NewsAPI search failed for '%s': %s", query, e)
            return []

        items = []
        for article in data.get("articles", []):
            try:
                published = datetime.fromisoformat(article["publishedAt"].replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError, KeyError):
                published = datetime.now()

            items.append(NewsItem(
                ticker=ticker,
                headline=article.get("title", ""),
                summary=article.get("description", "") or "",
                source=article.get("source", {}).get("name", ""),
                url=article.get("url", ""),
                published=published,
            ))
        items.sort(key=lambda n: n.published, reverse=True)
        return items
