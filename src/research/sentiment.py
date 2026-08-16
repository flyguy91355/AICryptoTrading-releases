"""News and sentiment analysis module — keyword-based scoring with aggregation.

Ported from AITrading's SentimentAnalyzer with a handful of crypto-specific
keywords added to both lists -- the core keyword-overlap scoring approach is
asset-class-agnostic.
"""

from dataclasses import dataclass

from src.data.news_feed import NewsItem, SentimentScore


POSITIVE_KEYWORDS = {
    "beat", "beats", "exceeded", "upgrade", "upgraded", "strong", "growth",
    "record", "profit", "bullish", "surge", "surged", "rally", "rallied",
    "breakout", "innovative", "partnership", "acquisition", "dividend",
    "buyback", "outperform", "upside", "momentum", "optimistic", "raised",
    "guidance", "expansion", "contract", "win", "approval", "launch",
    # Crypto-specific
    "adoption", "etf", "institutional", "halving", "upgrade", "mainnet",
    "listing", "listed", "inflow", "inflows", "accumulation", "staking",
}

NEGATIVE_KEYWORDS = {
    "miss", "missed", "downgrade", "downgraded", "weak", "decline", "loss",
    "bearish", "crash", "plunge", "plunged", "selloff", "sell-off",
    "lawsuit", "investigation", "recall", "layoff", "layoffs", "cut",
    "warning", "delay", "delayed", "concern", "risk", "default", "debt",
    "underperform", "overvalued", "fraud", "bankruptcy", "violation",
    "fine", "penalty", "shortfall",
    # Crypto-specific
    "hack", "hacked", "exploit", "exploited", "rug", "rugpull", "delisted",
    "delisting", "outflow", "outflows", "depeg", "insolvent", "insolvency",
    "liquidation", "liquidations", "halted",
}

CATALYST_KEYWORDS = {
    "earnings", "fda", "approval", "merger", "acquisition", "partnership",
    "contract", "launch", "ipo", "split", "dividend", "buyback",
    "guidance", "forecast", "outlook",
    # Crypto-specific
    "etf", "halving", "mainnet", "upgrade", "listing", "airdrop", "fork",
}

RISK_KEYWORDS = {
    "lawsuit", "sec", "investigation", "recall", "warning", "fraud",
    "bankruptcy", "default", "subpoena", "fine", "penalty", "probe",
    "indictment", "class action", "whistleblower",
    # Crypto-specific
    "hack", "exploit", "rugpull", "delisted", "depeg", "insolvency",
    "liquidation", "regulatory",
}


@dataclass
class SentimentAnalysis:
    ticker: str
    overall_sentiment: float
    news_count: int
    positive_count: int
    negative_count: int
    key_catalysts: list[str]
    risk_headlines: list[str]
    summary: str = ""


class SentimentAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    async def analyze(self, news_items: list[NewsItem]) -> SentimentAnalysis:
        if not news_items:
            return SentimentAnalysis(
                ticker="",
                overall_sentiment=0.0,
                news_count=0,
                positive_count=0,
                negative_count=0,
                key_catalysts=[],
                risk_headlines=[],
                summary="No recent news available.",
            )

        ticker = news_items[0].ticker
        positive_count = 0
        negative_count = 0
        catalysts = []
        risks = []

        for item in news_items:
            score = self._score_item(item)
            item.sentiment = self._to_sentiment_enum(score)
            item.relevance_score = abs(score)

            if score > 0:
                positive_count += 1
            elif score < 0:
                negative_count += 1

            headline_lower = item.headline.lower()
            if any(kw in headline_lower for kw in CATALYST_KEYWORDS):
                catalysts.append(item.headline)
            if any(kw in headline_lower for kw in RISK_KEYWORDS):
                risks.append(item.headline)

        total = len(news_items)
        overall = (positive_count - negative_count) / total if total > 0 else 0.0

        summary_parts = [f"{total} articles analyzed."]
        if positive_count > negative_count:
            summary_parts.append(f"Sentiment skews positive ({positive_count} positive vs {negative_count} negative).")
        elif negative_count > positive_count:
            summary_parts.append(f"Sentiment skews negative ({negative_count} negative vs {positive_count} positive).")
        else:
            summary_parts.append("Sentiment is neutral/mixed.")

        if catalysts:
            summary_parts.append(f"{len(catalysts)} potential catalyst(s) detected.")
        if risks:
            summary_parts.append(f"{len(risks)} risk headline(s) detected.")

        return SentimentAnalysis(
            ticker=ticker,
            overall_sentiment=round(overall, 2),
            news_count=total,
            positive_count=positive_count,
            negative_count=negative_count,
            key_catalysts=catalysts[:5],
            risk_headlines=risks[:5],
            summary=" ".join(summary_parts),
        )

    def _score_item(self, item: NewsItem) -> float:
        text = f"{item.headline} {item.summary}".lower()
        words = set(text.split())

        pos = len(words & POSITIVE_KEYWORDS)
        neg = len(words & NEGATIVE_KEYWORDS)

        if pos == 0 and neg == 0:
            return 0.0

        return (pos - neg) / max(pos + neg, 1)

    @staticmethod
    def _to_sentiment_enum(score: float) -> SentimentScore:
        if score >= 0.5:
            return SentimentScore.VERY_POSITIVE
        if score > 0:
            return SentimentScore.POSITIVE
        if score <= -0.5:
            return SentimentScore.VERY_NEGATIVE
        if score < 0:
            return SentimentScore.NEGATIVE
        return SentimentScore.NEUTRAL
