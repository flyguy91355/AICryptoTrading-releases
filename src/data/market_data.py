"""Crypto price and technical data fetching via yfinance.

Ported from AITrading's MarketDataFetcher. Confirmed live (2026-08-15) that
yfinance genuinely supports crypto tickers in its own "BTC-USD"-style format
(dash, not slash) -- real historical bars, real current price -- so this
reuses the exact same SMA/RSI/support-resistance computation AITrading
already has tested, rather than hand-rolling a new pipeline against Alpaca's
separate crypto-bars endpoint. Alpaca is still the broker of record for
actual order placement and account/position state (AlpacaBroker) -- this
class is purely the research-side data source, same division of
responsibility AITrading's own MarketDataFetcher/AlpacaBroker split already
uses (AlpacaBroker.get_quote is explicitly not the live quote path there
either).

Dropped entirely: Financials/get_financials() (no earnings/fundamentals data
exists for crypto) and the Finnhub-quote fallback (Finnhub's `/quote`
endpoint is equity-specific; no crypto equivalent used here for Phase 1).
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)


# Yahoo Finance disambiguates a handful of crypto tickers with a numeric ID suffix
# when the plain symbol collides with another asset on their platform (a stock
# ticker, an index, a future, etc.) -- confirmed live via yfinance's own
# yf.Search() for each entry below, not guessed (2026-08-15, real recurring "possibly
# delisted" errors on every scan cycle for these two). Checked first in
# to_yfinance_symbol(); every other symbol falls through to the plain transform.
# UNI collides with United Parcel Service-adjacent tickers/UNH/UAL on Yahoo's
# search; GRT collides with several MSCI index symbols.
_YFINANCE_TICKER_OVERRIDES = {
    "UNI/USD": "UNI7083-USD",
    "GRT/USD": "GRT6719-USD",
}


def to_yfinance_symbol(alpaca_symbol: str) -> str:
    """Converts this project's canonical Alpaca-format symbol ("BTC/USD") to
    yfinance's own format ("BTC-USD"). Pure string transform for the overwhelming
    majority of symbols -- both formats use the same base/quote pair, just a
    different separator -- except the small override list above for tickers Yahoo
    itself disambiguates."""
    if alpaca_symbol in _YFINANCE_TICKER_OVERRIDES:
        return _YFINANCE_TICKER_OVERRIDES[alpaca_symbol]
    return alpaca_symbol.replace("/", "-")


@dataclass
class Quote:
    ticker: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    timestamp: datetime
    change_pct: float = 0.0


@dataclass
class TechnicalIndicators:
    ticker: str
    sma_50: float = 0.0
    sma_200: float = 0.0
    rsi: float = 0.0
    avg_volume_30d: int = 0
    support_level: float = 0.0
    resistance_level: float = 0.0
    realized_vol_30d: float = 0.0   # annualized 30-day daily-return std dev, e.g. 0.85 = 85%


@dataclass
class LongTermTrend:
    """Multi-year price context, same reasoning as AITrading's own version:
    get_technicals() only looks at 1 year of history, so a coin that's had a
    much larger multi-year run gets analyzed with no visibility into whether
    a recent move is a genuine breakout or a bounce back toward an old high/low."""
    ticker: str
    years: int
    high: float = 0.0
    high_date: str = ""
    low: float = 0.0
    low_date: str = ""
    current_price: float = 0.0


def _compute_long_term_trend(
    ticker: str, years: int, closes: list[tuple[str, float]], current_price: float,
) -> LongTermTrend:
    """Pure: given (date_str, close) pairs spanning `years`, find the high/low and their
    dates. Separated from get_long_term_trend's yfinance fetch so this logic is directly
    unit-testable without a network call."""
    if not closes:
        return LongTermTrend(ticker=ticker, years=years, current_price=current_price)
    high_date, high = max(closes, key=lambda p: p[1])
    low_date, low = min(closes, key=lambda p: p[1])
    return LongTermTrend(
        ticker=ticker, years=years,
        high=round(high, 2), high_date=high_date,
        low=round(low, 2), low_date=low_date,
        current_price=current_price,
    )


def format_long_term_trend_summary(trend: LongTermTrend) -> str:
    """Pure formatter -- builds the ANALYSIS_PROMPT section text from a LongTermTrend.
    Returns "" when there's no usable data, so the prompt section can be omitted
    cleanly rather than showing garbage."""
    if trend.high <= 0 or trend.low <= 0:
        return ""
    pct_off_high = (trend.current_price - trend.high) / trend.high * 100
    pct_off_low = (trend.current_price - trend.low) / trend.low * 100
    return (
        f"{trend.years}-Year Range: Low ${trend.low:.2f} ({trend.low_date}) | "
        f"High ${trend.high:.2f} ({trend.high_date}) | Current ${trend.current_price:.2f} "
        f"is {pct_off_high:+.1f}% vs the {trend.years}yr high and "
        f"{pct_off_low:+.1f}% vs the {trend.years}yr low."
    )


class MarketDataFetcher:
    def __init__(self, config: dict):
        self.config = config
        self._market_change_cache: tuple[float, float] | None = None

    async def get_market_change_pct(
        self, index_ticker: str = "BTC/USD", cache_seconds: float = 300.0,
    ) -> float | None:
        """Today's % change for a broad crypto-market proxy -- BTC, since it's the most
        liquid, widely-tracked asset and commonly used as crypto's own market bellwether
        (same "real, liquid ticker" reasoning AITrading's own SPY-based version uses for
        equities). Cached for cache_seconds so a batch run building many prompts shares
        one real fetch. Returns None on any fetch failure -- never fabricates a market
        condition."""
        now = time.monotonic()
        if self._market_change_cache is not None:
            cached_value, fetched_at = self._market_change_cache
            if now - fetched_at < cache_seconds:
                return cached_value
        try:
            quote = await self.get_quote(index_ticker)
            value = quote.change_pct
        except Exception as e:
            logger.warning("Market change fetch failed for %s: %s", index_ticker, e)
            return None
        self._market_change_cache = (value, now)
        return value

    async def get_quote(self, ticker: str) -> Quote:
        yf_symbol = to_yfinance_symbol(ticker)

        def _fetch():
            t = yf.Ticker(yf_symbol)
            return t.fast_info, t.history(period="1d")
        info, hist = await asyncio.to_thread(_fetch)

        if hist.empty:
            raise RuntimeError(f"No quote data available for {ticker} (yfinance symbol {yf_symbol})")

        row = hist.iloc[-1]
        if math.isnan(row["Close"]):
            raise RuntimeError(f"yfinance returned NaN close for {ticker}")
        prev_close = info.previous_close if hasattr(info, "previous_close") else row["Close"]
        change_pct = ((row["Close"] - prev_close) / prev_close * 100) if prev_close else 0.0

        return Quote(
            ticker=ticker,
            price=float(row["Close"]),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=int(row["Volume"]),
            timestamp=datetime.now(),
            change_pct=round(change_pct, 2),
        )

    async def get_historical(self, ticker: str, period: str = "1y", interval: str = "1d") -> list[dict]:
        yf_symbol = to_yfinance_symbol(ticker)
        hist = await asyncio.to_thread(lambda: yf.Ticker(yf_symbol).history(period=period, interval=interval))
        results = []
        for date, row in hist.iterrows():
            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "timestamp": date.timestamp(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        return results

    async def get_technicals(self, ticker: str) -> TechnicalIndicators:
        yf_symbol = to_yfinance_symbol(ticker)
        hist = await asyncio.to_thread(lambda: yf.Ticker(yf_symbol).history(period="1y"))

        if hist.empty:
            return TechnicalIndicators(ticker=ticker)

        closes = hist["Close"]
        volumes = hist["Volume"]

        sma_50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float(closes.mean())
        sma_200 = float(closes.tail(200).mean()) if len(closes) >= 200 else float(closes.mean())

        rsi = self._compute_rsi(closes)

        avg_volume_30d = int(volumes.tail(30).mean()) if len(volumes) >= 30 else int(volumes.mean())

        recent = closes.tail(20)
        support_level = float(recent.min())
        resistance_level = float(recent.max())

        # 30-day realized volatility: annualized std dev of daily log-returns.
        # Crypto trades 365 days/year; sqrt(365) is the standard annualizer.
        daily_returns = closes.pct_change().dropna()
        vol_30d = float(daily_returns.tail(30).std() * (365 ** 0.5)) if len(daily_returns) >= 10 else 0.0

        return TechnicalIndicators(
            ticker=ticker,
            sma_50=round(sma_50, 2),
            sma_200=round(sma_200, 2),
            rsi=round(rsi, 2),
            avg_volume_30d=avg_volume_30d,
            support_level=round(support_level, 2),
            resistance_level=round(resistance_level, 2),
            realized_vol_30d=round(vol_30d, 4),
        )

    async def get_long_term_trend(
        self, ticker: str, current_price: float, years: int = 5,
    ) -> LongTermTrend:
        yf_symbol = to_yfinance_symbol(ticker)
        try:
            hist = await asyncio.to_thread(
                lambda: yf.Ticker(yf_symbol).history(period=f"{years}y"))
        except Exception as e:
            logger.warning("Long-term trend fetch failed for %s: %s", ticker, e)
            return LongTermTrend(ticker=ticker, years=years, current_price=current_price)

        if hist.empty:
            return LongTermTrend(ticker=ticker, years=years, current_price=current_price)

        closes = [(idx.strftime("%Y-%m-%d"), float(row["Close"])) for idx, row in hist.iterrows()]
        return _compute_long_term_trend(ticker, years, closes, current_price)

    @staticmethod
    def _compute_rsi(closes, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 50.0

        deltas = closes.diff().dropna()
        gains = deltas.where(deltas > 0, 0.0)
        losses = (-deltas).where(deltas < 0, 0.0)

        # Seed with SMA of first period bars, then apply Wilder's smoothing
        avg_gain = float(gains.iloc[:period].mean())
        avg_loss = float(losses.iloc[:period].mean())
        for gain, loss in zip(gains.iloc[period:], losses.iloc[period:]):
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
