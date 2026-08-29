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

import pandas as pd
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


def _round_price(price: float) -> float:
    """Round a price to a meaningful number of decimal places regardless of magnitude.
    Normal prices (>= $0.01): 2 decimal places (e.g. $1.23).
    Sub-cent prices (SHIB, PEPE etc.): 4 significant figures so $0.000004470 stays
    $0.000004470 instead of being truncated to $0.00 by round(x, 2)."""
    if price == 0.0:
        return 0.0
    import math
    if abs(price) >= 0.01:
        return round(price, 2)
    # Find how many decimal places needed to show 4 significant figures
    magnitude = math.floor(math.log10(abs(price)))
    decimals = max(2, -magnitude + 3)   # e.g. 4.47e-6 → magnitude=-6 → decimals=9
    return round(price, decimals)


def _fmt_price(price: float) -> str:
    """String-formatting sibling of _round_price above, for prompt/display use
    (2026-08-29) -- same magnitude-aware decimal-place logic (mirrors engine.py's
    own private _fmt_price, duplicated here rather than cross-module imported
    since both are small, module-local formatters). Avoids relying on Python's
    default float-to-str, which can render a tiny sub-cent price in scientific
    notation instead of a readable decimal."""
    if price == 0.0:
        return "0.00"
    if abs(price) >= 0.01:
        return f"{price:.2f}"
    decimals = max(2, -math.floor(math.log10(abs(price))) + 3)
    return f"{price:.{decimals}f}"


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
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx: float = 0.0
    obv_trend_pct: float = 0.0


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


def format_technical_summary(price: float, quote_change_pct: float, technicals: "TechnicalIndicators") -> str:
    """Pure formatter -- the single shared source for the "TECHNICAL CONTEXT" prompt
    section (2026-08-29, ported from AITrading/AIShortTrading the same day -- see
    those projects' CLAUDE_HISTORY.md 2026-08-29 entries for the full design
    rationale, including the market-context-section incident this framing
    deliberately avoids repeating). Uses _fmt_price throughout (not a plain
    :.2f) since crypto prices routinely fall below $0.01 (SHIB, PEPE, etc.).

    The added MACD/ADX/OBV line and the confluence instruction are deliberately
    NEUTRAL and METHODOLOGY-ONLY -- they teach Claude HOW to weigh multiple
    technical signals together, but never state or imply WHAT the current
    numbers already mean. See tests/test_technical_summary_format.py's own
    regression-lock test, which asserts no bullish/bearish/favorable-flavored
    word can ever appear in this function's output, for any input."""
    obv_sign = "+" if technicals.obv_trend_pct >= 0 else ""
    return (
        f"Price: ${_fmt_price(price)} | 24h change: {quote_change_pct:+.2f}% | "
        f"SMA50: ${_fmt_price(technicals.sma_50)} | SMA200: ${_fmt_price(technicals.sma_200)} | "
        f"RSI: {technicals.rsi:.1f} | Support: ${_fmt_price(technicals.support_level)} | "
        f"Resistance: ${_fmt_price(technicals.resistance_level)} | "
        f"Avg Volume 30d: {technicals.avg_volume_30d:,} | "
        f"30d Realized Vol: {technicals.realized_vol_30d*100:.0f}% annualized\n"
        f"MACD: line {_fmt_price(technicals.macd_line)}, signal {_fmt_price(technicals.macd_signal)}, "
        f"histogram {_fmt_price(technicals.macd_histogram)} (histogram = line minus signal) | "
        f"ADX: {technicals.adx:.1f} (trend strength, 0-100 scale; higher = stronger "
        f"trend, regardless of direction) | "
        f"20-Day Volume Trend: {obv_sign}{technicals.obv_trend_pct:.1f}% "
        f"(positive = net buying volume, negative = net selling volume)\n"
        f"Read MACD, ADX, and the volume trend together, not independently: weigh a "
        f"reading more heavily when multiple of these point the same direction, and "
        f"treat a single indicator moving alone -- or any signal during a weak trend "
        f"(low ADX) -- with more caution. If they conflict, treat the technical "
        f"picture as genuinely mixed rather than picking whichever one supports a "
        f"preferred conclusion."
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

        macd_line, macd_signal, macd_histogram = self._compute_macd(closes)

        return TechnicalIndicators(
            ticker=ticker,
            sma_50=_round_price(sma_50),
            sma_200=_round_price(sma_200),
            rsi=round(rsi, 2),
            avg_volume_30d=avg_volume_30d,
            support_level=_round_price(support_level),
            resistance_level=_round_price(resistance_level),
            realized_vol_30d=round(vol_30d, 4),
            macd_line=_round_price(macd_line),
            macd_signal=_round_price(macd_signal),
            macd_histogram=_round_price(macd_histogram),
            adx=round(self._compute_adx(hist), 2),
            obv_trend_pct=round(self._compute_obv_trend_pct(closes, volumes), 2),
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

    @staticmethod
    def _compute_macd(closes, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float]:
        """Standard MACD (2026-08-29, ported from AITrading/AIShortTrading same day) --
        fast/slow EMA line, its own signal-line EMA, and their difference (histogram).
        Complements SMA/RSI/realized-vol with a momentum-divergence signal none of
        those capture. Returns (0.0, 0.0, 0.0) when there isn't enough history for
        the signal line's own EMA to have a real seed."""
        if len(closes) < slow + signal:
            return (0.0, 0.0, 0.0)
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (float(macd_line.iloc[-1]), float(signal_line.iloc[-1]), float(histogram.iloc[-1]))

    @staticmethod
    def _compute_adx(hist, period: int = 14) -> float:
        """Average Directional Index -- trend STRENGTH, not direction (2026-08-29,
        ported from AITrading/AIShortTrading same day). Uses simple rolling-mean
        smoothing for +DM/-DM/TR and the final DX average, matching this file's own
        RSI/realized-vol simplification style over strict recursive Wilder
        smoothing. Needs roughly 2*period bars; returns 0.0 when there isn't enough
        history, or a period has zero directional movement (a 0/0 DX)."""
        if len(hist) < period * 2:
            return 0.0
        high = hist["High"]
        low = hist["Low"]
        close = hist["Close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        prev_close = close.shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        tr_smooth = true_range.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()

        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        di_sum = plus_di + minus_di
        dx = (100 * (plus_di - minus_di).abs() / di_sum).where(di_sum != 0, 0.0)

        adx = dx.rolling(period).mean()
        result = adx.iloc[-1]
        return float(result) if not pd.isna(result) else 0.0

    @staticmethod
    def _compute_obv_trend_pct(closes, volumes, lookback: int = 20) -> float:
        """On-Balance Volume, expressed as net signed volume over the lookback
        window as a % of total volume traded in that window (2026-08-29, ported
        from AITrading/AIShortTrading same day) -- NOT raw OBV's own % change (a
        known-bad metric: OBV's cumulative level is arbitrary/path-dependent).
        Bounded [-100, 100], comparable across assets regardless of OBV's own
        cumulative history. Returns 0.0 when there isn't enough history, or the
        window's total volume is zero."""
        if len(closes) < lookback + 1:
            return 0.0
        direction = closes.diff().apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
        signed_volume = (direction * volumes).tail(lookback)
        total_volume = volumes.tail(lookback).sum()
        if total_volume == 0:
            return 0.0
        return float(signed_volume.sum() / total_volume * 100)
