"""Alpaca broker implementation for crypto paper and live trading via alpaca-trade-api SDK.

Adapted from AITrading's (the sibling stock project) own AlpacaBroker, which this
class's rate-limit retry / order-status mapping / replace-order / reconciliation
methods are ported from near-verbatim -- none of that is asset-class-specific.
Two real differences from the stock version, both confirmed against Alpaca's own
docs before writing this (not guessed):

1. **time_in_force**: Alpaca crypto orders only support `gtc` and `ioc` -- `day`
   is NOT supported at all. This is the opposite constraint from the stock
   version, which requires `day` (never `gtc`) specifically for fractional
   share quantities. Every order in this file uses `gtc`.
2. **Symbol format**: crypto pairs use a slash, e.g. `"BTC/USD"` (Alpaca's
   current recommended format; the legacy no-slash `"BTCUSD"` form still works
   too, but this project standardizes on the slash form everywhere -- set once,
   in the asset universe config, not normalized here).
"""

import asyncio
import logging
import os
import random
import uuid
from datetime import datetime

import alpaca_trade_api as tradeapi

from src.execution.broker import Broker, Order, OrderSide, OrderType, OrderStatus, AccountInfo
from src.data.market_data import _round_price

logger = logging.getLogger(__name__)

ALPACA_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
    "pending_new": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "done_for_day": OrderStatus.SUBMITTED,
    "replaced": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.SUBMITTED,
    "suspended": OrderStatus.PENDING,
    "calculated": OrderStatus.PENDING,
}

# Exponential backoff delays (seconds) for the rate-limit retry below, plus a small random
# jitter added at call time -- overridable in tests via monkeypatch to avoid real sleeps.
_RATE_LIMIT_RETRY_DELAYS = [0.5, 1.0, 2.0]


class AlpacaBroker(Broker):
    def __init__(self, config: dict):
        self.config = config
        self.paper = config["trading"]["paper_trading"]
        self.api: tradeapi.REST | None = None
        # Alpaca's list_positions()/list_orders()/get_order() responses always return
        # a ticker's raw internal symbol (e.g. "LINKUSD", no separator), regardless of
        # which format was used to submit the original order -- confirmed live
        # 2026-08-15 against a real held position, contradicting this file's own prior
        # assumption ("the legacy no-slash form still works too... not normalized
        # here") that round-tripping was never necessary. Without this, comparing a
        # locally-tracked "LINK/USD" position against Alpaca's real position set
        # (keyed by "LINKUSD") always mismatches -- update_position_prices read that
        # as "the position is gone" and closed it locally on the very next tick after
        # every successful buy, real fill or not. Built from the configured universe
        # (the same source of truth every other canonical ticker already comes from).
        self._alpaca_to_canonical = {
            asset["ticker"].replace("/", ""): asset["ticker"]
            for asset in config.get("universe", [])
        }

    def _to_canonical(self, alpaca_symbol: str) -> str:
        """Alpaca's raw returned symbol -> this project's canonical "TICKER/USD"
        form. Falls back to the raw symbol unchanged for anything outside the
        configured universe, rather than raising -- a stray symbol should never
        crash a broker call over a display/comparison nicety."""
        return self._alpaca_to_canonical.get(alpaca_symbol, alpaca_symbol)

    @staticmethod
    def _to_alpaca_symbol(ticker: str) -> str:
        """This project's canonical "TICKER/USD" -> Alpaca's own raw internal form,
        for querying/filtering by symbol (list_positions/list_orders itself always
        returns the raw form regardless of what was used at submission time, so
        matching on the raw form is the reliable direction for a filter)."""
        return ticker.replace("/", "")

    async def _call_with_rate_limit_retry(self, func, *args, **kwargs):
        """Runs a blocking Alpaca SDK call via asyncio.to_thread, with bounded
        exponential-backoff retry specifically on HTTP 429 (Alpaca's trading API rate
        limit: 200 requests/minute). Any other exception, or a 429 that persists past
        the retry budget, propagates unchanged."""
        from alpaca_trade_api.rest import APIError

        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except APIError as e:
                if e.status_code != 429 or attempt >= len(_RATE_LIMIT_RETRY_DELAYS):
                    raise
                delay = _RATE_LIMIT_RETRY_DELAYS[attempt] + random.uniform(0, 0.25)
                logger.warning(
                    "Alpaca rate limit hit (429) — retrying in %.2fs (attempt %d/%d)",
                    delay, attempt + 1, len(_RATE_LIMIT_RETRY_DELAYS),
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def connect(self):
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "")

        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

        if not base_url:
            base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")

        account = await self._call_with_rate_limit_retry(self.api.get_account)
        mode = "PAPER" if self.paper else "LIVE"
        logger.info("Connected to Alpaca (%s) — Account: %s, Equity: $%s",
                     mode, account.account_number, account.equity)

    async def disconnect(self):
        self.api = None
        logger.info("Disconnected from Alpaca")

    def create_trade_updates_stream(self):
        """Returns a real-time Alpaca trade_updates WebSocket Stream, or None if
        credentials aren't available. Not part of the abstract Broker interface --
        OrderManager accesses this defensively via getattr."""
        from alpaca_trade_api.stream import Stream

        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "")
        if not api_key or not secret_key:
            return None
        if not base_url:
            base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        return Stream(key_id=api_key, secret_key=secret_key, base_url=base_url)

    async def get_account(self) -> AccountInfo:
        account = await self._call_with_rate_limit_retry(self.api.get_account)
        return AccountInfo(
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            equity=float(account.equity),
            last_equity=float(account.last_equity),
        )

    async def submit_order(self, order: Order) -> Order:
        side = "buy" if order.side == OrderSide.BUY else "sell"

        # Use notional (dollar amount) for fractional market buys; qty for everything else
        if order.notional_value and order.side == OrderSide.BUY and order.order_type == OrderType.MARKET:
            kwargs = {
                "symbol": order.ticker,
                "notional": str(round(order.notional_value, 2)),
                "side": side,
                "type": "market",
                "time_in_force": "gtc",  # crypto: day is not supported at all
            }
        else:
            qty_str = str(round(order.quantity, 9)).rstrip("0").rstrip(".") if order.quantity % 1 else str(int(order.quantity))
            if not qty_str or qty_str == ".":
                qty_str = "0"
            kwargs = {
                "symbol": order.ticker,
                "qty": qty_str,
                "side": side,
            }

        if order.notional_value:
            pass  # kwargs already complete (notional market buy built above)

        elif order.order_type == OrderType.MARKET:
            kwargs["type"] = "market"
            kwargs["time_in_force"] = "gtc"

        elif order.order_type == OrderType.LIMIT:
            kwargs["type"] = "limit"
            kwargs["time_in_force"] = "gtc"
            # 2026-08-28, audit finding: a flat round(price, 2) silently zeroes out a
            # sub-cent price (SHIB, PEPE etc.) entirely -- _round_price (already used
            # elsewhere in this codebase for the identical class of asset) keeps
            # meaningful precision regardless of magnitude instead.
            kwargs["limit_price"] = str(_round_price(order.limit_price))

        elif order.order_type == OrderType.STOP:
            # Confirmed against Alpaca's own docs before writing this file: crypto only
            # supports market/limit/stop_limit -- a plain "stop" order type is NOT
            # supported and would be rejected. OrderManager only ever places
            # OrderType.STOP_LIMIT for crypto stop-losses; this branch exists purely for
            # Broker-interface completeness and is never exercised by this project's own
            # code. Left in (not deleted) in case a future non-crypto asset class is ever
            # added to this broker class.
            kwargs["type"] = "stop"
            kwargs["time_in_force"] = "gtc"
            kwargs["stop_price"] = str(_round_price(order.stop_price))

        elif order.order_type == OrderType.STOP_LIMIT:
            kwargs["type"] = "stop_limit"
            kwargs["time_in_force"] = "gtc"
            # Same sub-cent fix as the LIMIT branch above. Note this does NOT fully
            # close the audit finding's own DOGE example (a thin 0.5% STOP_LIMIT
            # buffer collapsing at prices already >= $0.01, where _round_price's
            # threshold still rounds to 2 decimals) -- that would need Alpaca's own
            # real crypto price-precision limits verified before widening precision
            # further, per this project's standing rule to confirm against Alpaca's
            # docs before changing order-placement code. Tracked as still open.
            kwargs["stop_price"] = str(_round_price(order.stop_price))
            kwargs["limit_price"] = str(_round_price(order.limit_price))

        elif order.order_type == OrderType.TRAILING_STOP:
            # Not currently supported by Alpaca for crypto (verify before ever using --
            # left in for interface completeness, matching the stock version's shape;
            # the graduated trailing stop this project ports over computes its own stop
            # price and places a plain STOP order instead of relying on this).
            kwargs["type"] = "trailing_stop"
            kwargs["time_in_force"] = "gtc"
            kwargs["trail_percent"] = str(order.trail_pct)

        else:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        if not order.client_order_id:
            order.client_order_id = (
                f"{order.ticker.replace('/', '')}-{order.side.value}-{order.order_type.value}-{uuid.uuid4().hex[:8]}"
            )
        kwargs["client_order_id"] = order.client_order_id

        result = await self._call_with_rate_limit_retry(lambda: self.api.submit_order(**kwargs))

        order.broker_order_id = result.id
        order.status = ALPACA_STATUS_MAP.get(result.status, OrderStatus.PENDING)
        order.submitted_at = datetime.now()

        if result.filled_avg_price is not None:
            order.filled_price = float(result.filled_avg_price)
        if result.filled_qty is not None:
            order.filled_quantity = float(result.filled_qty)
        if hasattr(result, "filled_at") and result.filled_at:
            try:
                order.filled_at = datetime.fromisoformat(str(result.filled_at).replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, TypeError):
                pass

        if order.notional_value:
            logger.info("Alpaca order submitted: %s %s $%.2f notional — ID: %s, Status: %s",
                        order.side.value, order.ticker, order.notional_value,
                        order.broker_order_id, order.status.value)
        else:
            logger.info("Alpaca order submitted: %s %s %.9g %s — ID: %s, Status: %s",
                        order.side.value, order.ticker, order.quantity, order.order_type.value,
                        order.broker_order_id, order.status.value)
        return order

    async def cancel_order(self, broker_order_id: str) -> bool:
        try:
            await self._call_with_rate_limit_retry(self.api.cancel_order, broker_order_id)
            logger.info("Alpaca order cancelled: %s", broker_order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel Alpaca order %s: %s", broker_order_id, e)
            return False

    async def replace_order(self, broker_order_id: str, qty: float | None = None,
                             stop_price: float | None = None, limit_price: float | None = None) -> str | None:
        """Replace an existing order's qty/stop_price/limit_price in place via Alpaca's
        PATCH /v2/orders/{id} -- avoids the cancel-then-resubmit settlement-lag race
        this project's stock sibling documented at length. Returns the NEW order's
        broker_order_id on success, or None on any failure (including a rejected
        replace) -- callers MUST treat None as "fall back to cancel+place"."""
        kwargs: dict = {}
        if qty is not None:
            qty_str = str(round(qty, 9)).rstrip("0").rstrip(".") if qty % 1 else str(int(qty))
            kwargs["qty"] = qty_str or "0"
        if stop_price is not None:
            kwargs["stop_price"] = str(_round_price(stop_price))
        if limit_price is not None:
            kwargs["limit_price"] = str(_round_price(limit_price))
        try:
            result = await self._call_with_rate_limit_retry(
                lambda: self.api.replace_order(broker_order_id, **kwargs))
            logger.info("Alpaca order replaced: %s -> %s", broker_order_id, result.id)
            return result.id
        except Exception as e:
            logger.warning("Failed to replace Alpaca order %s: %s", broker_order_id, e)
            return None

    async def get_order_status(self, broker_order_id: str) -> Order:
        result = await self._call_with_rate_limit_retry(self.api.get_order, broker_order_id)

        side = OrderSide.BUY if result.side == "buy" else OrderSide.SELL
        try:
            order_type = OrderType(str(result.type))
        except ValueError:
            order_type = OrderType.MARKET

        return Order(
            ticker=self._to_canonical(result.symbol),
            side=side,
            order_type=order_type,
            quantity=float(result.qty) if result.qty else 0.0,
            status=ALPACA_STATUS_MAP.get(result.status, OrderStatus.PENDING),
            filled_price=float(result.filled_avg_price) if result.filled_avg_price is not None else None,
            filled_quantity=float(result.filled_qty) if result.filled_qty is not None else 0.0,
            broker_order_id=result.id,
        )

    async def get_positions(self) -> list[dict]:
        positions = await self._call_with_rate_limit_retry(self.api.list_positions)
        return [
            {
                "ticker": self._to_canonical(p.symbol),
                "shares": float(p.qty),
                "entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pnl": float(p.unrealized_pl),
                "unrealized_pnl_pct": float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ]

    async def get_position(self, ticker: str) -> dict | None:
        """Single-ticker real-time position lookup -- `qty` is the true current total;
        `qty_available` already accounts for any resting order. Returns None if Alpaca
        has no open position for this ticker at all."""
        try:
            p = await self._call_with_rate_limit_retry(self.api.get_position, self._to_alpaca_symbol(ticker))
        except Exception:
            return None
        return {
            "ticker": self._to_canonical(p.symbol),
            "shares": float(p.qty),
            "qty_available": float(getattr(p, "qty_available", p.qty)),
        }

    async def get_open_orders(self) -> list[dict]:
        orders = await self._call_with_rate_limit_retry(lambda: self.api.list_orders(status="open"))
        return [
            {
                "order_id": o.id,
                "ticker": self._to_canonical(o.symbol),
                "side": o.side,
                "type": o.type,
                "qty": float(o.qty) if o.qty else (float(o.notional) if getattr(o, "notional", None) else 0.0),
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "stop_price": float(o.stop_price) if o.stop_price else None,
            }
            for o in orders
        ]

    async def get_closed_orders(self, symbols: list[str] | None = None, limit: int = 100) -> list[dict]:
        """Return filled/closed orders — used to verify apparent position closes and
        reconstruct real fill data. symbols scopes the query to specific tickers so an
        unrelated account-wide order-churn burst can't push a real fill out of the
        window (same reasoning as AITrading's own INSW incident)."""
        orders = await self._call_with_rate_limit_retry(
            lambda: self.api.list_orders(
                status="closed", limit=limit, direction="desc",
                symbols=[self._to_alpaca_symbol(s) for s in symbols] if symbols else None,
            )
        )
        return [
            {
                "order_id": o.id, "symbol": self._to_canonical(o.symbol), "side": o.side, "status": o.status,
                "order_type": o.type,
                "qty": float(o.qty) if o.qty else 0.0,
                "filled_qty": float(o.filled_qty) if o.filled_qty else 0.0,
                "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            }
            for o in orders
            if o.status in ("filled", "partially_filled")
        ]

    async def get_portfolio_history(self) -> list[dict]:
        """Real account equity over time, via Alpaca's own portfolio-history endpoint.
        Returns [] on any error — this is display-only, never allowed to affect trading."""
        try:
            history = await self._call_with_rate_limit_retry(
                lambda: self.api.get_portfolio_history(period="30D", timeframe="1H")
            )
            timestamps = history.timestamp or []
            equities = history.equity or []
            return [
                {"t": float(t), "equity": float(e)}
                for t, e in zip(timestamps, equities)
                if e is not None and float(e) != 0.0
            ]
        except Exception as e:
            logger.warning("get_portfolio_history failed: %s", e)
            return []

    async def get_portfolio_history_daily(self) -> list[dict]:
        """Full daily equity history since inception (period='all', timeframe='1D').
        Returns list of {date: 'YYYY-MM-DD', equity: float}, oldest first.
        Returns [] on any error — display-only, never affects trading."""
        try:
            from datetime import timezone
            history = await self._call_with_rate_limit_retry(
                lambda: self.api.get_portfolio_history(period="all", timeframe="1D")
            )
            timestamps = history.timestamp or []
            equities = history.equity or []
            result = []
            for t, e in zip(timestamps, equities):
                if e is None or float(e) == 0.0:
                    continue
                dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
                result.append({"date": dt.strftime("%Y-%m-%d"), "equity": round(float(e), 2)})
            return result
        except Exception as e:
            logger.warning("get_portfolio_history_daily failed: %s", e)
            return []

    async def get_quote(self, ticker: str) -> float | None:
        """Latest quote via Alpaca's own crypto quote endpoint. Confirmed against the
        installed SDK's real method (get_latest_crypto_quotes -- plural, takes a list,
        returns a dict keyed by symbol) rather than assumed from the stock broker's
        get_latest_quote/`.ap`/`.bp` shape, which crypto does NOT share -- the crypto
        v1beta3 quote entity remaps fields to `.ask_price`/`.bid_price` instead."""
        quotes = await self._call_with_rate_limit_retry(self.api.get_latest_crypto_quotes, [ticker])
        quote = quotes.get(ticker)
        if quote is None:
            return None
        price = quote.ask_price if quote.ask_price is not None else quote.bid_price
        return float(price) if price is not None else None
