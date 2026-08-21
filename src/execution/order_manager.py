"""Order lifecycle management.

Deliberately a much leaner Phase 1 implementation than AITrading's own
order_manager.py (2500+ lines) -- that file's size comes from a long history
of real, live-trading incidents (settlement-lag races, partial-fill
reconciliation, wash-trade retries, stale-order replace bugs) discovered
over months of running against real markets, each with its own targeted
fix. None of that history exists yet for this project. Per the design
spec's own stated approach ("get it operational so we can iron out the
problems"), this starts with the current, correct CORE exit-order design
AITrading eventually converged on after its own FTV incident (a resting
stop always covers 100% of the position; take-profit is an ACTIVE check,
never a second resting order) rather than reproducing every defensive patch
around it. Expect this file to grow the same way AITrading's did, once this
system has run against real crypto volatility for a while.

Two real, confirmed-against-Alpaca's-docs differences from AITrading's
broker interaction:
- Crypto only supports `market`, `limit`, and `stop_limit` order types --
  never plain `stop`. Every stop-loss placed here is STOP_LIMIT, with the
  limit price set a small buffer below the stop price so the order is
  likely to actually fill once triggered (a stop-limit whose limit price is
  never reached simply doesn't execute -- a real risk a plain stop-market
  order doesn't have).
- `time_in_force` is always "gtc" (crypto doesn't support "day" at all).
"""

import asyncio
import logging
import uuid
from datetime import datetime

from src.execution.broker import Order, OrderSide, OrderType, OrderStatus
from src.execution.alpaca_broker import AlpacaBroker
from src.decision.portfolio import Portfolio, Position
from src.decision.exit_strategy import derive_stop_pct, graduated_trail_pct

logger = logging.getLogger(__name__)

# How far below the real stop price to set a STOP_LIMIT order's limit price, as a
# fraction of the stop price (e.g. 0.005 = 0.5%) -- crypto's volatility means price can
# move meaningfully in the instant between a stop triggering and the limit order
# resting, so this needs more room than a typical equity stop-limit buffer would.
_STOP_LIMIT_BUFFER_PCT = 0.005

# Growing delays (seconds) execute_buy polls get_order_status for, when a buy's
# synchronous submit_order response comes back non-terminal (e.g. PENDING) rather
# than an immediate fill -- ~10s total. Overridable in tests via monkeypatch to
# avoid real sleeps, same pattern as AlpacaBroker's own _RATE_LIMIT_RETRY_DELAYS.
_BUY_FILL_POLL_DELAYS = [1.0, 2.0, 3.0, 4.0]

_TERMINAL_ORDER_STATUSES = {
    OrderStatus.FILLED, OrderStatus.PARTIAL, OrderStatus.CANCELLED,
    OrderStatus.REJECTED, OrderStatus.EXPIRED,
}


class OrderManager:
    def __init__(self, config: dict, portfolio: Portfolio):
        self.config = config
        self.portfolio = portfolio
        self.broker: AlpacaBroker | None = None
        # broker_order_id of each position's current resting stop order, keyed by
        # ticker -- needed so sync_exit_orders/check_take_profits know what to cancel
        # before placing a replacement. In-memory only for Phase 1 (rebuilt from a
        # fresh broker query at startup via _sync_portfolio, same pattern AITrading
        # uses for its own equivalent tracking).
        self._stop_order_ids: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, ticker: str) -> asyncio.Lock:
        if ticker not in self._locks:
            self._locks[ticker] = asyncio.Lock()
        return self._locks[ticker]

    async def connect(self):
        broker_name = self.config["trading"]["broker"]
        if broker_name == "alpaca":
            self.broker = AlpacaBroker(self.config)
        else:
            raise ValueError(f"Unknown broker: {broker_name}")
        await self.broker.connect()
        await self._sync_portfolio()

    async def _sync_portfolio(self):
        """Reconciles local position share counts against Alpaca's real positions at
        startup -- the local DB is authoritative for entry price/stop/targets/trade_id
        (Alpaca doesn't know about any of that), but Alpaca's own qty is authoritative
        for how many units are actually held, in case of any drift while the process
        was down."""
        try:
            real_positions = {p["ticker"]: p for p in await self.broker.get_positions()}
        except Exception as e:
            logger.warning("_sync_portfolio: failed to fetch real positions: %s", e)
            return

        for ticker, pos in list(self.portfolio.positions.items()):
            real = real_positions.get(ticker)
            if real is None:
                logger.warning(
                    "_sync_portfolio: %s exists locally but not at Alpaca — leaving "
                    "as-is (manual review needed, not auto-removed)", ticker)
                continue
            if abs(real["shares"] - pos.shares) > 1e-9:
                logger.info("_sync_portfolio: correcting %s shares %.9g -> %.9g",
                            ticker, pos.shares, real["shares"])
                pos.shares = real["shares"]
                await self.portfolio._save_position(pos)

        # Any real Alpaca position with no local record at all is flagged, not
        # auto-adopted -- Phase 1 has no safe way to reconstruct its real entry
        # price/stop/targets, so this needs a human to look at it.
        for ticker in real_positions:
            if ticker not in self.portfolio.positions:
                logger.warning(
                    "_sync_portfolio: %s exists at Alpaca but not locally — "
                    "needs manual review, not auto-adopted", ticker)

        # Rebuild _stop_order_ids from Alpaca's real open orders (fixed 2026-08-19,
        # LINK/USD incident) -- this dict is in-memory only and, before this fix, was
        # NEVER restored from a real broker query at startup, only the docstring above
        # claimed it was. A restart therefore permanently orphaned the tracking for any
        # position whose stop was placed in a now-dead process: sync_exit_orders saw no
        # existing_id, skipped the cancel-or-replace path entirely, and tried to place a
        # brand-new full-size stop against shares Alpaca had already fully reserved for
        # the still-live, now-forgotten original order -- rejected every single cycle
        # with "insufficient balance", while that real (but never ratcheted) stop just
        # sat at its original price for days. Confirmed live: LINK/USD's stop from
        # 2026-08-15 was orphaned by a restart 18 seconds later and stayed un-ratcheted
        # for 4+ days while every sync_exit_orders tick since failed the same way.
        #
        # Verified against Alpaca's real docs and the live account before shipping this
        # (per this project's standing rule to confirm Alpaca behavior, never guess):
        # qty_available is documented as qty minus what's reserved by open orders, which
        # matches the observed "insufficient balance" symptom exactly; crypto stop_limit
        # orders ARE replaceable via PATCH (only notional orders are replace-blocked, and
        # this project's stops are always qty-based); and get_open_orders()'s existing
        # _to_canonical(o.symbol) call was confirmed correct against the real account --
        # Alpaca's orders endpoints return crypto symbols WITH the slash ("LINK/USD"),
        # unlike the positions endpoint (which returns "LINKUSD"), so the symbol survives
        # unchanged through _to_canonical's no-slash-keyed lookup table via its fallback
        # path rather than an actual table hit. One real, still-open limit: get_open_orders
        # calls list_orders(status="open") with no explicit `limit`, so it defaults to
        # Alpaca's own page size of 50 -- fine at this project's current single-digit
        # position count, but would need an explicit higher limit if the held-position
        # count ever approaches that.
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as e:
            logger.warning("_sync_portfolio: failed to fetch open orders: %s", e)
            return

        for o in open_orders:
            ticker = o["ticker"]
            if ticker not in self.portfolio.positions:
                continue
            if o["side"] != "sell" or o["type"] not in ("stop_limit", "stop"):
                continue
            if ticker in self._stop_order_ids:
                logger.warning(
                    "_sync_portfolio: multiple resting stop orders found for %s at "
                    "Alpaca (%s and %s) -- tracking the most recently seen one; the "
                    "other needs manual review", ticker, self._stop_order_ids[ticker],
                    o["order_id"])
            self._stop_order_ids[ticker] = o["order_id"]

    async def _await_terminal_fill(self, order: Order) -> Order:
        """A synchronous submit_order response can still be non-terminal (PENDING,
        or any of Alpaca's other in-flight statuses) even though the order settles a
        moment later -- confirmed live 2026-08-15 (LINK/USD incident): treating that
        as an outright failure abandoned a real fill with zero local tracking and
        zero stop-loss protection, because nothing ever checked back. Polls
        get_order_status with growing waits (~10s total budget) until the order
        reaches a genuinely terminal state, and returns whatever the last observed
        status was either way -- a caller that still sees a non-terminal status
        after this made a real, bounded effort, not a lone synchronous guess."""
        if order.status in _TERMINAL_ORDER_STATUSES:
            return order
        for delay in _BUY_FILL_POLL_DELAYS:
            await asyncio.sleep(delay)
            try:
                order = await self.broker.get_order_status(order.broker_order_id)
            except Exception as e:
                logger.warning("_await_terminal_fill: status poll failed for %s: %s", order.ticker, e)
                continue
            if order.status in _TERMINAL_ORDER_STATUSES:
                return order
        return order

    async def execute_buy(self, ticker: str, asset_name: str, entry_price: float,
                           stop_loss: float, take_profit_targets: list[float],
                           position_size_dollars: float,
                           final_trail_pct: float | None = None,
                           signal=None) -> bool:
        """Places a notional market buy, and on a real fill, records the position and
        places its stop-loss order. Returns True on success. Any failure logs and
        returns False -- callers must not assume the position was opened."""
        order = Order(
            ticker=ticker, side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_value=position_size_dollars,
        )
        try:
            result = await self.broker.submit_order(order)
        except Exception as e:
            logger.error("execute_buy: order submission failed for %s: %s", ticker, e)
            return False

        result = await self._await_terminal_fill(result)
        if result.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            logger.warning("execute_buy: %s order not filled after polling (status=%s)", ticker, result.status)
            return False

        real_shares = result.filled_quantity
        real_price = result.filled_price or entry_price
        if real_shares <= 0:
            logger.warning("execute_buy: %s filled with zero shares — treating as failed", ticker)
            return False

        trade_id = uuid.uuid4().hex
        # "Why AI Bought This" snapshot (2026-08-21, same feature as AITrading's own) --
        # captured once, here, from whatever real report/numbers the caller's signal
        # actually carries. signal is None for any caller that doesn't have one (there
        # are none today, but this keeps the parameter genuinely optional rather than
        # assuming every future caller has a TradeSignal) -- falls back to empty/None
        # fields rather than guessing.
        _report = getattr(signal, "research_report", None) if signal else None
        position = Position(
            ticker=ticker,
            shares=real_shares,
            entry_price=real_price,
            current_price=real_price,
            stop_loss=stop_loss,
            take_profit_targets=list(take_profit_targets),
            sector="",  # no crypto category system in Phase 1
            opened_at=datetime.now(),
            t1_target_price=take_profit_targets[0] if len(take_profit_targets) > 0 else None,
            t2_target_price=take_profit_targets[1] if len(take_profit_targets) > 1 else None,
            trade_id=trade_id,
            final_trail_pct=final_trail_pct,
            buy_thesis=_report.thesis if _report else "",
            buy_reasoning=_report.reasoning if _report else "",
            buy_conviction=signal.conviction if signal else None,
            buy_signal=(signal.signal.value if signal and hasattr(signal.signal, "value")
                        else str(signal.signal) if signal else ""),
            buy_rr=getattr(signal, "rr", None) if signal else None,
            buy_required_rr=getattr(signal, "required_rr", None) if signal else None,
            buy_fair_value=_report.fair_value_estimate if _report else None,
        )
        await self.portfolio.add_position_async(position)
        await self._log_buy(ticker, real_shares, real_price, trade_id)

        stop_id = await self._place_stop_order(ticker, real_shares, stop_loss)
        if stop_id:
            self._stop_order_ids[ticker] = stop_id
        else:
            logger.error(
                "execute_buy: %s position opened but stop-loss placement FAILED — "
                "unprotected until the next sync_exit_orders cycle", ticker)

        logger.info("Bought %.9g %s @ $%.4f ($%.2f notional)", real_shares, ticker, real_price, position_size_dollars)
        return True

    async def _log_buy(self, ticker: str, shares: float, price: float, trade_id: str):
        if not self.portfolio._db:
            return
        await self.portfolio._db.execute(
            "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
            "VALUES (?, 'BUY', ?, ?, NULL, ?, 'Buy', ?)",
            (ticker, shares, price, datetime.now().isoformat(), trade_id),
        )
        await self.portfolio._db.commit()

    async def _place_stop_order(self, ticker: str, shares: float, stop_price: float) -> str | None:
        """STOP_LIMIT only -- see this module's own docstring for why plain STOP isn't
        an option for crypto. Returns the broker_order_id on success, None on failure
        (logged, never raised -- callers treat None as "this position is currently
        unprotected, needs remediation").

        Retries once against the broker's own real available quantity on an
        "insufficient balance" rejection (2026-08-21, real BTC/USD incident: every
        attempt requested 0.012319033 while Alpaca's real position only ever held
        0.012288235, confirmed live via a direct query -- the position sat completely
        unprotected for 100+ minutes with the identical rejection every single retry,
        since nothing ever corrected the requested quantity). Mirrors AITrading's own
        established pattern for this exact class of settlement/rounding drift between
        a locally-tracked quantity and what the broker actually holds -- see that
        project's "Order API hardening" history. Also corrects the locally-tracked
        Position.shares to the real value, not just this one retried order, since
        every other part of the system (P&L, a future sell, the next sync_exit_orders
        tick) needs the real number too, not just this call."""
        limit_price = stop_price * (1 - _STOP_LIMIT_BUFFER_PCT)
        order = Order(
            ticker=ticker, side=OrderSide.SELL, order_type=OrderType.STOP_LIMIT,
            quantity=shares, stop_price=stop_price, limit_price=limit_price,
        )
        try:
            result = await self.broker.submit_order(order)
            return result.broker_order_id
        except Exception as e:
            if "insufficient balance" not in str(e).lower():
                logger.error("_place_stop_order: failed for %s: %s", ticker, e)
                return None
            logger.warning(
                "_place_stop_order: %s insufficient balance for requested %.9g -- "
                "checking broker's real available quantity", ticker, shares)

        try:
            real_positions = {p["ticker"]: p["shares"] for p in await self.broker.get_positions()}
        except Exception as e2:
            logger.error("_place_stop_order: %s real-position lookup failed after "
                          "insufficient-balance rejection: %s", ticker, e2)
            return None

        real_qty = real_positions.get(ticker)
        if real_qty is None or real_qty <= 0:
            logger.error("_place_stop_order: %s not found in real positions after "
                          "insufficient-balance rejection -- giving up", ticker)
            return None

        logger.warning("_place_stop_order: %s retrying with broker's real available "
                        "%.9g (was requesting %.9g)", ticker, real_qty, shares)
        if ticker in self.portfolio.positions:
            self.portfolio.positions[ticker].shares = real_qty
            await self.portfolio._save_position(self.portfolio.positions[ticker])

        order.quantity = real_qty
        try:
            result = await self.broker.submit_order(order)
            return result.broker_order_id
        except Exception as e3:
            logger.error("_place_stop_order: retry with real qty also failed for %s: %s", ticker, e3)
            return None

    async def sync_exit_orders(self):
        """Runs periodically (every tick of the main loop): for each open position,
        recompute the graduated trailing-stop price and, if it's moved meaningfully,
        replace the resting stop order. Uses AlpacaBroker.replace_order (in-place PATCH)
        first, falling back to cancel+place only if replace fails or no stop is
        currently tracked -- avoids the settlement-lag race a cancel-then-resubmit can
        hit (same reasoning as AITrading's own MET-incident fix)."""
        rm_cfg = self.config.get("risk_management", {})
        tp_cfg = self.config.get("take_profit", {})
        default_stop_pct = tp_cfg.get("stop_loss_pct", 12.0)
        final_pct = tp_cfg.get("final_tranche_trail_pct", 1.5)
        follow_tp = rm_cfg.get("trailing_stop_follow_tp_targets", False)

        for ticker, pos in list(self.portfolio.positions.items()):
            async with self._lock_for(ticker):
                if pos.current_price <= pos.entry_price:
                    continue  # not yet in profit -- ride the plain stop_loss, no trailing yet

                start_pct = derive_stop_pct(pos.entry_price, pos.stop_loss, default_stop_pct)
                t3 = pos.take_profit_targets[-1] if pos.take_profit_targets else None
                position_final_pct = pos.final_trail_pct if pos.final_trail_pct is not None else final_pct
                trail_pct = graduated_trail_pct(
                    pos.entry_price, pos.current_price, start_pct, position_final_pct,
                    pos.t1_target_price, pos.t2_target_price, t3, follow_tp,
                )
                new_stop = round(pos.current_price * (1 - trail_pct / 100), 6)
                if pos.trailing_stop is not None and new_stop <= pos.trailing_stop:
                    continue  # one-way ratchet -- never loosen

                # pos.trailing_stop is only updated (and persisted) once the broker
                # action actually succeeds -- fixed 2026-08-19, LINK/USD incident.
                # Previously this was set and saved BEFORE attempting the replace/place
                # below, so a failed placement (e.g. "insufficient balance") still left
                # the local record looking like the ratchet had happened. That silently
                # broke the one-way-ratchet check above for every later tick too: a
                # freshly computed new_stop would get compared against the already-
                # inflated (but never actually applied) trailing_stop, found not higher,
                # and skipped -- permanently masking the fact that the REAL resting
                # order was still stuck at a stale price.
                limit_price = new_stop * (1 - _STOP_LIMIT_BUFFER_PCT)
                existing_id = self._stop_order_ids.get(ticker)
                replaced = None
                if existing_id:
                    replaced = await self.broker.replace_order(
                        existing_id, stop_price=new_stop, limit_price=limit_price)
                if replaced:
                    self._stop_order_ids[ticker] = replaced
                    pos.trailing_stop = new_stop
                    await self.portfolio._save_position(pos)
                    continue

                # Replace failed or nothing tracked -- fall back to cancel+place
                if existing_id:
                    await self.broker.cancel_order(existing_id)
                    await asyncio.sleep(1.0)  # settlement pause, same reasoning as AITrading's own
                new_id = await self._place_stop_order(ticker, pos.shares, new_stop)
                if new_id:
                    self._stop_order_ids[ticker] = new_id
                    pos.trailing_stop = new_stop
                    await self.portfolio._save_position(pos)
                else:
                    logger.error("sync_exit_orders: failed to (re)place stop for %s", ticker)

    async def check_take_profits(self):
        """Active take-profit check (runs every tick, same design AITrading converged
        on after its FTV incident): when price crosses the next open target, cancel the
        current stop, market-sell that one tranche (1/3 of the position's ORIGINAL
        target count -- i.e. shares/len(remaining_targets)), record the fill, and place
        a fresh 100%-of-the-remainder stop. Nothing but the stop ever rests, so there's
        never a coexistence race to manage."""
        for ticker, pos in list(self.portfolio.positions.items()):
            if not pos.take_profit_targets:
                continue
            async with self._lock_for(ticker):
                if pos.current_price < pos.take_profit_targets[0]:
                    continue

                target_price = pos.take_profit_targets[0]
                tranche_shares = round(pos.shares / len(pos.take_profit_targets), 9)
                remaining_shares = round(pos.shares - tranche_shares, 9)

                existing_id = self._stop_order_ids.pop(ticker, None)
                if existing_id:
                    await self.broker.cancel_order(existing_id)
                    await asyncio.sleep(1.0)

                sell_order = Order(
                    ticker=ticker, side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=tranche_shares,
                )
                try:
                    result = await self.broker.submit_order(sell_order)
                except Exception as e:
                    logger.error("check_take_profits: sell failed for %s: %s", ticker, e)
                    # Re-arm the stop we just cancelled so the position isn't left bare
                    new_id = await self._place_stop_order(ticker, pos.shares, pos.trailing_stop or pos.stop_loss)
                    if new_id:
                        self._stop_order_ids[ticker] = new_id
                    continue

                fill_price = result.filled_price or target_price
                fill_qty = result.filled_quantity or tranche_shares
                realized = (fill_price - pos.entry_price) * fill_qty
                pos.realized_pnl += realized
                pos.shares_sold += fill_qty
                pos.shares = round(pos.shares - fill_qty, 9)
                pos.take_profit_targets = pos.take_profit_targets[1:]
                is_final = pos.shares <= 1e-9 or not pos.take_profit_targets

                if self.portfolio._db:
                    await self.portfolio._db.execute(
                        "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
                        "VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?)",
                        (ticker, fill_qty, fill_price, realized, datetime.now().isoformat(),
                         "Take-Profit (final tranche)" if is_final else "Take-Profit", pos.trade_id),
                    )
                    await self.portfolio._db.commit()

                if is_final:
                    self.portfolio.positions.pop(ticker, None)
                    await self.portfolio._remove_position_db(ticker)
                    self.portfolio.cash += fill_price * fill_qty
                    self.portfolio.update_peak()
                    # Persist cash immediately (2026-08-20, LINK/USD incident) -- both
                    # branches here credited self.portfolio.cash in memory only, with
                    # no _save_state() call, unlike every other cash-changing path
                    # (add_position_async on buy, close_position_async on a full
                    # manual/stop close). A restart landing between a TP fill and
                    # whichever of those unrelated paths next happened to fire
                    # permanently lost the cash credit while the position's reduced
                    # share count (already persisted via _save_position/
                    # _remove_position_db) survived -- confirmed live: a restart 70s
                    # after a real TP1 fill made the dashboard show total P&L roughly
                    # $350 more negative than reality, even though the real Alpaca
                    # account was correct the whole time.
                    await self.portfolio._save_state()
                    logger.info("Take-profit final tranche closed %s: %.9g @ $%.4f (P&L $%.2f)",
                                ticker, fill_qty, fill_price, realized)
                else:
                    self.portfolio.cash += fill_price * fill_qty
                    await self.portfolio._save_position(pos)
                    await self.portfolio._save_state()
                    logger.info("Take-profit tranche filled %s: %.9g @ $%.4f (P&L $%.2f, %d target(s) left)",
                                ticker, fill_qty, fill_price, realized, len(pos.take_profit_targets))
                    new_id = await self._place_stop_order(ticker, pos.shares, pos.trailing_stop or pos.stop_loss)
                    if new_id:
                        self._stop_order_ids[ticker] = new_id

    async def update_position_prices(self, quotes: dict[str, float]):
        """Refreshes current_price for every open position from a caller-supplied
        {ticker: price} map (the main loop's own per-cycle quote fetch) and detects a
        position Alpaca no longer lists (stop fired since the last check)."""
        try:
            real_positions = {p["ticker"] for p in await self.broker.get_positions()}
        except Exception as e:
            logger.warning("update_position_prices: failed to fetch real positions: %s", e)
            real_positions = None  # unknown -- skip the apparent-close check this cycle

        for ticker, pos in list(self.portfolio.positions.items()):
            if ticker in quotes:
                pos.current_price = quotes[ticker]
            if real_positions is not None and ticker not in real_positions:
                await self._handle_apparent_close(ticker)
        self.portfolio.update_peak()

    async def _handle_apparent_close(self, ticker: str):
        """Alpaca no longer shows this position -- almost always the resting stop
        firing. Verifies via a real closed-order lookup before recording anything,
        rather than guessing a price (same AI-Data-Integrity-adjacent principle
        AITrading's own fabricated-fill audit established: never invent a fill)."""
        pos = self.portfolio.positions.get(ticker)
        if pos is None:
            return
        try:
            closed = await self.broker.get_closed_orders(symbols=[ticker], limit=20)
        except Exception as e:
            logger.warning("_handle_apparent_close: closed-order lookup failed for %s: %s", ticker, e)
            closed = []

        real_sell = next((o for o in closed if o["side"] == "sell" and o["filled_qty"] > 0), None)
        if real_sell and real_sell["filled_avg_price"]:
            fill_price = real_sell["filled_avg_price"]
            fill_qty = real_sell["filled_qty"]
            reason = "Stop Loss"
        else:
            # Couldn't confirm a real fill -- use the live current_price as an honest
            # "roughly where this traded" estimate, never a guessed target price.
            fill_price = pos.current_price
            fill_qty = pos.shares
            reason = "UNRECONCILED FILL (real order not found — needs manual review)"
            logger.warning("_handle_apparent_close: %s closed at Alpaca with no matching "
                            "real order found — recording an unreconciled fill", ticker)

        self._stop_order_ids.pop(ticker, None)
        await self.portfolio.close_position_async(
            ticker, exit_shares=fill_qty, exit_price=fill_price, reason=reason)
        logger.info("Position closed (detected): %s %.9g @ $%.4f (%s)", ticker, fill_qty, fill_price, reason)

    async def execute_sell(self, ticker: str, reason: str = "Manual sell") -> bool:
        """Full manual close of a position, e.g. from a dashboard action."""
        pos = self.portfolio.positions.get(ticker)
        if pos is None:
            return False
        async with self._lock_for(ticker):
            existing_id = self._stop_order_ids.pop(ticker, None)
            if existing_id:
                await self.broker.cancel_order(existing_id)
                await asyncio.sleep(1.0)

            order = Order(ticker=ticker, side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=pos.shares)
            try:
                result = await self.broker.submit_order(order)
            except Exception as e:
                logger.error("execute_sell: failed for %s: %s", ticker, e)
                return False

            fill_price = result.filled_price or pos.current_price
            fill_qty = result.filled_quantity or pos.shares
            await self.portfolio.close_position_async(
                ticker, exit_shares=fill_qty, exit_price=fill_price, reason=reason)
            logger.info("Sold %.9g %s @ $%.4f (%s)", fill_qty, ticker, fill_price, reason)
            return True
