"""Hilton's AI Crypto Trading — FastAPI dashboard + trading engine.

Phase 1 (see docs/superpowers/specs/2026-08-15-phase1-design.md): one
continuous scan loop across a fixed, small crypto universe, one research
tier, direct buy/sell decisions (no On Deck staging), a simple dashboard.
Deliberately much smaller than AITrading's own web/app.py -- most of that
file's size is dashboard features (On Deck/On Shore, Deep Dive modals, the
Update Available badge, dozens of Settings-page fields) explicitly deferred
past Phase 1, not core trading logic.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from src.utils.config import load_config, update_settings_yaml
from src.update.version import read_local_version, write_local_version, is_newer
from src.update.release_client import fetch_latest_release
from src.update.apply import extract_release_archive, copy_updatable_files, requirements_changed
from src.research.event_triggers import check_event_triggers, compute_rsi_from_buffer
from src.data.market_data import MarketDataFetcher, _round_price
from src.data.news_feed import NewsFeed
from src.research.engine import ResearchEngine
from src.decision.portfolio import Portfolio
from src.decision.risk_manager import RiskManager
from src.decision.signal_generator import SignalGenerator, compute_rr, _required_rr
from src.execution.order_manager import OrderManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
_SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "dev-only-insecure-key")


def _default_take_profit_targets(entry_price: float, tp_cfg: dict) -> list[float]:
    """Config-percentage-based T1/T2/T3 ladder (2026-08-20, LINK/USD incident) -- same
    exact formula ResearchEngine's own rule-based fallback already uses
    (src/research/engine.py), just extracted as a shared, independently callable
    function. Used as a safety-net backfill when a REAL (non-fallback) Claude report
    qualifies for a buy but its own take_profit_targets came back empty -- LINK/USD
    was bought this way on 2026-08-15 with zero targets, and since a held position is
    never re-analyzed (see _act_on_report's "already held -- position management
    handles exits"), nothing would ever have caught or corrected that gap on its own.
    Never overrides a report that DID return real targets -- only fills a genuine
    void so every buy leaves this system with an active take-profit ladder."""
    t1_pct = tp_cfg.get("t1_pct", 8.0) / 100
    t2_pct = tp_cfg.get("t2_pct", 16.0) / 100
    t3_pct = tp_cfg.get("t3_pct", 28.0) / 100
    return [
        _round_price(entry_price * (1 + t1_pct)),
        _round_price(entry_price * (1 + t2_pct)),
        _round_price(entry_price * (1 + t3_pct)),
    ]


def _watch_conviction_changed_meaningfully(
    old_conviction: float | None, new_conviction: float, epsilon: float,
) -> bool:
    """Whether a fresh watchlist conviction score differs enough from the last recorded
    one to reset watching_loop's stale-backoff streak (2026-08-19, cost audit follow-up
    -- see _watch_recheck_due's docstring for the full incident). Unlike a dip low or
    an R/R ratio, conviction has no natural "only matters getting deeper" direction --
    a real rise toward the buy gate and a real fall out of contention are equally worth
    noticing -- so this is a plain symmetric |delta| >= epsilon check, not a one-sided
    comparison. A missing old_conviction (first time this ticker was ever checked)
    always counts as meaningfully different."""
    if old_conviction is None:
        return True
    return abs(new_conviction - old_conviction) >= epsilon


def _watch_recheck_due(
    streak: int, base_interval_min: float, backoff_multiplier: float,
    max_interval_min: float, minutes_since_last: float,
) -> bool:
    """Whether enough time has passed to spend another real watchlist re-check on this
    ticker (2026-08-19, owner-requested cost audit follow-up) -- live-caught: BTC/USD
    sat on the watchlist getting re-analyzed every 45 minutes for hours, landing on the
    identical conviction score (5.2/10) four checks in a row. Conviction is entirely
    Claude-derived, unlike AITrading's dip-low/R/R checks, which have a free, non-Claude
    price computation available to decide whether a call is even worth making -- there's
    no equivalent cheap proxy here, so instead of skipping outright, each consecutive
    unchanged result (tracked as `streak`, reset to 0 by
    _watch_conviction_changed_meaningfully) doubles the required gap before the next
    real check, same principle as this codebase's own exit-order retry backoff.
    Clamped to max_interval_min -- by design, the owner-set default matches the full
    scan's own interval, since beyond that point the scheduled scan re-touches this
    ticker anyway and further backoff would just be redundant with it."""
    interval = min(base_interval_min * (backoff_multiplier ** streak), max_interval_min)
    return minutes_since_last >= interval


def _apply_live_quotes_to_reports(reports: dict, quotes: dict) -> None:
    """Mutates each report's current_price in place from a ticker->price map (2026-08-19,
    owner request) -- gives the universe/candidate cards a live price alongside their
    AI-recommended entry_price, which is otherwise a static snapshot from whenever that
    report was generated (potentially hours stale), unlike the Positions cards, which
    already refresh current_price every ~20s. Reuses the quotes _run_event_scan already
    fetches every cycle (concurrently, via asyncio.gather) for its own trigger-checking,
    rather than adding a second dedicated fetch loop -- no new network calls, no risk to
    position_update_cycle's own timing. Leaves a report's current_price untouched (never
    reset to None) for any ticker whose quote wasn't fetched this cycle -- a transient
    failure, or a ticker genuinely outside the universe -- so a value already shown on
    the dashboard doesn't flicker away."""
    for ticker, price in quotes.items():
        if ticker in reports:
            reports[ticker]["current_price"] = price


class DashboardState:
    def __init__(self):
        self.config = load_config()
        self.portfolio = Portfolio(self.config)
        self.risk_manager = RiskManager(self.config)
        self.market_data = MarketDataFetcher(self.config)
        self.news_feed = NewsFeed(self.config)
        self.research_engine = ResearchEngine(self.config, self.market_data, self.news_feed)
        self.signal_generator = SignalGenerator(self.config, self.research_engine, self.risk_manager, self.portfolio)
        self.order_manager = OrderManager(self.config, self.portfolio)

        self.universe: list[dict] = self.config.get("universe", [])
        self.websockets: list[WebSocket] = []

        # latest_reports persistence -- same restart-survival pattern as ai_log.
        # Each serialized report is a plain JSON-safe dict so a flat JSON file works.
        self._reports_cache_path = "data/latest_reports_cache.json"
        self.latest_reports: dict[str, dict] = self._load_reports_cache()
        self._market_tz = ZoneInfo(self.config.get("research", {}).get("market_timezone", "America/New_York"))
        self._scan_in_progress = False
        self._apply_update_in_progress: bool = False
        self._update_status_cache: dict | None = None
        self._update_status_cache_time: datetime | None = None

        # Pause/Stop (2026-08-20, owner request) -- two independent severity levels,
        # both persisted to disk together (crypto trades 24/7 with no natural
        # market-closed quiet window the way AITrading has, so an owner who pauses or
        # stops specifically to halt activity would otherwise have that silently
        # undone by the next restart/deploy):
        #   paused=True  -- stops every AI-spend loop (scan_loop, watching_loop,
        #                    event_scan_loop, asset_profile_refresh_loop). Zero Claude
        #                    calls. position_loop/heartbeat_loop keep running --
        #                    held positions stay fully protected (stop-loss/
        #                    trailing-stop/take-profit).
        #   stopped=True -- everything above PLUS position_loop itself stops. No
        #                    broker-side management of any kind (own request: "no
        #                    management of any positions at the broker will take
        #                    place"). Deliberately does NOT kill the process itself
        #                    (owner chose this over a real systemd stop, 2026-08-20 --
        #                    a genuine process kill has no self-service recovery path
        #                    for a non-technical customer; this way the dashboard
        #                    stays reachable and a Start System click brings
        #                    everything back with no SSH/support needed).
        # heartbeat_loop is never gated by either -- it's a pure WS liveness ping with
        # no cost, and the dashboard must stay reachable in both states.
        self._run_state_path = "data/run_state.json"
        self.paused: bool
        self.stopped: bool
        self.paused, self.stopped = self._load_run_state()

        # Event-triggered scanning state -- in-memory only (losing a 60-min cooldown
        # on a restart is a minor cost vs. the complexity of persisting these).
        self._event_price_buffers: dict[str, list[float]] = {}   # rolling prices for RSI
        self._event_last_prices: dict[str, float] = {}           # price from previous tick
        self._event_claude_cooldown: dict[str, datetime] = {}    # per-ticker cooldown

        # Per-ticker lock around _act_on_report's qualify/confirm/buy decision
        # (2026-08-18, cost audit) -- four independent producers (the batch scan, the
        # sequential fallback scan, watching_loop, and the event-scan trigger) can each
        # call _act_on_report for the same ticker with no coordination between them.
        # Without this, two of those loops processing the same ticker around the same
        # time could each pass the "not already held" check, each fire a real, billed
        # buy-confirmation Claude call for the same ticker, and -- worse than the wasted
        # spend alone -- each proceed to execute_buy, risking a double buy. The lock
        # makes the qualify/confirm/buy sequence mutually exclusive per ticker; a
        # concurrent call blocks until the first finishes, then re-checks
        # self.portfolio.positions (now updated if the first call bought) before doing
        # anything itself. In-memory only, same not-worth-persisting precedent as the
        # cooldown dicts above -- a lock has no meaning to persist across a restart.
        self._act_on_report_locks: dict[str, asyncio.Lock] = {}

        # Near-miss watchlist (2026-08-17): assets above watch_floor_conviction but
        # below the buy gate -- re-scanned every watch_interval_minutes by watching_loop
        # so a strengthening setup is caught quickly. In-memory only; repopulated by
        # the next full scan after a restart.
        self.watching_candidates: dict[str, dict] = {}

        # Watchlist staleness backoff (2026-08-19, owner-requested cost audit
        # follow-up) -- live-caught: BTC/USD sat on the watchlist getting re-analyzed
        # every 45 minutes for hours, landing on the identical conviction (5.2/10) four
        # checks in a row. Conviction is Claude-derived with no cheap non-Claude proxy
        # (unlike AITrading's dip-low/R/R checks), so watching_loop can't decide to skip
        # a ticker without first paying for the check -- instead, each consecutive
        # unchanged result stretches the NEXT required gap further (see
        # _watch_recheck_due). In-memory only, same not-worth-persisting precedent as
        # every other cooldown/lock dict in this file -- losing a backoff streak on
        # restart just means one ticker's next check reverts to the base interval.
        self._watch_last_check: dict[str, datetime] = {}
        self._watch_stale_streak: dict[str, int] = {}
        self._watch_last_conviction: dict[str, float] = {}

        # AI log persistence (2026-08-16, owner report: "the ai research engine is not
        # keeping the history") -- self.ai_log was purely in-memory, so it reset to
        # empty on every restart. This project got restarted several times in one
        # session (each real deploy), and every restart silently wiped the whole log
        # with no trace it had ever happened -- same real bug class AITrading already
        # fixed for its own ai_log (mirrored here near-verbatim: a real sqlite table,
        # loaded on startup, one fire-and-forget persist per entry so a busy scan
        # cycle's ~hundreds of add_log calls never block the event loop on a
        # synchronous sqlite3 connect+insert).
        self._log_db_path = self.config.get("database", {}).get("path", "data/aicryptotrading.db")
        self._init_log_db()
        self.ai_log: list[dict] = self._load_log_from_db()

    def _now_et(self) -> datetime:
        return datetime.now(self._market_tz)

    def _load_run_state(self) -> tuple[bool, bool]:
        try:
            data = json.loads(Path(self._run_state_path).read_text(encoding="utf-8"))
            return bool(data.get("paused", False)), bool(data.get("stopped", False))
        except Exception:
            return False, False

    def _save_run_state(self):
        try:
            Path(self._run_state_path).write_text(
                json.dumps({"paused": self.paused, "stopped": self.stopped}), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save run state: %s", e)

    def run_status(self) -> str:
        """Single source of truth for the 3-way UI state -- 'stopped' takes priority
        over 'paused' since it's the stronger condition (see the __init__ comment for
        exactly what each level gates)."""
        if self.stopped:
            return "stopped"
        if self.paused:
            return "paused"
        return "running"

    def _load_reports_cache(self) -> dict:
        try:
            return json.loads(Path(self._reports_cache_path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_reports_cache(self):
        try:
            Path(self._reports_cache_path).write_text(
                json.dumps(self.latest_reports, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save reports cache: %s", e)

    def _init_log_db(self):
        import sqlite3
        Path(self._log_db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._log_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT,
                    phase TEXT,
                    content TEXT,
                    level TEXT,
                    created_at TEXT
                )
            """)
            conn.commit()

    def _load_log_from_db(self) -> list[dict]:
        import sqlite3
        with sqlite3.connect(self._log_db_path) as conn:
            rows = conn.execute(
                "SELECT time, phase, content, level FROM ai_log ORDER BY id DESC LIMIT 300"
            ).fetchall()
        return [{"time": r[0], "phase": r[1], "content": r[2], "level": r[3]} for r in reversed(rows)]

    def _persist_log_entry(self, entry: dict):
        import sqlite3
        try:
            with sqlite3.connect(self._log_db_path) as conn:
                conn.execute(
                    "INSERT INTO ai_log (time, phase, content, level, created_at) VALUES (?, ?, ?, ?, ?)",
                    (entry["time"], entry["phase"], entry["content"], entry["level"], datetime.now().isoformat()),
                )
                # Keep the table from growing forever -- prune to the most recent 1000 rows.
                conn.execute(
                    "DELETE FROM ai_log WHERE id NOT IN (SELECT id FROM ai_log ORDER BY id DESC LIMIT 1000)"
                )
                conn.commit()
        except Exception as e:
            logger.warning("ai_log persist error: %s", e)

    def add_log(self, phase: str, content: str, level: str = "INFO"):
        entry = {
            "phase": phase,
            "content": content,
            "level": level,
            "time": self._now_et().strftime("%b %d %H:%M:%S"),
        }
        self.ai_log.append(entry)
        self.ai_log = self.ai_log[-500:]
        asyncio.create_task(asyncio.to_thread(self._persist_log_entry, entry))
        asyncio.create_task(self.broadcast({"type": "ai_log", "entry": entry}))

    async def heartbeat_loop(self):
        """A real scan cycle only broadcasts while it's actively running, and
        position_update_cycle broadcasts nothing at all with zero open positions --
        so a client can go many minutes with no message at all during a genuinely
        healthy quiet period (confirmed live 2026-08-15: a 15-minute scan gap with no
        held positions produced zero broadcasts the whole time). Without this, a
        client can't tell "quiet because nothing happened" apart from "silently dead
        connection" -- this periodic heartbeat gives it a fixed, predictable signal
        to check liveness against instead of guessing from data traffic."""
        while True:
            await asyncio.sleep(20)
            await self.broadcast({"type": "heartbeat"})

    async def broadcast(self, message: dict):
        dead = []
        text = json.dumps(message, default=str)
        for ws in self.websockets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.websockets:
                self.websockets.remove(ws)

    def get_portfolio_snapshot(self) -> dict:
        p = self.portfolio
        return {
            "cash": p.cash,
            "total_value": p.total_value,
            "total_pnl": p.total_pnl,
            "total_pnl_pct": p.total_pnl_pct,
            "day_pnl": p.day_pnl,
            "day_pnl_pct": round(p.day_pnl / p.day_start_value * 100, 4) if p.day_start_value else 0.0,
            "peak_value": p.peak_value,
            "positions": [
                {
                    "ticker": pos.ticker,
                    "shares": pos.shares,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "lifetime_pnl": pos.lifetime_pnl,
                    "stop_loss": pos.stop_loss,
                    "trailing_stop": pos.trailing_stop,
                    "final_trail_pct": pos.final_trail_pct,
                    "take_profit_targets": pos.take_profit_targets,
                    "t1_target_price": pos.t1_target_price,
                    "t2_target_price": pos.t2_target_price,
                    "opened_at": pos.opened_at.isoformat(),
                }
                for pos in p.positions.values()
            ],
        }

    async def get_init_payload(self) -> dict:
        return {
            "type": "init",
            "portfolio": self.get_portfolio_snapshot(),
            "universe": self.universe,
            "reports": self.latest_reports,
            "ai_log": self.ai_log[-100:],
            "paper_trading": self.config["trading"]["paper_trading"],
            "recent_sells": await self.portfolio.get_recent_sells(),
            "watching": list(self.watching_candidates.keys()),
            "run_status": self.run_status(),
        }

    async def run_scan_cycle(self):
        """One full pass: analyze every asset in the universe, and act on any
        qualifying signal immediately (direct buy/sell, no On Deck staging -- see
        design spec). Tries the Anthropic Batch API first (2026-08-16, real
        cost-reduction request -- 50% off list price, same mechanism AITrading's own
        nightly universe scan already uses), falling back to the original sequential
        per-ticker path if the batch never gets a batch_id or stalls."""
        if self._scan_in_progress:
            self.add_log("SCAN", "Skipped — previous scan cycle still running", "WARNING")
            return
        self._scan_in_progress = True
        self.add_log("SCAN", f"Starting scan cycle — {len(self.universe)} assets")
        try:
            if not await self._run_batch_scan():
                self.add_log("SCAN", "Batch scan unavailable/stalled — falling back to sequential", "WARNING")
                await self._run_sequential_scan()

            await self.broadcast({"type": "reports", "reports": self.latest_reports, "watching": list(self.watching_candidates.keys())})
            self.add_log("SCAN", "Scan cycle complete")
        finally:
            self._scan_in_progress = False

    def _update_watchlist(self, ticker: str, report) -> None:
        """Manages the near-miss watchlist: assets above watch_floor_conviction but
        below the buy gate get added here and re-scanned more frequently by
        watching_loop, so a strengthening setup is caught within minutes rather than
        waiting for the next 3-hour full cycle."""
        research_cfg = self.config.get("research", {})
        watch_floor = research_cfg.get("watch_floor_conviction", 4.5)
        min_conviction = research_cfg.get("min_conviction_score", 6.5)

        if report.conviction_score >= watch_floor:
            was_watching = ticker in self.watching_candidates
            self.watching_candidates[ticker] = self.latest_reports[ticker]
            if not was_watching and report.conviction_score < min_conviction:
                self.add_log(
                    ticker,
                    f"Added to watchlist — conviction {report.conviction_score}/10 "
                    f"(gate is {min_conviction}), monitoring for setup to strengthen",
                )
        elif ticker in self.watching_candidates:
            del self.watching_candidates[ticker]
            # Clear staleness-backoff state too (2026-08-19) -- a ticker that later
            # re-qualifies for the watchlist should start fresh at the base interval,
            # not inherit a long-stretched backoff from a previous, unrelated stint.
            self._watch_last_check.pop(ticker, None)
            self._watch_stale_streak.pop(ticker, None)
            self._watch_last_conviction.pop(ticker, None)
            self.add_log(
                ticker,
                f"Removed from watchlist — conviction fell to {report.conviction_score}/10",
                "WARNING",
            )

    async def _act_on_report(self, ticker: str, name: str, report):
        """Shared post-analysis logic for both the batch and sequential scan paths --
        log, check for a real position action, execute a buy. One source of truth so
        the two submission mechanisms can't drift in what counts as a qualifying
        signal."""
        self.latest_reports[ticker] = self._serialize_report(report)
        asyncio.create_task(asyncio.to_thread(self._save_reports_cache))

        if report.is_fallback:
            self.add_log(ticker, f"Analysis failed — {report.thesis}", "ERROR")
            return

        self.add_log(
            ticker,
            f"{report.signal.value} — conviction {report.conviction_score}/10, "
            f"entry ${report.entry_price:.2f}",
        )

        if ticker in self.portfolio.positions:
            return  # already held -- position management handles exits

        # Maintain near-miss watchlist so watching_loop can re-scan close candidates
        self._update_watchlist(ticker, report)

        signal = self.signal_generator._evaluate_report(report)
        if signal is None or not signal.should_execute:
            return

        # Per-ticker lock (2026-08-18, cost audit) -- see its own comment in __init__.
        # Everything from here on (the real qualify/confirm/buy decision) is mutually
        # exclusive per ticker, closing the gap where two independent scan loops could
        # otherwise both fire a real confirmation call, or both buy, for the same
        # ticker around the same time.
        lock = self._act_on_report_locks.setdefault(ticker, asyncio.Lock())
        async with lock:
            # Re-check under the lock -- a concurrent call for this same ticker may
            # have already bought it while this call was waiting to acquire the lock.
            if ticker in self.portfolio.positions:
                return

            if len(self.portfolio.positions) >= self.config.get("portfolio", {}).get("max_positions", 8):
                self.add_log(ticker, "Signal qualifies but max_positions reached — skipped", "WARNING")
                return

            # Confirmation pass: run a second independent Claude call before buying.
            # Both must agree (BUY/STRONG BUY, conviction >= gate, R/R clears) to execute.
            research_cfg = self.config.get("research", {})
            if research_cfg.get("buy_confirm_required", True):
                self.add_log(ticker, "BUY qualified — running confirmation analysis before executing")
                try:
                    trade_history = await self.portfolio.get_trade_history_summary(ticker)
                    confirm_report = await self.research_engine.analyze_asset(ticker, name, trade_history)
                    confirm_signal = self.signal_generator._evaluate_report(confirm_report)
                    if confirm_signal is None or not confirm_signal.should_execute:
                        self.add_log(
                            ticker,
                            f"BUY confirmation disagreed — conviction {confirm_report.conviction_score}/10 "
                            f"({confirm_report.signal.value}) — skipped",
                            "WARNING",
                        )
                        return
                    signal = confirm_signal
                    self.latest_reports[ticker] = self._serialize_report(confirm_report)
                except Exception as e:
                    self.add_log(ticker, f"BUY confirmation error — skipped: {e}", "ERROR")
                    return

            # Backfill missing take-profit targets before executing (2026-08-20,
            # LINK/USD incident) -- Claude's own report occasionally comes back with
            # an empty take_profit_targets list with no error/fallback flag to catch
            # it, and since a held position is never re-analyzed once bought (see
            # this function's own early "already held" return above), a gap here
            # would otherwise never self-correct. Only ever fills a genuine void --
            # a report that DID return real targets is never overridden.
            tp_targets = signal.take_profit_targets
            if not tp_targets:
                tp_targets = _default_take_profit_targets(
                    signal.entry_price, self.config.get("take_profit", {}))
                self.add_log(ticker,
                    f"AI report had no take-profit targets — using config-based "
                    f"fallback ladder: {[f'${t:.4f}' for t in tp_targets]}", "WARNING")

            ok = await self.order_manager.execute_buy(
                ticker, name, signal.entry_price, signal.stop_loss,
                tp_targets, signal.position_size_dollars,
                signal.final_trail_pct,
            )
            if ok:
                self.watching_candidates.pop(ticker, None)
                self.add_log(
                    ticker,
                    f"BUY executed — ${signal.position_size_dollars:.2f} "
                    f"(conviction {signal.conviction}/10)",
                )
                await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})
            else:
                self.add_log(ticker, "BUY signal qualified but order execution failed", "ERROR")

    async def _run_sequential_scan(self):
        """Original per-ticker path -- safety net for when the batch API is
        unavailable (no client) or stalls. Also the only path exercised when
        ANTHROPIC_API_KEY is unset, since analyze_asset itself degrades to the
        rule-based fallback in that case."""
        for asset in self.universe:
            ticker, name = asset["ticker"], asset["name"]
            try:
                trade_history = await self.portfolio.get_trade_history_summary(ticker)
                report = await self.research_engine.analyze_asset(ticker, name, trade_history)
                await self._act_on_report(ticker, name, report)
            except Exception as e:
                logger.exception("Scan cycle failed for %s", ticker)
                self.add_log(ticker, f"Scan error: {e}", "ERROR")
            await asyncio.sleep(1)  # small pacing gap between assets

    async def _run_batch_scan(self) -> bool:
        """Submits the whole universe as one Anthropic Batch API call and waits for
        it to complete, acting on each result via _act_on_report once they're ready.
        Returns True if the batch path handled the scan (an individual ticker's
        data-gathering failure just gets it skipped, same as the sequential path's
        per-ticker try/except -- that alone doesn't count as a reason to fall back),
        False if the caller should fall back to _run_sequential_scan entirely (no
        batch could be submitted, or it stalled and was cancelled). A 22-asset batch
        is tiny next to AITrading's own 1,500+-ticker batches (confirmed there to
        never take longer than ~7 minutes even at 977 requests), so this uses a much
        shorter timeout than that system's adaptive multi-chunk orchestrator -- no
        chunking is needed at this universe size."""
        batch_id, inputs_by_ticker = await self.research_engine.submit_analysis_batch(self.universe)
        if not batch_id:
            return False

        POLL_INTERVAL = 10.0
        STUCK_TIMEOUT = 180.0   # 3 min of genuinely zero progress -> assume stuck
        HARD_CAP = 600.0        # 10 min hard cap even if slowly progressing
        start = time.monotonic()
        last_progress_ts = start
        last_done_count = 0

        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                status = await self.research_engine.poll_batch_status(batch_id)
            except Exception as e:
                logger.warning("Batch poll error for %s: %s", batch_id, e)
                continue

            if status.processing_status == "ended":
                break

            counts = status.request_counts
            done_count = counts.succeeded + counts.errored + counts.canceled + counts.expired
            now = time.monotonic()
            if done_count > last_done_count:
                last_done_count = done_count
                last_progress_ts = now

            if now - last_progress_ts > STUCK_TIMEOUT:
                logger.error(
                    "Batch %s stuck (no progress for %.0fs, %s) — cancelling, "
                    "falling back to sequential", batch_id, STUCK_TIMEOUT, counts)
                await self.research_engine.cancel_batch(batch_id)
                return False
            if now - start > HARD_CAP:
                logger.error(
                    "Batch %s exceeded hard cap (%.0fs, %s) — cancelling, "
                    "falling back to sequential", batch_id, HARD_CAP, counts)
                await self.research_engine.cancel_batch(batch_id)
                return False

        reports = await self.research_engine.fetch_batch_results(batch_id, inputs_by_ticker)
        for ticker, report in reports.items():
            name = inputs_by_ticker[ticker]["asset_name"]
            try:
                await self._act_on_report(ticker, name, report)
            except Exception as e:
                logger.exception("Post-batch action failed for %s", ticker)
                self.add_log(ticker, f"Scan error: {e}", "ERROR")

        skipped = {a["ticker"] for a in self.universe} - set(inputs_by_ticker.keys())
        if skipped:
            self.add_log(
                "SCAN",
                f"{len(skipped)} asset(s) skipped this cycle (data gathering failed): "
                f"{', '.join(sorted(skipped))}", "WARNING")

        return True

    def _serialize_report(self, report) -> dict:
        research_cfg = self.config.get("research", {})
        min_conviction = research_cfg.get("min_conviction_score", 7)
        rr = compute_rr(report.entry_price, report.stop_loss, report.fair_value_estimate)
        required_rr = _required_rr(
            report.conviction_score, min_conviction,
            research_cfg.get("min_risk_reward_ratio", 2.0),
            research_cfg.get("rr_conviction_step", 0.1),
            research_cfg.get("rr_floor", 1.5),
        )
        return {
            "ticker": report.ticker,
            "asset_name": report.asset_name,
            "generated_at": report.generated_at.isoformat(),
            "conviction_score": report.conviction_score,
            "signal": report.signal.value,
            "risk_level": report.risk_level.value,
            "thesis": report.thesis,
            "asset_summary": report.asset_summary,
            "technical_summary": report.technical_summary,
            "news_summary": report.news_summary,
            "risk_factors": report.risk_factors,
            "entry_price": report.entry_price,
            "stop_loss": report.stop_loss,
            "take_profit_targets": report.take_profit_targets,
            "reasoning": report.reasoning,
            "recommended_action": report.recommended_action,
            "time_horizon": report.time_horizon,
            "fair_value_estimate": report.fair_value_estimate,
            "margin_of_safety_pct": report.margin_of_safety_pct,
            "is_fallback": report.is_fallback,
            "rr": rr,
            "required_rr": required_rr,
            "current_price": None,  # filled in by _apply_live_quotes_to_reports
        }

    async def position_update_cycle(self):
        """Runs frequently (every ~20s): refresh quotes for held positions, sync exit
        orders (graduated trailing stop), and check take-profit crossings."""
        if not self.portfolio.positions:
            return
        quotes = {}
        for ticker in list(self.portfolio.positions.keys()):
            try:
                q = await self.market_data.get_quote(ticker)
                quotes[ticker] = q.price
            except Exception as e:
                logger.warning("position_update_cycle: quote fetch failed for %s: %s", ticker, e)

        await self.order_manager.update_position_prices(quotes)
        await self.order_manager.sync_exit_orders()
        await self.order_manager.check_take_profits()
        await self.broadcast({"type": "portfolio", "portfolio": self.get_portfolio_snapshot()})

    async def scan_loop(self):
        interval = self.config.get("scan", {}).get("interval_minutes", 15) * 60
        while True:
            try:
                if not self.paused and not self.stopped:
                    await self.run_scan_cycle()
            except Exception:
                logger.exception("scan_loop: unhandled error in scan cycle")
            await asyncio.sleep(interval)

    async def position_loop(self):
        while True:
            try:
                # Gated on stopped only, NOT paused -- position management (quote
                # refresh, exit-order sync, take-profit checks) has zero AI cost and
                # must keep running through an ordinary pause. Only a full Stop halts
                # it too (see the __init__ comment for the full paused/stopped split).
                if not self.stopped:
                    await self.position_update_cycle()
            except Exception:
                logger.exception("position_loop: unhandled error in position update cycle")
            await asyncio.sleep(20)

    async def asset_profile_refresh_loop(self):
        """Checked once a day; scripts.build_asset_profiles.generate_all_profiles()
        itself skips any ticker whose profile is still younger than
        research.asset_profile_refresh_days, so most days this is a fast no-op check
        rather than a real Claude spend -- see that module for why the refresh
        cadence lives there (single source of truth) rather than being duplicated
        here. Runs once shortly after startup too (not just once every 24h from a
        cold clock), so a box that restarts daily still eventually refreshes."""
        from scripts.build_asset_profiles import generate_all_profiles
        while True:
            try:
                if not self.paused and not self.stopped:
                    profiles = await generate_all_profiles(self.config)
                    self.research_engine.asset_profiles = profiles
                    self.add_log("SYSTEM", f"Asset profile refresh check complete ({len(profiles)} profiles cached)")
            except Exception:
                logger.exception("asset_profile_refresh_loop: unhandled error")
            await asyncio.sleep(24 * 3600)

    async def watching_loop(self):
        """Re-scans near-miss candidates (above watch_floor_conviction, below the buy
        gate) at a shorter cadence than the full universe scan. No Claude spend when
        nothing is watching; skips tickers that get bought or drop off the list between
        intervals. Waits one full interval before first firing so the initial full scan
        populates the watchlist first."""
        research_cfg = self.config.get("research", {})
        interval = research_cfg.get("watch_interval_minutes", 45) * 60
        await asyncio.sleep(interval)
        while True:
            try:
                if self.watching_candidates and not self._scan_in_progress and not self.paused and not self.stopped:
                    tickers = list(self.watching_candidates)
                    stale_cfg = self.config.get("research", {})
                    backoff_mult = stale_cfg.get("watch_stale_backoff_multiplier", 2.0)
                    backoff_max = stale_cfg.get("watch_stale_backoff_max_minutes", 180)
                    conviction_epsilon = stale_cfg.get("watch_stale_conviction_epsilon", 0.5)
                    base_interval_min = interval / 60
                    now = datetime.now()

                    # Staleness backoff (2026-08-19, cost audit follow-up) -- filters
                    # to only the tickers actually due this cycle BEFORE logging/
                    # spending anything, so a ticker whose conviction has repeatedly
                    # come back unchanged (real live pattern: BTC/USD stuck at 5.2/10
                    # for 4 consecutive checks) gets checked less often over time
                    # instead of every single interval forever. See _watch_recheck_due.
                    due_tickers = [
                        t for t in tickers
                        if _watch_recheck_due(
                            streak=self._watch_stale_streak.get(t, 0),
                            base_interval_min=base_interval_min,
                            backoff_multiplier=backoff_mult,
                            max_interval_min=backoff_max,
                            minutes_since_last=(
                                (now - self._watch_last_check[t]).total_seconds() / 60
                                if t in self._watch_last_check else base_interval_min
                            ),
                        )
                    ]
                    if due_tickers:
                        self.add_log("WATCH", f"Re-scanning {len(due_tickers)} near-miss candidate(s)")
                    for ticker in due_tickers:
                        if ticker not in self.watching_candidates:
                            continue  # evicted by a concurrent full scan
                        if ticker in self.portfolio.positions:
                            self.watching_candidates.pop(ticker, None)
                            continue
                        asset = next((a for a in self.universe if a["ticker"] == ticker), None)
                        if asset is None:
                            self.watching_candidates.pop(ticker, None)
                            continue
                        try:
                            trade_history = await self.portfolio.get_trade_history_summary(ticker)
                            report = await self.research_engine.analyze_asset(
                                ticker, asset["name"], trade_history
                            )
                            self._watch_last_check[ticker] = now
                            if _watch_conviction_changed_meaningfully(
                                    self._watch_last_conviction.get(ticker),
                                    report.conviction_score, conviction_epsilon):
                                self._watch_stale_streak[ticker] = 0
                            else:
                                self._watch_stale_streak[ticker] = self._watch_stale_streak.get(ticker, 0) + 1
                            self._watch_last_conviction[ticker] = report.conviction_score
                            await self._act_on_report(ticker, asset["name"], report)
                        except Exception:
                            logger.exception("watching_loop: analysis failed for %s", ticker)
                        await asyncio.sleep(1)
                    if due_tickers:
                        await self.broadcast({"type": "reports", "reports": self.latest_reports, "watching": list(self.watching_candidates.keys())})
            except Exception:
                logger.exception("watching_loop: unhandled error")
            await asyncio.sleep(interval)

    async def event_scan_loop(self):
        """Fast cheap loop: parallel quote fetches for all universe assets every few
        minutes, real Claude call only when a genuine trigger fires AND the per-ticker
        cooldown has expired. Fires after an initial delay equal to one interval so
        the startup Batch scan runs first and the price buffers can begin filling.
        Skipped entirely (returns without looping) when event_scan.enabled is False,
        so toggling the setting live requires a restart to take effect -- acceptable
        given this is a background loop."""
        cfg = self.config.get("event_scan", {})
        if not cfg.get("enabled", True):
            self.add_log("SYSTEM", "Event scan disabled in config — event_scan_loop idle")
            return
        interval = cfg.get("interval_minutes", 3) * 60
        await asyncio.sleep(interval)
        while True:
            try:
                if not self.paused and not self.stopped:
                    await self._run_event_scan()
            except Exception:
                logger.exception("event_scan_loop: unhandled error")
            await asyncio.sleep(interval)

    async def _fetch_universe_quotes(self):
        """Concurrent quote fetch for the whole universe -- shared by _run_event_scan's
        own per-cycle pass and _initial_universe_price_fill's one-off startup fill, so
        there's exactly one fetch implementation. Returns (tickers, raw) where raw is
        positionally aligned with tickers and each entry is either a Quote or an
        Exception (asyncio.gather(return_exceptions=True))."""
        tickers = [(a["ticker"], a["name"]) for a in self.universe]
        raw = await asyncio.gather(
            *[self.market_data.get_quote(t) for t, _ in tickers],
            return_exceptions=True,
        )
        return tickers, raw

    async def _initial_universe_price_fill(self):
        """Immediate one-off quote fetch right after startup (2026-08-19, owner report
        right after the 'Current' price feature shipped) -- event_scan_loop deliberately
        waits a full event_scan.interval_minutes before its own first tick (so the
        startup batch scan gets priority), which left every universe card showing the
        null-price '—' fallback for minutes after any restart. No trigger-checking, no
        Claude spend -- just the same shared fetch _run_event_scan uses, applied and
        broadcast once. Any failure here is non-fatal (the regular event_scan_loop tick
        still fills this in a few minutes later either way)."""
        try:
            tickers, raw = await self._fetch_universe_quotes()
            quotes = {t: r.price for (t, _), r in zip(tickers, raw) if not isinstance(r, Exception)}
            _apply_live_quotes_to_reports(self.latest_reports, quotes)
            await self.broadcast({"type": "reports", "reports": self.latest_reports, "watching": list(self.watching_candidates.keys())})
        except Exception:
            logger.exception("_initial_universe_price_fill: failed")

    async def _run_event_scan(self):
        """One event-scan tick: fetch quotes in parallel, update rolling price buffers,
        check triggers for unowned assets, and fire a single-ticker Claude call if a
        trigger fires and the cooldown allows."""
        cfg = self.config.get("event_scan", {})
        cooldown_mins = cfg.get("claude_cooldown_minutes", 60)
        now = datetime.now()

        # Parallel quote fetch -- the only I/O in the normal (no-trigger) fast path
        tickers, raw = await self._fetch_universe_quotes()

        # Collected regardless of the per-ticker skip branches below (held position,
        # cooldown, no trigger) -- feeds the universe cards' live "Current" price via
        # _apply_live_quotes_to_reports, reusing this fetch rather than a second one.
        live_quotes: dict[str, float] = {}

        for (ticker, name), result in zip(tickers, raw):
            if isinstance(result, Exception):
                continue

            price = result.price
            live_quotes[ticker] = price

            # Maintain rolling buffer (max 30 prices) for RSI -- pure in-process math
            buf = self._event_price_buffers.setdefault(ticker, [])
            buf.append(price)
            if len(buf) > 30:
                buf.pop(0)

            prev_price = self._event_last_prices.get(ticker)
            self._event_last_prices[ticker] = price

            # Held positions: position_loop handles all exit/monitoring logic
            if ticker in self.portfolio.positions:
                continue

            # Skip if in per-ticker cooldown
            cooldown_until = self._event_claude_cooldown.get(ticker)
            if cooldown_until and now < cooldown_until:
                continue

            # Don't pile event calls on top of an already-running full scan
            if self._scan_in_progress:
                continue

            rsi = compute_rsi_from_buffer(buf)
            triggers = check_event_triggers(ticker, price, prev_price, rsi, cfg)
            if not triggers:
                continue

            # Gate the real Claude call on minimum conviction floor. Assets with a
            # prior score below this are structurally weak -- their RSI oversold events
            # are typically the asset declining, not a genuine setup. Assets with no
            # prior report (conviction=0) always get analyzed so new assets aren't skipped.
            event_min = cfg.get("event_min_conviction", 4.0)
            last_conviction = self.latest_reports.get(ticker, {}).get("conviction_score", 0)
            if last_conviction > 0 and last_conviction < event_min:
                continue

            trigger_desc = ", ".join(triggers)
            self.add_log(
                ticker,
                f"EVENT TRIGGER: {trigger_desc} (price ${price:.4f}) — running full analysis",
                "WARNING",
            )

            try:
                trade_history = await self.portfolio.get_trade_history_summary(ticker)
                report = await self.research_engine.analyze_asset(ticker, name, trade_history)
                await self._act_on_report(ticker, name, report)
                # Broadcast the updated report so the dashboard card refreshes immediately
                await self.broadcast({"type": "reports", "reports": self.latest_reports, "watching": list(self.watching_candidates.keys())})
            except Exception as e:
                logger.exception("Event scan analysis failed for %s", ticker)
                self.add_log(ticker, f"Event analysis error: {e}", "ERROR")

            # Cooldown applies regardless of analysis outcome -- prevents re-firing
            # on an unchanged condition between full scan cycles
            self._event_claude_cooldown[ticker] = now + timedelta(minutes=cooldown_mins)

        # Refresh the universe cards' live "Current" price every tick (2026-08-19,
        # owner request) -- no extra Claude spend, no extra network calls, just applying
        # the quotes already fetched above. Broadcasts even when no event trigger fired
        # this cycle (the common case), since that's the only path that ever updates it.
        _apply_live_quotes_to_reports(self.latest_reports, live_quotes)
        await self.broadcast({"type": "reports", "reports": self.latest_reports, "watching": list(self.watching_candidates.keys())})


state = DashboardState()
_VERSION_FILE_PATH = str(Path(__file__).resolve().parent.parent / "VERSION")
_INSTALL_ROOT = str(Path(__file__).resolve().parent.parent)

app = FastAPI(title="Hilton's AI Crypto Trading")

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


_LOGIN_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hilton's AI Crypto Trading — Login</title>
<style>
  body { background:#0a0e17; color:#e5e7eb; font-family:system-ui,-apple-system,sans-serif;
         display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
  form { background:#1a2332; border:1px solid #2a3a4e; border-radius:10px; padding:32px;
         width:280px; box-shadow:0 8px 24px rgba(0,0,0,0.4); }
  h1 { font-size:17px; margin:0 0 20px; text-align:center; }
  input { width:100%; box-sizing:border-box; padding:10px 12px; border-radius:6px;
          border:1px solid #2a3a4e; background:#0a0e17; color:#e5e7eb; font-size:14px;
          margin-bottom:14px; }
  button { width:100%; padding:10px; border-radius:6px; border:none; background:#3987e5;
           color:#fff; font-size:14px; font-weight:600; cursor:pointer; }
  button:hover { background:#2a78d6; }
  .error { color:#e66767; font-size:12.5px; margin-bottom:12px; text-align:center; }
</style></head>
<body>
  <form method="post" action="/login">
    <h1>Hilton's AI Crypto Trading</h1>
    __ERROR_HTML__
    <input type="password" name="password" placeholder="Password" autofocus required>
    <button type="submit">Log In</button>
  </form>
</body></html>"""


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("authenticated"):
        return RedirectResponse("/")
    return _LOGIN_PAGE_HTML.replace("__ERROR_HTML__", "")


@app.post("/login")
async def login_submit(request: Request):
    form = await request.form()
    password = form.get("password", "")
    if password and password == _DASHBOARD_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=303)
    error_html = '<div class="error">Incorrect password</div>'
    return HTMLResponse(_LOGIN_PAGE_HTML.replace("__ERROR_HTML__", error_html), status_code=401)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


_AUTH_EXEMPT_PATHS = {"/login"}


@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or path.startswith("/static/"):
        return await call_next(request)
    if not request.session.get("authenticated"):
        if path.startswith("/api/"):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        return RedirectResponse("/login")
    return await call_next(request)


app.add_middleware(
    SessionMiddleware,
    secret_key=_SESSION_SECRET_KEY,
    https_only=bool(os.environ.get("SSL_CERTFILE") and os.environ.get("SSL_KEYFILE")),
    max_age=30 * 24 * 60 * 60,
    # Distinct cookie name (2026-08-20, owner report) -- this app shares its physical
    # box AND Tailscale hostname with AIShortTrading (only the port differs: 8081 vs
    # 8082), and cookies are not scoped by port. Both apps previously used Starlette's
    # default cookie name ("session"), so logging into one silently overwrote the
    # other's session cookie with a value signed by a DIFFERENT secret key -- the
    # other app couldn't decode it, forced a re-login, and that re-login overwrote the
    # cookie again, breaking the first app right back. Renamed so the two can never
    # collide again, regardless of what else ever ends up sharing this box.
    session_cookie="aicrypto_session",
)


@app.on_event("startup")
async def startup():
    await state.portfolio.initialize()
    try:
        await state.order_manager.connect()
    except Exception:
        logger.exception("startup: broker connect failed — dashboard will run but trading is disabled")
    state.add_log("SYSTEM", "Hilton's AI Crypto Trading started")
    asyncio.create_task(state.scan_loop())
    asyncio.create_task(state.position_loop())
    asyncio.create_task(state.heartbeat_loop())
    asyncio.create_task(state.asset_profile_refresh_loop())
    asyncio.create_task(state.event_scan_loop())
    asyncio.create_task(state.watching_loop())
    asyncio.create_task(state._initial_universe_price_fill())


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    template_path = Path(__file__).parent / "templates" / "settings.html"
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


# Whitelisted "section.key" -> expected type. GET/POST /api/settings only ever
# read/write these -- an unlisted config field (universe, database.path, system.*,
# update.releases_repo) is never exposed to this endpoint at all.
_SETTINGS_FIELDS: dict[str, type] = {
    "trading.auto_execute": bool,
    "scan.interval_minutes": int,
    "portfolio.max_positions": int,
    "risk_management.max_position_pct": float,
    "risk_management.max_loss_per_trade_pct": float,
    "risk_management.min_cash_reserve_pct": float,
    "risk_management.daily_loss_limit_pct": float,
    "risk_management.drawdown_halt_pct": float,
    "risk_management.drawdown_defensive_pct": float,
    "risk_management.drawdown_exit_review_pct": float,
    "risk_management.trailing_stop_follow_tp_targets": bool,
    "take_profit.stop_loss_pct": float,
    "take_profit.t1_pct": float,
    "take_profit.t2_pct": float,
    "take_profit.t3_pct": float,
    "take_profit.final_tranche_trail_pct": float,
    "research.min_conviction_score": float,
    "research.min_risk_reward_ratio": float,
    "research.rr_conviction_step": float,
    "research.rr_floor": float,
    "research.long_term_trend_years": int,
    "research.model_quick_scan": str,
    "research.ai_chosen_stop_tp_enabled": bool,
    "research.ai_stop_loss_min_pct": float,
    "research.ai_stop_loss_max_pct": float,
    "research.ai_final_trail_min_pct": float,
    "research.ai_final_trail_max_pct": float,
    "event_scan.enabled": bool,
    "event_scan.interval_minutes": int,
    "event_scan.rsi_oversold": float,
    "event_scan.price_dip_pct": float,
    "event_scan.price_surge_pct": float,
    "event_scan.claude_cooldown_minutes": int,
    "research.sma_mode": lambda v: str(v) if v is not None else "auto",
    # Nullable (2026-08-20, owner report: settings save crashed with "int() argument
    # ... not 'NoneType'") -- these two are only meaningful in Manual mode (see the
    # Settings page's own field description) and are legitimately None the rest of the
    # time, but a plain settings-page save always resubmits every field's current
    # value regardless of mode. A bare `int` coercion crashed the WHOLE save (this
    # endpoint applies all-or-nothing) the moment sma_mode was anything but manual --
    # confirmed live: the deployed settings.yaml was even missing sma_mode/
    # sma_fast_period/sma_slow_period entirely (a real config-drift gap, config/
    # settings.yaml is never touched by Apply Update), so every save failed
    # unconditionally until this fix.
    "research.sma_fast_period": lambda v: (int(v) if v not in (None, "") else None),
    "research.sma_slow_period": lambda v: (int(v) if v not in (None, "") else None),
}


def _read_settings_snapshot() -> dict:
    result: dict = {}
    for dotted in _SETTINGS_FIELDS:
        section, key = dotted.split(".", 1)
        result.setdefault(section, {})[key] = state.config.get(section, {}).get(key)
    return result


@app.get("/api/settings")
async def get_settings():
    return _read_settings_snapshot()


@app.post("/api/settings")
async def save_settings(request: Request):
    body = await request.json()
    updates: dict = {}
    try:
        for dotted, raw_value in body.items():
            if dotted not in _SETTINGS_FIELDS:
                continue  # ignore unknown/unexpected fields rather than failing the whole save
            updates[dotted] = _SETTINGS_FIELDS[dotted](raw_value)
    except (TypeError, ValueError) as e:
        return JSONResponse({"status": "error", "error": f"Invalid value: {e}"}, status_code=400)

    if not updates:
        return JSONResponse({"status": "error", "error": "No valid fields in request"}, status_code=400)

    try:
        update_settings_yaml("config/settings.yaml", updates)
    except Exception as e:
        logger.exception("save_settings: failed to persist to settings.yaml")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

    # Applied live immediately -- state.config is the same dict object every module
    # (Portfolio, RiskManager, ResearchEngine, OrderManager, ...) holds a reference
    # to, so mutating it in place here takes effect on the very next read, no
    # restart needed.
    for dotted, value in updates.items():
        section, key = dotted.split(".", 1)
        state.config.setdefault(section, {})[key] = value

    state.add_log("SETTINGS", f"Settings updated: {', '.join(sorted(updates))}")
    return {"status": "ok", "config": _read_settings_snapshot()}


@app.get("/api/dashboard-poll")
async def dashboard_poll():
    return await state.get_init_payload()


@app.get("/api/recent-sells")
async def recent_sells():
    """Backs the dashboard's Recent Sells panel -- re-fetched by the frontend
    whenever a 'portfolio' WebSocket message arrives, since every sell path already
    triggers that broadcast."""
    return {"sells": await state.portfolio.get_recent_sells()}


@app.post("/api/manual-sell/{ticker:path}")
async def manual_sell(ticker: str):
    ok = await state.order_manager.execute_sell(ticker, reason="Manual sell")
    if ok:
        await state.broadcast({"type": "portfolio", "portfolio": state.get_portfolio_snapshot()})
    return {"status": "ok" if ok else "failed"}


@app.get("/api/asset-history/{ticker:path}")
async def asset_history(ticker: str, period: str = "1mo"):
    """Real historical daily OHLCV for one asset, backing both the tiny card
    sparklines (period=1mo, the default -- unchanged from before) and the detailed
    per-asset modal chart (period=6mo, real candles + SMA/support/resistance/entry/
    stop/TP reference lines). Deliberately separate from a portfolio-value-over-time
    chart -- no snapshot mechanism exists yet for this project's own equity curve
    (unlike AITrading's performance_history), so this only ever charts real
    per-asset PRICE, which market_data.get_historical()/get_technicals() already
    fetch live -- no fabricated data. ticker arrives URL-encoded (crypto symbols
    contain "/", e.g. BTC%2FUSD) -- FastAPI's :path converter decodes it back to
    the real "BTC/USD" form automatically. 'value' is kept alongside the full OHLC
    on every point so the existing card-sparkline caller needs no changes."""
    try:
        bars = await state.market_data.get_historical(ticker, period=period, interval="1d")
        technicals = await state.market_data.get_technicals(ticker)
    except Exception as e:
        logger.warning("asset_history failed for %s: %s", ticker, e)
        return {"points": [], "technicals": None}
    points = [
        {
            "time": b["date"], "value": b["close"],
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "volume": b["volume"],
        }
        for b in bars
    ]
    # Determine which SMA periods to render for this asset.
    # "auto":   use AI-chosen periods stored in the asset profile (if available)
    # "manual": use the two period numbers from settings
    # "flat":   always 50/200
    research_cfg = state.config.get("research", {})
    sma_mode = research_cfg.get("sma_mode", "auto")
    sma_fast, sma_slow = 50, 200  # safe defaults
    if sma_mode == "auto":
        profile = state.research_engine.asset_profiles.get(ticker, {})
        sma_fast = int(profile.get("sma_fast_period", 50))
        sma_slow = int(profile.get("sma_slow_period", 200))
    elif sma_mode == "manual":
        sma_fast = int(research_cfg.get("sma_fast_period", 50))
        sma_slow = int(research_cfg.get("sma_slow_period", 200))

    return {
        "points": points,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "technicals": {
            "sma_50": technicals.sma_50,
            "sma_200": technicals.sma_200,
            "rsi": technicals.rsi,
            "support_level": technicals.support_level,
            "resistance_level": technicals.resistance_level,
        },
    }


@app.post("/api/trigger-scan")
async def trigger_scan():
    """Manual on-demand scan trigger — real Claude spend across the whole universe,
    same as the scheduled cycle. No confirmation dialog at this layer (Phase 1 has no
    such UI yet) -- the caller is responsible for confirming with the owner before
    hitting this, same operational rule AITrading documents for its own equivalent."""
    asyncio.create_task(state.run_scan_cycle())
    return {"status": "started"}


@app.post("/api/pause")
async def pause_trading():
    """Stops every AI-spend loop (scan_loop, watching_loop, event_scan_loop,
    asset_profile_refresh_loop) from doing any real work -- zero Claude calls while
    paused (2026-08-20, owner request). Deliberately does NOT touch position_loop
    (quote refresh, exit-order sync, take-profit checks) -- held positions stay fully
    protected the whole time. Persisted to disk so a restart/deploy while paused
    doesn't silently resume spending."""
    state.paused = True
    state._save_run_state()
    state.add_log("SYSTEM",
        "Trading PAUSED — no new scans or AI analysis will run. Existing positions "
        "remain protected (stop-loss/trailing-stop/take-profit still active).", "WARNING")
    await state.broadcast({"type": "run_status", "run_status": state.run_status()})
    return {"status": "paused", "run_status": state.run_status()}


@app.post("/api/resume")
async def resume_trading():
    """Un-pauses only -- the counterpart to /api/pause. If the system is fully
    stopped, use /api/start instead (clears both flags)."""
    state.paused = False
    state._save_run_state()
    state.add_log("SYSTEM", "Trading resumed — scans and AI analysis active again.")
    await state.broadcast({"type": "run_status", "run_status": state.run_status()})
    return {"status": "resumed", "run_status": state.run_status()}


@app.post("/api/stop")
async def stop_trading():
    """Full stop (2026-08-20, owner request): every loop halts, including
    position_loop -- no broker-side position management of any kind happens while
    stopped, on top of everything /api/pause already blocks. Deliberately does NOT
    kill the process itself (own decision, 2026-08-20) -- a real process kill has no
    self-service recovery path for a non-technical customer (no dashboard to reach, no
    button to click, only SSH+systemctl); this way the dashboard and Start System
    button stay reachable, with the exact same real-world effect (zero AI spend, zero
    broker-side management) while stopped."""
    state.stopped = True
    state._save_run_state()
    state.add_log("SYSTEM",
        "Trading STOPPED — no scans, AI analysis, or position management (stop-loss/"
        "trailing-stop/take-profit) will run until Start System is clicked.", "WARNING")
    await state.broadcast({"type": "run_status", "run_status": state.run_status()})
    return {"status": "stopped", "run_status": state.run_status()}


@app.post("/api/start")
async def start_trading():
    """Clears both stopped and paused -- the counterpart to /api/stop, brings the
    system back to fully running from either a pause or a full stop."""
    state.paused = False
    state.stopped = False
    state._save_run_state()
    state.add_log("SYSTEM", "System started — scans, AI analysis, and position management all active again.")
    await state.broadcast({"type": "run_status", "run_status": state.run_status()})
    return {"status": "running", "run_status": state.run_status()}


@app.get("/api/update-status")
async def get_update_status():
    """Compares this install's local VERSION against the latest release on the
    distribution repo (update.releases_repo in config) — no credential needed since
    that repo is public. The frontend polls this every 60s so the badge appears on an
    already-open dashboard without a manual reload. The real GitHub lookup is cached
    for update.check_interval_minutes (default 2) via state._update_status_cache so
    frequent client polling never hits GitHub directly."""
    current = read_local_version(_VERSION_FILE_PATH) or "v0.0.0"
    repo = state.config.get("update", {}).get("releases_repo", "")

    if not repo:
        return {"current": current, "latest": None, "available": False,
                "notes": "", "severity": "routine"}

    check_interval = timedelta(
        minutes=state.config.get("update", {}).get("check_interval_minutes", 2)
    )
    cache_is_fresh = (
        state._update_status_cache is not None
        and state._update_status_cache_time is not None
        and datetime.now() - state._update_status_cache_time < check_interval
    )
    if not cache_is_fresh:
        try:
            state._update_status_cache = fetch_latest_release(repo)
            state._update_status_cache_time = datetime.now()
        except Exception:
            pass  # keep serving the last known-good cached result, if any

    release = state._update_status_cache
    if release is None:
        return {"current": current, "latest": None, "available": False,
                "notes": "", "severity": "routine"}

    return {
        "current": current,
        "latest": release["tag_name"],
        "available": is_newer(current, release["tag_name"]),
        "notes": release["notes"],
        "severity": release["severity"],
    }


_perf_history_cache: list[dict] | None = None
_perf_history_cache_time: datetime | None = None


@app.get("/api/perf-history")
async def get_perf_history():
    """Full daily equity history from Alpaca, cached 5 min. Returns points for
    the popup charts/lists plus computed week/month/ytd tile values."""
    global _perf_history_cache, _perf_history_cache_time
    now = datetime.now()
    if (
        _perf_history_cache is not None
        and _perf_history_cache_time is not None
        and (now - _perf_history_cache_time).total_seconds() < 300
    ):
        points = _perf_history_cache
    else:
        points = await state.order_manager.broker.get_portfolio_history_daily()
        _perf_history_cache = points
        _perf_history_cache_time = now

    p = state.portfolio
    current = p.total_value
    today = now.date()

    def start_equity(iso_floor: str) -> float | None:
        for pt in reversed(points):
            if pt["date"] < iso_floor:
                return pt["equity"]
        return points[0]["equity"] if points else None

    def tile(iso_floor: str) -> dict:
        ref = start_equity(iso_floor)
        if ref is None or ref == 0:
            return {"pnl": None, "pct": None}
        pnl = current - ref
        return {"pnl": round(pnl, 2), "pct": round(pnl / ref * 100, 4)}

    week_start = (today - timedelta(days=today.weekday())).isoformat()
    month_start = today.replace(day=1).isoformat()
    ytd_start = today.replace(month=1, day=1).isoformat()

    return {
        "points": points,
        "current_equity": round(current, 2),
        "tiles": {
            "week": tile(week_start),
            "month": tile(month_start),
            "ytd": tile(ytd_start),
        },
    }


def _restart_service_after_delay():
    """Runs in a background thread so the HTTP response can reach the client
    before the service (and this very process) restarts."""
    time.sleep(2)
    subprocess.run(["systemctl", "restart", "aicryptotrading"], check=False)


@app.post("/api/apply-update")
async def apply_update():
    """Manually-triggered only — never called automatically. Downloads the latest
    release, replaces only is_path_updatable()-allowed paths, reinstalls dependencies
    if requirements.txt changed, updates VERSION, and restarts. Guarded by
    _apply_update_in_progress so a second concurrent click is rejected cleanly.
    Uses sys.executable -m pip (not bare 'pip') so the right venv is always used."""
    if state._apply_update_in_progress:
        return {"status": "already_applying",
                "detail": "An update is already being applied — wait for it to finish."}
    state._apply_update_in_progress = True
    try:
        repo = state.config.get("update", {}).get("releases_repo", "")
        if not repo:
            return {"status": "error", "detail": "update.releases_repo not configured"}

        try:
            release = fetch_latest_release(repo)
        except Exception as exc:
            return {"status": "error", "detail": f"could not fetch latest release: {exc}"}

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = str(Path(tmp_dir) / "release.tar.gz")
            try:
                response = requests.get(release["tarball_url"], timeout=60)
                response.raise_for_status()
                Path(archive_path).write_bytes(response.content)
            except Exception as exc:
                return {"status": "error", "detail": f"could not download release archive: {exc}"}

            try:
                extract_dir = str(Path(tmp_dir) / "extracted")
                Path(extract_dir).mkdir()
                extracted_root = extract_release_archive(archive_path, extract_dir)
            except Exception as exc:
                return {"status": "error", "detail": f"could not extract release archive: {exc}"}

            old_req_path = Path(_INSTALL_ROOT) / "requirements.txt"
            old_req = old_req_path.read_text() if old_req_path.exists() else ""
            new_req_path = Path(extracted_root) / "requirements.txt"
            new_req = new_req_path.read_text() if new_req_path.exists() else old_req
            needs_pip = requirements_changed(old_req, new_req)

            copy_updatable_files(extracted_root, _INSTALL_ROOT)

            if needs_pip:
                pip_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    cwd=_INSTALL_ROOT,
                    capture_output=True,
                    text=True,
                )
                if pip_result.returncode != 0:
                    return {"status": "error",
                            "detail": f"pip install failed, service NOT restarted: {pip_result.stderr}"}

            write_local_version(_VERSION_FILE_PATH, release["tag_name"])

        threading.Thread(target=_restart_service_after_delay, daemon=True).start()
        return {"status": "applying", "target_version": release["tag_name"]}
    finally:
        state._apply_update_in_progress = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.websockets.append(websocket)
    try:
        await websocket.send_text(json.dumps(await state.get_init_payload(), default=str))
        while True:
            await websocket.receive_text()  # this app doesn't act on inbound WS messages yet
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.websockets:
            state.websockets.remove(websocket)
