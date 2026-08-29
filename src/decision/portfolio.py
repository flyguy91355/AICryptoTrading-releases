"""Portfolio state and tracking with SQLite persistence.

Ported from AITrading's Portfolio/Position. Two real differences:

- **No wash-sale tracking** (`recent_losses`, `_load_recent_losses`) -- dropped
  per the design spec, since the wash-sale rule doesn't apply to crypto under
  current IRS guidance.
- **No column-migration cruft.** AITrading's version carries a long history of
  `ALTER TABLE ... ADD COLUMN` migrations because its schema evolved under a
  live, already-running database over many months. This project starts fresh,
  so the schema below just includes every column from day one -- nothing to
  migrate.

Everything else (Position's P&L properties, the graduated take-profit/
trailing-stop fields, trade_id lot-linkage, day P/L snapshotting) is
asset-class-agnostic and carries over unchanged.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


def _format_trade_reason(reason: str | None) -> str:
    """Python port of formatSellReason() in dashboard.html -- kept in sync manually.
    Falls back to the raw string unchanged for anything not recognized."""
    if not reason:
        return "Not recorded"

    def frac(n: str) -> str:
        return "1/3" if n == "1" else "2/3"

    m = re.match(r"^Take-Profit T(\d)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)})"

    m = re.match(r"^Take-Profit T(\d) \(estimated\)$", reason)
    if m:
        return f"{frac(m.group(1))} — Take-Profit (T{m.group(1)}, estimated)"

    if reason == "Take-Profit (final tranche)":
        return "Final 1/3 — Take-Profit"

    if reason == "UNRECONCILED FILL (real order not found — needs manual review)":
        return "Unreconciled Fill — real order not found, needs manual review"

    if reason == "Stop Loss (gap-through market sell, full close)":
        return "Full Close — Stop Loss (price gapped through, sold at market)"
    if reason == "Stop Loss (gap-through market sell)":
        return "Partial — Stop Loss (price gapped through, sold at market)"

    if reason == "Stop loss hit":
        return "Full Close — Stop Loss"
    if reason == "Trailing stop hit":
        return "Full Close — Trailing Stop"
    if reason == "Manual sell":
        return "Full Close — Manual Sell"

    return reason


def _format_trade_history_summary(ticker: str, rows: list[tuple]) -> str:
    """Formats raw trade_history rows for one asset into a compact, chronological
    summary for inclusion in a Claude prompt as minor supplementary context. Each
    row is (ticker, action, shares, price, pnl, timestamp, reason). Returns an
    empty string for an asset with no trade history at all -- callers must treat
    that as "omit this section entirely", not display filler text."""
    if not rows:
        return ""
    lines = [f"Prior trading history on {ticker}:"]
    for _ticker, action, shares, price, pnl, timestamp, reason in rows:
        date = timestamp.split("T")[0] if timestamp else "unknown date"
        if action == "BUY":
            lines.append(f"- {date}: Bought at ${price:.2f}")
            continue
        reason_str = _format_trade_reason(reason)
        pnl_str = ""
        if pnl is not None:
            sign = "+" if pnl >= 0 else "-"
            pnl_str = f", P&L {sign}${abs(pnl):.2f}"
            if shares:
                implied_entry = price - (pnl / shares)
                if implied_entry > 0:
                    pnl_pct = (pnl / (shares * implied_entry)) * 100
                    pct_sign = "+" if pnl_pct >= 0 else ""
                    pnl_str += f" ({pct_sign}{pnl_pct:.1f}%)"
        lines.append(f"- {date}: Sold ({reason_str}) at ${price:.2f}{pnl_str}")
    return "\n".join(lines)


def _format_sell_analysis_summary(rows: list[dict]) -> str:
    """Formats completed "Recent Sell" post-mortems (2026-08-21, same feature as
    AITrading/AIShortTrading's own) for inclusion in a Claude prompt as minor
    supplementary context -- same "for reference only" framing as
    _format_trade_history_summary, appended alongside it rather than replacing it. Each
    dict is one row from Portfolio.get_recent_sell_analyses (already filtered to
    post_mortem_thesis IS NOT NULL, newest close first). Includes the delayed follow-up
    verdict when one has been generated. Returns an empty string for no rows -- callers
    must treat that as "omit this section entirely"."""
    if not rows:
        return ""
    lines = ["Past sell post-mortem(s) for this asset (for reference only):"]
    for r in rows:
        date = (r.get("closed_at") or "").split("T")[0] or "unknown date"
        lines.append(f"- {date}: {r.get('post_mortem_thesis', '')}")
        if r.get("followup_reasoning"):
            lines.append(f"  Follow-up: {r['followup_reasoning']}")
    return "\n".join(lines)


def _format_analysis_history_summary(ticker: str, rows: list[tuple]) -> str:
    """Formats every past real analysis for this ticker into a compact, chronological
    summary for _build_analysis_history_section (2026-08-21, same feature as
    AITrading/AIShortTrading's own). Each row is (generated_at, conviction_score,
    signal, entry_price, fair_value_estimate, watch_condition). Deliberately includes
    ALL history the caller passes in, not a capped recent window. Returns an empty
    string for a ticker with no history at all."""
    if not rows:
        return ""
    lines = [f"Prior analysis history on {ticker} (chronological):"]
    for generated_at, conviction_score, signal, entry_price, fair_value_estimate, watch_condition in rows:
        date = generated_at.split("T")[0] if generated_at else "unknown date"
        line = f"- {date}: {signal}, conviction {conviction_score:.1f}/10"
        if entry_price:
            line += f", entry ${entry_price:.2f}"
        if fair_value_estimate:
            line += f", fair value ${fair_value_estimate:.2f}"
        line += f" — watch: {watch_condition}" if watch_condition else " — no watch condition stated"
        lines.append(line)
    return "\n".join(lines)


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_targets: list[float]
    sector: str
    opened_at: datetime
    trailing_stop: float | None = None
    day_open_price: float | None = None
    final_tranche_start_price: float | None = None
    realized_pnl: float = 0.0
    shares_sold: float = 0.0
    t1_target_price: float | None = None
    t2_target_price: float | None = None
    profit_target_hit: bool = False
    trade_id: str | None = None
    final_trail_pct: float | None = None
    # "Why AI Bought This" (2026-08-21, same feature as AITrading's own -- see that
    # project's CLAUDE_HISTORY.md 2026-08-21 entry for the full design) -- a snapshot
    # of the AI's real buy-time decision, captured once at the moment of purchase and
    # never regenerated or overwritten by a later re-analysis while the position is
    # held. buy_rr/buy_required_rr are the exact numbers that cleared the gate at that
    # moment. None/"" for any position opened before this field existed.
    buy_thesis: str = ""
    buy_reasoning: str = ""
    buy_conviction: int | None = None
    buy_signal: str = ""
    buy_rr: float | None = None
    buy_required_rr: float | None = None
    buy_fair_value: float | None = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

    @property
    def lifetime_pnl(self) -> float:
        """Total gain/loss since this position was first opened: unrealized gain on
        whatever shares remain, PLUS every partial exit's real realized gain along
        the way."""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def lifetime_pnl_pct(self) -> float:
        original_cost = (self.shares + self.shares_sold) * self.entry_price
        if original_cost == 0:
            return 0.0
        return (self.lifetime_pnl / original_cost) * 100

    @property
    def day_pnl(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        return (self.current_price - ref) * self.shares

    @property
    def day_pnl_pct(self) -> float:
        ref = self.day_open_price if self.day_open_price is not None else self.entry_price
        if not ref:
            return 0.0
        return ((self.current_price - ref) / ref) * 100


class Portfolio:
    def __init__(self, config: dict):
        self.config = config
        self.initial_capital = config["portfolio"]["initial_capital"]
        self.cash = self.initial_capital
        self.positions: dict[str, Position] = {}
        self.peak_value = self.initial_capital
        self.day_start_value = self.initial_capital
        self.day_start_date: str | None = None
        # Set while a rotation swap's sell is in flight (submitted but not yet filled) —
        # during this window the sold position and its replacement can be briefly
        # double-counted in total_value, so update_peak() must not ratchet on it.
        self._rotation_in_progress = False
        self.db_path = config.get("database", {}).get("path", "data/aicryptotrading.db")
        self._db: aiosqlite.Connection | None = None

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def total_pnl(self) -> float:
        return self.total_value - self.initial_capital

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100

    @property
    def cash_pct(self) -> float:
        if self.total_value == 0:
            return 0.0
        return (self.cash / self.total_value) * 100

    @property
    def day_pnl(self) -> float:
        return self.total_value - self.day_start_value

    async def initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        # 2026-08-28, audit finding, same class already fixed on the sibling stock
        # projects ("SQLite Concurrency Hardening") -- widened past sqlite3's
        # implicit 5.0s default as defense-in-depth against lock contention from
        # this file's other writer (web/app.py's ai_log persistence, raw sqlite3
        # on a background thread). WAL mode itself is set once at app startup via
        # web/app.py's _ensure_wal_mode, since it's a property of the DB file, not
        # any one connection.
        self._db = await aiosqlite.connect(self.db_path, timeout=20.0)

        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                ticker TEXT PRIMARY KEY,
                shares REAL,
                entry_price REAL,
                current_price REAL,
                stop_loss REAL,
                take_profit_targets TEXT,
                sector TEXT,
                opened_at TEXT,
                trailing_stop REAL,
                day_open_price REAL,
                final_tranche_start_price REAL,
                realized_pnl REAL,
                shares_sold REAL,
                t1_target_price REAL,
                t2_target_price REAL,
                profit_target_hit INTEGER,
                trade_id TEXT,
                final_trail_pct REAL
            )
        """)
        # final_trail_pct was added after this project's initial launch (2026-08-15,
        # AI-judged trailing-stop feature) -- CREATE TABLE IF NOT EXISTS above only
        # covers a brand-new DB, so an already-live one (real paper positions since
        # launch day) needs this one-time ALTER. try/except is the standard SQLite
        # idiom for "add column if missing" -- no IF NOT EXISTS support for ADD COLUMN.
        try:
            await self._db.execute("ALTER TABLE positions ADD COLUMN final_trail_pct REAL")
            await self._db.commit()
        except Exception:
            pass  # column already exists
        # "Why AI Bought This" (2026-08-21, same feature as AITrading's own) -- see
        # Position's own field docstring. Existing rows get NULL/empty on every one of
        # these 7 columns, same "add column if missing" try/except idiom as
        # final_trail_pct above.
        for _col, _type in (
            ("buy_thesis", "TEXT"), ("buy_reasoning", "TEXT"),
            ("buy_conviction", "INTEGER"), ("buy_signal", "TEXT"),
            ("buy_rr", "REAL"), ("buy_required_rr", "REAL"), ("buy_fair_value", "REAL"),
        ):
            try:
                await self._db.execute(f"ALTER TABLE positions ADD COLUMN {_col} {_type}")
                await self._db.commit()
            except Exception:
                pass  # column already exists
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL,
                peak_value REAL,
                day_start_value REAL,
                day_start_date TEXT
            )
        """)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                action TEXT,
                shares REAL,
                price REAL,
                pnl REAL,
                timestamp TEXT,
                reason TEXT,
                trade_id TEXT
            )
        """)
        # "Recent Sell" post-mortem (2026-08-21, same feature as AITrading/
        # AIShortTrading's own) -- a position's buy-side thesis/reasoning/etc. is about
        # to be lost the moment close_position_async deletes the Position row, so it's
        # snapshotted here (see close_position_async below) alongside the raw close
        # facts, one row per trade_id. post_mortem_* fields start NULL and are filled
        # in by a separate periodic job in web/app.py (this file has no Claude client)
        # once per closed trade; followup_* fields are filled in by a second, delayed
        # pass using price action after the sale.
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS sell_analysis (
                trade_id TEXT PRIMARY KEY,
                ticker TEXT,
                buy_thesis TEXT,
                buy_reasoning TEXT,
                buy_conviction INTEGER,
                buy_signal TEXT,
                buy_rr REAL,
                buy_required_rr REAL,
                buy_fair_value REAL,
                entry_price REAL,
                exit_price REAL,
                opened_at TEXT,
                closed_at TEXT,
                post_mortem_thesis TEXT,
                post_mortem_reasoning TEXT,
                generated_at TEXT,
                followup_due_date TEXT,
                followup_reasoning TEXT,
                followup_generated_at TEXT
            )
        """)
        await self._db.commit()

        # "Analysis History" feed-forward (2026-08-21, same feature as AITrading/
        # AIShortTrading's own -- see AITrading's CLAUDE_HISTORY.md 2026-08-21 entry)
        # -- one row per real (non-fallback) analysis of a ticker, appended (never
        # overwritten, unlike the research_reports/reports_cache.json "latest only"
        # cache) so a recurring candidate's whole arc -- including any
        # previously-stated watch_condition -- survives being scanned again and again.
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                generated_at TEXT,
                conviction_score REAL,
                signal TEXT,
                entry_price REAL,
                fair_value_estimate REAL,
                watch_condition TEXT
            )
        """)
        await self._db.commit()

        await self._load_state()

    async def _load_state(self):
        async with self._db.execute(
            "SELECT cash, peak_value, day_start_value, day_start_date FROM portfolio_state WHERE id = 1"
        ) as cur:
            row = await cur.fetchone()
            if row:
                self.cash = row[0]
                self.peak_value = row[1]
                self.day_start_value = row[2]
                self.day_start_date = row[3] if len(row) > 3 else None

        async with self._db.execute("SELECT * FROM positions") as cur:
            async for row in cur:
                targets = json.loads(row[5]) if row[5] else []
                self.positions[row[0]] = Position(
                    ticker=row[0],
                    shares=row[1],
                    entry_price=row[2],
                    current_price=row[3],
                    stop_loss=row[4],
                    take_profit_targets=targets,
                    sector=row[6],
                    opened_at=datetime.fromisoformat(row[7]),
                    trailing_stop=row[8],
                    day_open_price=row[9] if len(row) > 9 else None,
                    final_tranche_start_price=row[10] if len(row) > 10 else None,
                    realized_pnl=row[11] if len(row) > 11 and row[11] is not None else 0.0,
                    shares_sold=row[12] if len(row) > 12 and row[12] is not None else 0.0,
                    t1_target_price=row[13] if len(row) > 13 else None,
                    t2_target_price=row[14] if len(row) > 14 else None,
                    profit_target_hit=bool(row[15]) if len(row) > 15 and row[15] is not None else False,
                    trade_id=row[16] if len(row) > 16 else None,
                    final_trail_pct=row[17] if len(row) > 17 else None,
                    buy_thesis=row[18] if len(row) > 18 and row[18] is not None else "",
                    buy_reasoning=row[19] if len(row) > 19 and row[19] is not None else "",
                    buy_conviction=row[20] if len(row) > 20 else None,
                    buy_signal=row[21] if len(row) > 21 and row[21] is not None else "",
                    buy_rr=row[22] if len(row) > 22 else None,
                    buy_required_rr=row[23] if len(row) > 23 else None,
                    buy_fair_value=row[24] if len(row) > 24 else None,
                )

    async def _save_state(self):
        if not self._db:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO portfolio_state (id, cash, peak_value, day_start_value, day_start_date) VALUES (1, ?, ?, ?, ?)",
            (self.cash, self.peak_value, self.day_start_value, self.day_start_date),
        )
        await self._db.commit()

    async def _save_position(self, position: Position):
        if not self._db:
            return
        await self._db.execute(
            "INSERT OR REPLACE INTO positions (ticker, shares, entry_price, current_price, stop_loss, take_profit_targets, sector, opened_at, trailing_stop, day_open_price, final_tranche_start_price, realized_pnl, shares_sold, t1_target_price, t2_target_price, profit_target_hit, trade_id, final_trail_pct, buy_thesis, buy_reasoning, buy_conviction, buy_signal, buy_rr, buy_required_rr, buy_fair_value) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                position.ticker, position.shares, position.entry_price,
                position.current_price, position.stop_loss,
                json.dumps(position.take_profit_targets), position.sector,
                position.opened_at.isoformat(), position.trailing_stop,
                position.day_open_price, position.final_tranche_start_price,
                position.realized_pnl, position.shares_sold,
                position.t1_target_price, position.t2_target_price,
                int(position.profit_target_hit), position.trade_id,
                position.final_trail_pct,
                position.buy_thesis, position.buy_reasoning, position.buy_conviction,
                position.buy_signal, position.buy_rr, position.buy_required_rr,
                position.buy_fair_value,
            ),
        )
        await self._db.commit()

    async def _remove_position_db(self, ticker: str):
        if not self._db:
            return
        await self._db.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
        await self._db.commit()

    async def get_trade_history_summary(self, ticker: str) -> str:
        """Every trade_history row for this asset, formatted as compact context for a
        Claude prompt when reconsidering it as a buy candidate. Fails open (returns
        "") on any DB error -- must never delay or break a real buy decision."""
        if not self._db:
            return ""
        try:
            async with self._db.execute(
                "SELECT ticker, action, shares, price, pnl, timestamp, reason "
                "FROM trade_history WHERE ticker = ? ORDER BY timestamp",
                (ticker,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_trade_history_summary failed for %s: %s", ticker, e)
            return ""
        summary = _format_trade_history_summary(ticker, rows)

        # "Recent Sell" post-mortem feed-forward (2026-08-21, same feature as
        # AITrading/AIShortTrading's own) -- same minor, supplementary framing as the
        # trade-history summary above. Fails open exactly like the query above.
        try:
            sell_analyses = await self.get_recent_sell_analyses(ticker)
        except Exception as e:
            logger.warning("get_recent_sell_analyses failed for %s: %s", ticker, e)
            sell_analyses = []
        post_mortem_summary = _format_sell_analysis_summary(sell_analyses)
        if post_mortem_summary:
            summary = f"{summary}\n\n{post_mortem_summary}" if summary else post_mortem_summary
        return summary

    async def get_recent_sells(self, limit: int = 30) -> list[dict]:
        """Most-recent-first SELL rows from trade_history, for the dashboard's Recent
        Sells panel. Fails open (returns []) on any DB error -- this is display-only,
        must never block anything else."""
        if not self._db:
            return []
        try:
            async with self._db.execute(
                "SELECT ticker, shares, price, pnl, timestamp, reason, trade_id "
                "FROM trade_history WHERE action = 'SELL' ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_recent_sells failed: %s", e)
            return []
        return [
            {
                "ticker": r[0], "shares": r[1], "price": r[2],
                "pnl": r[3], "timestamp": r[4], "reason": r[5],
                # trade_id (2026-08-21) -- backs the Recent Sells row click / "Recent
                # Sell" post-mortem popup; NULL (None) for a row that predates
                # trade_id tracking, which the frontend treats as "not clickable".
                "trade_id": r[6],
            }
            for r in rows
        ]

    def update_peak(self):
        if self._rotation_in_progress:
            return
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value

    def new_trading_day(self, today_str: str | None = None):
        """Snapshots the "start of today" baseline both portfolio-wide (day_start_value)
        and per-position (day_open_price). Crypto trades continuously, so "today" is
        just a plain calendar-day boundary (ET, for consistency with AITrading's own
        convention) -- not tied to any market session."""
        self.day_start_value = self.total_value
        if today_str is not None:
            self.day_start_date = today_str
        for pos in self.positions.values():
            pos.day_open_price = pos.current_price

    def add_position(self, position: Position):
        self.positions[position.ticker] = position
        self.cash -= position.cost_basis

    def close_position(self, ticker: str) -> float:
        position = self.positions.pop(ticker, None)
        if position is None:
            return 0.0
        self.cash += position.market_value
        self.update_peak()
        return position.unrealized_pnl

    async def add_position_async(self, position: Position):
        self.add_position(position)
        await self._save_position(position)
        await self._save_state()

    async def close_position_async(
        self, ticker: str, *, exit_shares: float = None, exit_price: float = None,
        reason: str = "",
    ) -> float:
        """Real, confirmed bug fixed 2026-08-21: this branch's own comment claimed
        "cash was already credited per-fill in check_take_profits," but neither of
        this function's two REAL callers with both exit_shares/exit_price set
        (OrderManager._handle_apparent_close, for a stop-loss/apparent close;
        OrderManager.execute_sell, for a manual dashboard sell) ever credits cash
        beforehand -- check_take_profits doesn't even call this function at all; it
        credits cash and closes the position entirely inline, itself. Every position
        closed via either real caller silently never had its sale proceeds added to
        cash. Confirmed live: LINK/USD's final tranche closed via
        _handle_apparent_close (an unreconciled fill, ~$390.44 in real proceeds) with
        the PNL correctly recorded in trade_history for display, but the cash itself
        never credited -- local cash read ~$391 below Alpaca's real cash, the root
        cause behind a real "-261"-magnitude Day P/L distortion the owner reported
        twice (see docs/CLAUDE_HISTORY.md's 2026-08-21 entry for the full incident,
        including the separate, also-real day_start_value/day_start_date bug found
        alongside this one)."""
        pos = self.positions.get(ticker)
        _exit_shares = exit_shares if exit_shares is not None else (pos.shares if pos else 0)
        _exit_price = exit_price if exit_price is not None else (pos.current_price if pos else 0)
        _trade_id = pos.trade_id if pos else None
        if exit_shares is not None and exit_price is not None:
            position = self.positions.pop(ticker, None)
            if position:
                self.cash += _exit_price * _exit_shares
                self.update_peak()
            pnl = (_exit_price - position.entry_price) * _exit_shares if position else 0
        else:
            pnl = self.close_position(ticker)
        await self._remove_position_db(ticker)
        await self._save_state()
        if self._db:
            await self._db.execute(
                "INSERT INTO trade_history (ticker, action, shares, price, pnl, timestamp, reason, trade_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ticker, "SELL", _exit_shares, _exit_price, pnl, datetime.now().isoformat(), reason, _trade_id),
            )
            await self._db.commit()
            # "Recent Sell" post-mortem snapshot (2026-08-21, same feature as
            # AITrading/AIShortTrading's own) -- close_position_async is always a FULL
            # close (the position is gone from self.positions either way by this
            # point), so this is exactly the one moment pos's buy_thesis/etc is still
            # available before it's lost for good. Skipped for a ticker with no real
            # buy-time rationale (pre-feature position, or a non-AI-driven signal
            # source) -- nothing worth a post-mortem without it.
            if pos and pos.trade_id and pos.buy_thesis:
                await self._db.execute(
                    "INSERT OR IGNORE INTO sell_analysis "
                    "(trade_id, ticker, buy_thesis, buy_reasoning, buy_conviction, buy_signal, "
                    "buy_rr, buy_required_rr, buy_fair_value, entry_price, opened_at, closed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        pos.trade_id, ticker, pos.buy_thesis, pos.buy_reasoning,
                        pos.buy_conviction, pos.buy_signal, pos.buy_rr, pos.buy_required_rr,
                        pos.buy_fair_value, pos.entry_price,
                        pos.opened_at.isoformat() if pos.opened_at else None,
                        datetime.now().isoformat(),
                    ),
                )
                await self._db.commit()
        return pnl

    _SELL_ANALYSIS_COLUMNS = (
        "trade_id", "ticker", "buy_thesis", "buy_reasoning", "buy_conviction", "buy_signal",
        "buy_rr", "buy_required_rr", "buy_fair_value", "entry_price", "exit_price",
        "opened_at", "closed_at",
        "post_mortem_thesis", "post_mortem_reasoning", "generated_at",
        "followup_due_date", "followup_reasoning", "followup_generated_at",
    )

    async def get_pending_sell_analyses(self, limit: int = 5) -> list[dict]:
        """Closed trades whose immediate post-mortem hasn't been generated yet -- the
        queue a periodic job (web/app.py, which owns the real Claude client) drains one
        real API call at a time."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE post_mortem_thesis IS NULL "
            "ORDER BY closed_at ASC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def get_sell_analysis(self, trade_id: str) -> dict | None:
        if not self._db:
            return None
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE trade_id = ?", (trade_id,),
        ) as cur:
            row = await cur.fetchone()
        return dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) if row else None

    async def save_sell_analysis_post_mortem(
        self, trade_id: str, thesis: str, reasoning: str, followup_due_date: str,
        exit_price: float | None = None,
    ):
        if not self._db:
            return
        await self._db.execute(
            "UPDATE sell_analysis SET post_mortem_thesis = ?, post_mortem_reasoning = ?, "
            "generated_at = ?, followup_due_date = ?, exit_price = ? WHERE trade_id = ?",
            (thesis, reasoning, datetime.now().isoformat(), followup_due_date,
             exit_price, trade_id),
        )
        await self._db.commit()

    async def get_due_sell_analysis_followups(self, today_str: str, limit: int = 5) -> list[dict]:
        """Trades whose delayed follow-up check (price action after the sale) is due."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE post_mortem_thesis IS NOT NULL "
            "AND followup_generated_at IS NULL AND followup_due_date IS NOT NULL "
            "AND followup_due_date <= ? ORDER BY followup_due_date ASC LIMIT ?",
            (today_str, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def save_sell_analysis_followup(self, trade_id: str, followup_reasoning: str):
        if not self._db:
            return
        await self._db.execute(
            "UPDATE sell_analysis SET followup_reasoning = ?, followup_generated_at = ? "
            "WHERE trade_id = ?",
            (followup_reasoning, datetime.now().isoformat(), trade_id),
        )
        await self._db.commit()

    async def get_recent_sell_analyses(self, ticker: str, limit: int = 2) -> list[dict]:
        """Completed post-mortems for this ticker, newest close first -- feeds
        get_trade_history_summary's minor supplementary prompt context. Excludes a
        still-pending row (no post_mortem_thesis yet)."""
        if not self._db:
            return []
        cols = ", ".join(self._SELL_ANALYSIS_COLUMNS)
        async with self._db.execute(
            f"SELECT {cols} FROM sell_analysis WHERE ticker = ? AND post_mortem_thesis IS NOT NULL "
            "ORDER BY closed_at DESC LIMIT ?",
            (ticker, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(zip(self._SELL_ANALYSIS_COLUMNS, row)) for row in rows]

    async def save_analysis_history(
        self, ticker: str, generated_at: str, conviction_score: float, signal: str,
        entry_price: float | None, fair_value_estimate: float | None, watch_condition: str,
    ) -> None:
        """Appends one row per real analysis (2026-08-21, "Analysis History"
        feed-forward feature, same as AITrading/AIShortTrading's own) -- called by
        DashboardState._persist_report (web/app.py), fire-and-forget. Never overwrites
        (unlike research_reports/reports_cache.json, which only ever keeps the LATEST
        report per ticker)."""
        if not self._db:
            return
        await self._db.execute(
            "INSERT INTO analysis_history "
            "(ticker, generated_at, conviction_score, signal, entry_price, "
            "fair_value_estimate, watch_condition) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticker, generated_at, conviction_score, signal, entry_price,
             fair_value_estimate, watch_condition),
        )
        await self._db.commit()

    async def get_analysis_history_summary(self, ticker: str) -> str:
        """Every past real analysis for this ticker, formatted as prompt context for
        the NEXT analysis of the same ticker (2026-08-21). Fails open (returns "") on
        any DB error or if the DB hasn't been initialized yet."""
        if not self._db:
            return ""
        try:
            async with self._db.execute(
                "SELECT generated_at, conviction_score, signal, entry_price, "
                "fair_value_estimate, watch_condition FROM analysis_history "
                "WHERE ticker = ? ORDER BY generated_at",
                (ticker,),
            ) as cur:
                rows = await cur.fetchall()
        except Exception as e:
            logger.warning("get_analysis_history_summary failed for %s: %s", ticker, e)
            return ""
        return _format_analysis_history_summary(ticker, rows)
