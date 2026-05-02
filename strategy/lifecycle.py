# strategy/lifecycle.py

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from execution.notifier import TelegramNotifier
from strategy.account_state import account_state

def _tg_debug(msg: str) -> None:
    """Fire-and-forget debug message to Telegram. Never raises."""
    try:
        TelegramNotifier().debug(f"[LIFECYCLE] {msg}")
    except Exception:
        print(f"[TG DEBUG FALLBACK] {msg}")

def _update_atr(prev_atr, prev_close, high, low, close, period=14):
    tr = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )
    alpha = 2 / (period + 1)
    return tr if prev_atr is None else prev_atr + alpha * (tr - prev_atr)

POSITIONS_DIR = "data/positions"
POSITIONS_FILE   = os.path.join(POSITIONS_DIR, "open_positions.json")
BAR_HISTORY_FILE = os.path.join(POSITIONS_DIR, "bar_history.json")
ENTRY_TS_FILE    = os.path.join(POSITIONS_DIR, "last_entry_ts.json")
EXECUTED_SIGNALS_FILE   = os.path.join(POSITIONS_DIR, "executed_signals.json")
REENTRY_LOCK_FILE = os.path.join(POSITIONS_DIR, "reentry_lock.json")

class PositionManager:

    # ==========================================================
    # TRADE LIFECYCLE SETTINGS (mirrors SignalBacktester exactly)
    # ==========================================================
    POSITION_VALUE_USDT = 10
    INCUBATION_BARS = 6        # 30 minutes (6×5m)
    VALIDATION_BARS = 18       # 90 minutes total
    PRESSURE_BARS   = 6        # stop proximity exit

    NO_FOLLOW_MFE  = 0.3       # 0.3R required after validation
    STOP_PROXIMITY = 0.2       # within 0.2R of stop = danger

    ATR_MULT         = 1.5
    ATR_AFTER_HALF_R = 3.0   # wide early trail — gives trade room
    ATR_AFTER_ONE_R  = 1.5   # tighter once 1R+ secured

    USE_ACCOUNT_GATES = False

    # ==========================================================
    # EXECUTION COST MODEL
    # ==========================================================
    SLIPPAGE_BPS = 5      # 0.05% market order slippage
    SPREAD_BPS   = 3      # 0.03% half-spread (taker crosses spread)
    TOTAL_COST_BPS = SLIPPAGE_BPS + SPREAD_BPS   # 8bps per side
    SIGNAL_EXPIRY_BARS      = 6    # replay: signal dies after 6×5m = 30 minutes
    SIGNAL_EXPIRY_BARS_LIVE = 11   # live: allow up to 55 minutes for scheduler jitter

    def __init__(self, persist=True, notify=True):
        self.persist  = persist
        self.notify   = notify
        # proxy for live vs replay: replay uses persist=False, live uses persist=True
        self._is_live = persist
        self.positions = {}
        os.makedirs(POSITIONS_DIR, exist_ok=True)

        self.notifier = TelegramNotifier()

        if self.persist:
            self._load()
            if self.USE_ACCOUNT_GATES:
                account_state.open_positions = len(self.positions)
                account_state._save()

        # Prevent duplicate entries on same candle
        self._last_entry_ts = {}

        # 5m bar cache per symbol
        self._bar_history: dict[str, list] = {}

        # Streaming ATR state per symbol (FAST)
        self._atr_state = {}
        self._executed_signals = set()
        self._reentry_lock: dict[str, int] = {}
        self._reentry_lock_ts: dict[str, pd.Timestamp] = {}
        self._dirty = False

    # --------------------------------------------------
    # MAIN UPDATE  (called once per new 5M candle close)
    # --------------------------------------------------
    def update(
        self,
        df: pd.DataFrame,
        symbol: str,
        lltf_df: pd.DataFrame,
        external_signal: int,
        external_row: pd.Series,
        current_5m_row: pd.Series,
    ):

        signal = 0 if pd.isna(external_signal) else int(external_signal)
        position = self.positions.get(symbol)

        o = float(current_5m_row["open"])
        h = float(current_5m_row["high"])
        l = float(current_5m_row["low"])
        c = float(current_5m_row["close"])

        # =====================================================
        # ROLLING ATR (precomputed — O(1) lookup)
        # =====================================================
        atr = float(current_5m_row["ATR"]) if "ATR" in current_5m_row.index and not pd.isna(current_5m_row["ATR"]) else None

        current_ts = current_5m_row.name

        signal = 0 if pd.isna(external_signal) else int(external_signal)

        # -------------------------------------------------------
        # APPEND current 5M bar to history (for window checks)
        # -------------------------------------------------------
        if position:

            # APPEND BAR FIRST
            self._bar_history.setdefault(symbol, []).append({
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "ATR": atr,
                "ts": current_ts,
            })

            # KEEP WINDOW SIZE
            if len(self._bar_history[symbol]) > 200:
                self._bar_history[symbol] = self._bar_history[symbol][-200:]

            # DEFINE THIS EARLY (BEFORE USING IT)
            is_entry_candle = pd.Timestamp(position["entry_5m_ts"]) == current_ts
            # print(f"[DEBUG] bars_in_trade={position['bars_in_trade']} is_entry_candle={is_entry_candle} entry_5m_ts={position['entry_5m_ts']} current_ts={current_ts}")

            if is_entry_candle:
                # still allow bar history update + ATR tracking
                skip_exit_checks = True
            else:
                skip_exit_checks = False

            entry_price = position["entry_price"]
            stop        = position["stop_loss"]
            side        = position["direction"]
            R           = abs(entry_price - position["initial_stop"])  # always original risk

            if R == 0:
                R = 1e-9   # safety

            # Current price for R calculation (best price this bar)
            current_price = h if side == 1 else l

            # Trail anchored to MFE price (best price ever seen)
            if side == 1:
                if h - entry_price >= position["MFE"]:
                    position["mfe_price"] = h
                    position["last_mfe_bar"] = position["bars_in_trade"]
            else:
                if entry_price - l >= position["MFE"]:
                    position["mfe_price"] = l
                    position["last_mfe_bar"] = position["bars_in_trade"]

            trail_price = position.get("mfe_price", c)

            # =========================
            # TRUE MFE / MAE TRACKING
            # =========================

            if "MFE" not in position:
                position["MFE"] = 0.0
            if "MAE" not in position:
                position["MAE"] = 0.0

            # per-bar movement (NOT candle-close derived pnl)
            if side == 1:
                move_high = h - entry_price
                move_low  = l - entry_price

                position["MFE"] = max(position["MFE"], move_high)
                position["MAE"] = min(position["MAE"], move_low)

            else:
                move_high = entry_price - l
                move_low  = entry_price - h

                position["MFE"] = max(position["MFE"], move_high)
                position["MAE"] = min(position["MAE"], move_low)

            # -------------------------------------------------------
            # BUILD 5M WINDOW (REAL ATR HISTORY — BACKTEST PARITY)
            # -------------------------------------------------------
            window_5m = pd.DataFrame(self._bar_history[symbol])

            # convert to R
            position["mfe_r"] = position["MFE"] / R

            # ONLY for reporting (not exit logic anymore)
            if side == 1:
                position["pnl_r"] = (c - entry_price) / R
            else:
                position["pnl_r"] = (entry_price - c) / R

            exit_reason = None
            fill_price  = None

            def try_exit(reason, price=None):
                nonlocal exit_reason, fill_price
                if exit_reason is None:
                    exit_reason = reason
                    if price is not None:
                        fill_price = price

            # ===================================================
            # STAGE 2 — EXPANSION (>90m)
            # Backtest: opposite_impulse again (labeled momentum_decay)
            # ===================================================
            if not skip_exit_checks:
                self._update_dynamic_stop(position, trail_price, atr, side)

                # if self._stealth_distribution_exit(window_5m, position, side):
                #     try_exit("stealth_distrib", current_price)

                # if self._momentum_decay_exit(position):
                #     try_exit("momentum_decay", current_price)

                if self._opposite_impulse_exit(window_5m, side):
                    try_exit("opposite_impulse", current_price)

                # ===================================================
                # HARD STOP — always last
                # ===================================================
                if side == 1 and l <= position["stop_loss"]:
                    try_exit("stop_loss", position["stop_loss"])
                elif side == -1 and h >= position["stop_loss"]:
                    try_exit("stop_loss", position["stop_loss"])

                if exit_reason:
                    if fill_price is None:
                        fill_price = float(current_5m_row["close"])

                    closed = self._close(symbol, fill_price, current_ts, exit_reason)

                    return {"state": "CLOSED", "exit": closed}

            # increment real trade age AFTER exit checks (backtest parity)
            position["bars_in_trade"] += 1
            bars_5m = position["bars_in_trade"]
        
        # =====================================================
        # 🔓 REENTRY UNLOCK LOGIC
        # =====================================================
        if symbol in self._reentry_lock:
            locked_dir = self._reentry_lock[symbol]

            # unlock when signal resets or flips
            if signal == 0 or signal == -locked_dir:
                _tg_debug(f"[REENTRY UNLOCK] {symbol} signal={signal}")
                self._reentry_lock.pop(symbol, None)
                self._dirty = True

        # ===================================================
        # SIGNAL EXPIRY CHECK
        # ===================================================
        if signal != 0 and not position:
            signal_age_bars = len(
                lltf_df[(lltf_df.index > external_row.name) & (lltf_df.index <= current_ts)]
            )
            # Use wider expiry window — replay uses strict 6-bar window,
            # but live needs buffer for scheduler jitter and late cron fires
            # self._is_live is set in __init__ based on persist flag as a proxy
            expiry_limit = self.SIGNAL_EXPIRY_BARS_LIVE if self._is_live else self.SIGNAL_EXPIRY_BARS
            _tg_debug(f"[EXPIRY CHECK] {symbol} ts={current_ts} signal_age={signal_age_bars} limit={expiry_limit} signal={signal}")
            if signal_age_bars > expiry_limit:
                _tg_debug(f"[SIGNAL EXPIRED] {symbol} age={signal_age_bars} bars > {expiry_limit} — SIGNAL KILLED")
                signal = 0
            
        # =====================================================
        # ENTRY LOGIC (MUST BE LAST STEP PER BAR)
        # This guarantees: EXIT and ENTRY cannot happen on same candle
        # =====================================================
        position = self.positions.get(symbol)

        if not position and signal != 0:

            if symbol in self._reentry_lock:
                locked_dir = self._reentry_lock[symbol]
                if signal == locked_dir:
                    locked_at = self._reentry_lock_ts.get(symbol, "unknown")
                    _tg_debug(f"[ENTRY BLOCKED — REENTRY LOCK] {symbol} dir={signal} locked_at={locked_at}")

            signal_ts = current_5m_row.name

            # latest_bar_ts = lltf_df.index[-1] -- LIVE ONLY
            # if signal_ts != latest_bar_ts:
            #     print(f"[ENTRY BLOCKED — NOT LATEST BAR] {symbol} signal_ts={signal_ts} latest={latest_bar_ts}")
            #     return {"state": "FLAT"}

            signal_id = (
                symbol + "|" +
                str(signal_ts) + "|" +
                str(signal) + "|" +
                str(external_row.name)
            )

            if signal_id in self._executed_signals:
                _tg_debug(f"[EXECUTED SIGNAL BLOCK] {symbol} signal_id={signal_id} — already executed, skipping")
                return {"state": "FLAT"}

            self._executed_signals.add(signal_id)
            _tg_debug(f"[EXECUTED SIGNAL REGISTERED] {symbol} signal_id={signal_id}")

            raw_entry   = float(external_row["open"])
            cost_mult   = self.TOTAL_COST_BPS / 10_000
            entry_price = raw_entry * (1 + cost_mult) if signal == 1 else raw_entry * (1 - cost_mult)

            if atr is None or atr <= 0 or np.isnan(atr):
                _tg_debug(f"[WARN ATR INVALID] {symbol} ts={current_ts} atr={atr}")
                atr = 0.000001

            atr = float(atr)

            _tg_debug(f"[DEBUG ENTRY] {symbol} signal={signal} atr={atr} ts={current_ts}")

            if atr <= 0:
                _tg_debug(f"[ENTRY BLOCKED] {symbol} @ {current_ts} — atr={atr}, skipping")
                return {"state": "FLAT"}

            new_pos = self._open(symbol, signal, entry_price, current_ts, atr)

            if new_pos:
                # initialize bar history with entry candle
                self._bar_history[symbol] = [{
                    "open": o,
                    "high": h,
                    "low":  l,
                    "close": c,
                    "ATR": atr,
                    "ts": current_ts,
                }]

                return {"state": "OPEN", "position": new_pos}

        return {"state": "FLAT"}

    # --------------------------------------------------
    # LIFECYCLE HELPERS  (exact mirrors of backtest)
    # --------------------------------------------------
    def _momentum_decay_exit(self, position: dict) -> bool:

        mfe_r = position.get("mfe_r", 0.0)
        if mfe_r < 2.0:
            return False  # only protect meaningful winners

        bars = position.get("bars_in_trade", 0)

        # -------------------------------------------------
        # Track bars since last new MFE
        # -------------------------------------------------
        last_mfe_bar = position.get("last_mfe_bar")

        # If we never stored it yet, initialise it now
        if last_mfe_bar is None:
            position["last_mfe_bar"] = bars
            return False

        bars_since_mfe = bars - last_mfe_bar

        # -------------------------------------------------
        # Stall detection thresholds
        # -------------------------------------------------
        # After 2R → allow ~1 hour of stall
        # After 3R → allow less stall (trend should accelerate)
        if mfe_r > 3.0:
            stall_limit = 4    # 20 minutes
        elif mfe_r > 2.0:
            stall_limit = 6    # 30 minutes
        else:
            stall_limit = 8    # 40 minutes (rarely used)

        if bars_since_mfe >= stall_limit:
            return True

        return False

    def _stealth_distribution_exit(self, window: pd.DataFrame, position: dict, side: int) -> bool:
        # Only look for distribution if we are in solid profit
        if position.get("mfe_r", 0.0) < 2.0:
            return False

        LOOKBACK = 6 # 30 mins of quiet consolidation
        if len(window) < LOOKBACK:
            return False

        recent = window.iloc[-LOOKBACK:]
        mfe_price = position.get("mfe_price")
        
        if not mfe_price:
            return False

        # Use the most recent valid ATR
        atr = recent["ATR"].iloc[-1]
        if pd.isna(atr) or atr <= 0:
            atr = window.iloc[0]["ATR"]

        # Stealth requires tight, quiet candles (compressing volatility)
        bodies = abs(recent["close"] - recent["open"])
        avg_body = bodies.mean()
        
        if avg_body > (atr * 0.6):
            return False # Too volatile to be "stealthy"

        if side == 1: # LONG
            # 1. Proximity: Hovering near the top (all recent closes within 1 ATR of MFE)
            near_high = (mfe_price - recent["close"]).max() < atr 
            # 2. Exhaustion: Failing to break the peak in the recent window
            failing_to_push = recent["high"].max() < mfe_price
            
            return near_high and failing_to_push

        else: # SHORT (Stealth Accumulation near lows)
            # 1. Proximity: Hovering near the bottom
            near_low = (recent["close"] - mfe_price).max() < atr
            # 2. Exhaustion: Failing to break the low
            failing_to_push = recent["low"].min() > mfe_price

            return near_low and failing_to_push

    def _opposite_impulse_exit(self, window: pd.DataFrame, side: int) -> bool:
        if len(window) < 1:
            return False

        last = window.iloc[-1]

        # Use best available ATR: rolling if warmed up, else entry bar ATR
        atr = last["ATR"]
        if pd.isna(atr) or atr <= 0:
            # fall back to entry bar ATR (first bar in window)
            atr = window.iloc[0]["ATR"]
        if pd.isna(atr) or atr <= 0:
            return False

        body = abs(last["close"] - last["open"])
        big_candle = body > atr * 1.2

        if side == 1:
            return big_candle and last["close"] < last["open"]
        else:
            return big_candle and last["close"] > last["open"]

    def _stop_pressure_exit(
        self, window: pd.DataFrame, stop_price: float, side: int, R: float
    ) -> bool:
        if len(window) < self.PRESSURE_BARS:
            return False
        recent = window.iloc[-self.PRESSURE_BARS:]
        if side == 1:
            dist = recent["close"] - stop_price
        else:
            dist = stop_price - recent["close"]
        return (dist <= R * self.STOP_PROXIMITY).all()

    def _no_follow_through_exit(self, mfe_r: float, bars_5m: int) -> bool:
        if bars_5m < self.VALIDATION_BARS:
            return False
        return mfe_r < self.NO_FOLLOW_MFE

    def _update_dynamic_stop(self, position, current_price, atr, side):
        mfe_r = position["mfe_r"]

        if mfe_r <= 0.75:
            return

        bars = position.get("bars_in_trade", 0)
        last_trail_bar = position.get("last_trail_bar", 0)
        if bars - last_trail_bar < 12:
            return
        position["last_trail_bar"] = bars

        entry = position["entry_price"]
        current_stop = position["stop_loss"]

        # Use tighter multiplier only after 2R secured
        atr_mult = self.ATR_AFTER_ONE_R if mfe_r > 2.0 else self.ATR_AFTER_HALF_R

        if side == 1:
            trail_candidate = current_price - atr * atr_mult
            if mfe_r > 2.0:
                trail_candidate = max(trail_candidate, entry)
            # only move stop if trail candidate is profitable
            if trail_candidate <= entry:
                return
            new_stop = max(current_stop, trail_candidate)
        else:
            trail_candidate = current_price + atr * atr_mult
            if mfe_r > 2.0:
                trail_candidate = min(trail_candidate, entry)
            # only move stop if trail candidate is profitable
            if trail_candidate >= entry:
                return
            new_stop = min(current_stop, trail_candidate)

        if new_stop != current_stop:
            position["stop_loss"] = new_stop
            position["trailing_activated"] = True

    # --------------------------------------------------
    # OPEN / CLOSE
    # --------------------------------------------------
    def _open(self, symbol, direction, price, ts, atr):
        stop = (
            price - self.ATR_MULT * atr if direction == 1
            else price + self.ATR_MULT * atr
        )

        position_value = self.POSITION_VALUE_USDT
        stop_dist = abs(price - stop)
        stop_pct = stop_dist / price if price > 0 else 0
        risk_usd = round(position_value * stop_pct, 4)
        quantity = round(position_value / price, 4) if price > 0 else 0

        position = {
            "symbol":        symbol,
            "trade_id":      TelegramNotifier.make_trade_id(symbol),
            "direction":     direction,
            "entry_price":   price,
            "entry_time":    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "entry_5m_ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "stop_loss":     stop,
            "initial_stop":  stop,
            "R": abs(price - stop),
            "state":         "OPEN",
            "mfe_r": 0.0,
            "pnl_r": 0.0,
            "MAE": 0.0,
            "MFE": 0.0,
            # ── new ──
            "risk_usd":      risk_usd,
            "quantity":      quantity,
            "position_value": position_value,
            "bars_in_trade": 0,
            "last_mfe_bar": 0,
            "last_trail_bar": 0,
        }

        self.positions[symbol] = position
        if self.persist:
            self._save()

        if self.USE_ACCOUNT_GATES:
            account_state.on_position_open()

        if self.notify:
            self.notifier.notify_open(
                symbol=symbol,
                direction=direction,
                entry_price=price,
                stop_loss=stop,
                timestamp=ts,
                trade_id=position["trade_id"],
                risk_usd=risk_usd,
                quantity=quantity,
                position_value=position_value
            )

        print(f"[OPEN EXECUTED] {symbol} dir={direction} price={price}")

        self._dirty = True

        return position.copy()

    def _close(self, symbol, price, ts, reason):
        if price is None:
            raise ValueError(f"[_close] fill_price is None for {symbol}, reason={reason}")
        
        pos = self.positions.pop(symbol)
        duration_bars = pos.get("bars_in_trade", 0)
        
        self._reentry_lock[symbol] = pos["direction"]
        self._reentry_lock_ts[symbol] = pd.Timestamp.now(tz="UTC")
        self._bar_history.pop(symbol, None)
        self._last_entry_ts.pop(symbol, None)
        
        direction = pos["direction"]
        entry     = pos["entry_price"]
        qty       = pos.get("quantity", 0.0)  # Grab the quantity

        if reason == "stop_loss":
            theoretical_stop = pos["stop_loss"]
            if price is not None:
                if pos["direction"] == 1:
                    fill_price = min(price, theoretical_stop)
                else:
                    fill_price = max(price, theoretical_stop)
            else:
                fill_price = theoretical_stop
        else:
            fill_price = price

        # 1. Price Difference
        price_diff = (fill_price - entry) if direction == 1 else (entry - fill_price)
        
        # 2. Actual Dollar PnL (Price Diff * Quantity)
        pnl_usd = price_diff * qty

        # 3. R-Multiple
        R = pos.get("R", None)
        pnl_r = price_diff / R if R and R > 0 else 0.0

        pos["state"] = "CLOSED"
        pos["exit"]  = {
            "price":   fill_price,
            "time":    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "reason":  reason,
            "pnl":     pnl_usd,  # <-- Pass the Dollar amount!
            "pnl_r":   round(pnl_r, 3),
            "duration": duration_bars, 
        }

        # Update account balance with actual dollars, not price cents
        if self.USE_ACCOUNT_GATES:
            account_state.on_position_close(pnl_usd)

        if self.notify:
            self.notifier.notify_close(
                symbol=symbol,
                direction=direction,
                exit_price=price,
                timestamp=ts,
                reason=reason,
                pnl_r=pnl_r,
                trade_id=pos["trade_id"],
                trailing_activated=pos.get("trailing_activated", False),
                risk_usd=pos.get("risk_usd", 0),
                entry_time=pos.get("entry_time"),
            )

        if self.persist:
            self._save()

        self._dirty = True

        print(
            f"[CLOSE EXECUTED] {symbol} | "
            f"Reason: {reason} | "
            f"Price: {fill_price} | "
            f"PnL $: {round(pnl_usd, 3)} | "
            f"PnL R: {round(pnl_r, 3)} | "
            f"Duration: {duration_bars} bars"
        )

        return pos.copy()

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self.positions
    
    def flush(self):
        """Persist state to disk only if something changed."""
        if not self.persist:
            return
        if not self._dirty:
            return

        self._save()
        self._dirty = False

    # --------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------
    def _load(self):
        # POSITIONS
        if os.path.exists(POSITIONS_FILE):
            try:
                with open(POSITIONS_FILE, "r") as f:
                    content = f.read().strip()
                self.positions = json.loads(content) if content else {}
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted positions file — starting fresh")
                self.positions = {}

        # BAR HISTORY
        if os.path.exists(BAR_HISTORY_FILE):
            try:
                with open(BAR_HISTORY_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                # re-cast ts strings back to pd.Timestamp (lost on JSON round-trip)
                for sym, bars in raw.items():
                    for bar in bars:
                        if "ts" in bar and isinstance(bar["ts"], str):
                            try:
                                bar["ts"] = pd.Timestamp(bar["ts"])
                            except Exception:
                                pass
                self._bar_history = raw
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted bar history file — starting fresh")
                self._bar_history = {}

        # ENTRY TIMESTAMPS
        if os.path.exists(ENTRY_TS_FILE):
            try:
                with open(ENTRY_TS_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                self._last_entry_ts = {k: pd.Timestamp(v) for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted entry timestamps — starting fresh")
                self._last_entry_ts = {}

        # EXECUTED SIGNALS
        if os.path.exists(EXECUTED_SIGNALS_FILE):
            try:
                with open(EXECUTED_SIGNALS_FILE, "r") as f:
                    content = f.read().strip()
                loaded = set(json.loads(content)) if content else set()
                # FIX: 2-hour window is enough to prevent duplicates within a session
                # 48 hours was creating phantom blocks that persisted across restarts
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
                kept = set()
                for s in loaded:
                    try:
                        parts = s.split("|")
                        if len(parts) < 2:
                            continue
                        ts = pd.Timestamp(parts[1])
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        if ts >= cutoff:
                            kept.add(s)
                    except Exception:
                        pass  # drop malformed entries silently
                self._executed_signals = kept
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted executed signals — starting fresh")
                self._executed_signals = set()

        # REENTRY LOCK
        if os.path.exists(REENTRY_LOCK_FILE):
            try:
                with open(REENTRY_LOCK_FILE, "r") as f:
                    content = f.read().strip()
                raw = json.loads(content) if content else {}
                # convert values to int, drop entries older than 48h
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
                self._reentry_lock = {}
                self._reentry_lock_ts = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        locked_at = pd.Timestamp(v.get("locked_at", "2000-01-01"), tz="UTC")
                        if locked_at >= cutoff:
                            self._reentry_lock[k] = int(v["direction"])
                            self._reentry_lock_ts[k] = locked_at
                    else:
                        # legacy format — drop it, no timestamp to validate
                        pass
            except (json.JSONDecodeError, ValueError):
                print(f"[WARN] Corrupted reentry lock — starting fresh")
                self._reentry_lock = {}
                self._reentry_lock_ts = {}

    def _save(self):
        if not self.persist:
            return
        with open(POSITIONS_FILE + ".tmp", "w") as f:
            json.dump(self.positions, f, indent=2, default=str)
        os.replace(POSITIONS_FILE + ".tmp", POSITIONS_FILE)

        with open(BAR_HISTORY_FILE + ".tmp", "w") as f:
            json.dump(self._bar_history, f, indent=2, default=str)
        os.replace(BAR_HISTORY_FILE + ".tmp", BAR_HISTORY_FILE)

        with open(ENTRY_TS_FILE + ".tmp", "w") as f:
            json.dump(
                {k: v.isoformat() for k, v in self._last_entry_ts.items()},
                f, indent=2
            )
        os.replace(ENTRY_TS_FILE + ".tmp", ENTRY_TS_FILE)

        # REENTRY LOCK
        lock_payload = {
            k: {
                "direction": v,
                "locked_at": self._reentry_lock_ts.get(k, pd.Timestamp.now(tz="UTC")).isoformat()
            }
            for k, v in self._reentry_lock.items()
        }
        with open(REENTRY_LOCK_FILE + ".tmp", "w") as f:
            json.dump(lock_payload, f, indent=2)
        os.replace(REENTRY_LOCK_FILE + ".tmp", REENTRY_LOCK_FILE)

        # EXECUTED SIGNALS
        with open(EXECUTED_SIGNALS_FILE + ".tmp", "w") as f:
            json.dump(list(self._executed_signals), f, indent=2)
        os.replace(EXECUTED_SIGNALS_FILE + ".tmp", EXECUTED_SIGNALS_FILE)