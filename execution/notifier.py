import requests
import os
import uuid
from typing import Optional


class TelegramNotifier:
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token or not self.chat_id:
            raise ValueError("Telegram bot token and chat ID must be set.")

        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def notify_open(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        stop_loss: float,
        timestamp,
        trade_id: str,
        risk_usd: float,
        quantity: float,
        position_value: float,
    ) -> None:
        dir_text   = "BUY" if direction == 1 else "SELL"
        trade_id   = trade_id or self._make_id(symbol)
        position_value = float(position_value or 0)
        quantity = float(quantity or 0)
        risk_usd = float(risk_usd or 0)
        stop_dist  = abs(entry_price - stop_loss)
        stop_pct   = (stop_dist / entry_price * 100) if entry_price else 0

        ts_str = self._fmt_ts(timestamp)

        msg = (
            f"🟢 *TRADE READY*\n"
            f"ID: `{trade_id}`\n"
            f"Symbol: `{symbol}`\n"
            f"Side: `{dir_text}`\n"
            f"Entry Time: `{ts_str}`\n"
            f"Entry Price: `{entry_price:.6f}`\n"
            f"Stop Loss: `{stop_loss:.6f}`\n"
            f"Stop Distance: `{stop_dist:.6f}` (`{stop_pct:.2f}%`)\n"
            f"Position Value: `${position_value:.0f}`\n"
            f"Quantity: `{quantity}`\n"
            f"Risk: `${risk_usd:.3f}`\n"
            f"\n*EXECUTE ON EXCHANGE* 👇\n"
            f"Market Order: `{symbol} {quantity}`\n"
            f"Stop Order: `{stop_loss:.6f}`"
        )
        self._send(msg, parse_mode="Markdown")

    def notify_close(
        self,
        symbol:             str,
        direction:          int,
        exit_price:         float,
        timestamp,
        reason:             str,
        pnl_r:              float,
        trade_id:           Optional[str] = None,
        trailing_activated: bool = False,
        risk_usd:           float = 0,
        entry_time:         Optional[str] = None,
    ) -> None:
        dir_text  = "LONG" if direction == 1 else "SHORT"
        trade_id  = trade_id or "unknown"
        risk_usd  = float(risk_usd or 0)
        pnl_usd   = pnl_r * risk_usd

        exit_ts_str  = self._fmt_ts(timestamp)
        entry_ts_str = entry_time or "unknown"

        msg = (
            f"🔴 *TRADE CLOSED*\n"
            f"ID: `{trade_id}`\n"
            f"Symbol: `{symbol}`\n"
            f"Direction: `{dir_text}`\n"
            f"Entry Time: `{entry_ts_str}`\n"
            f"Exit Time: `{exit_ts_str}`\n"
            f"Exit Price: `{exit_price:.6f}`\n"
            f"Reason: `{reason}`\n"
            f"Result:\n"
            f"PnL: `{pnl_r:+.2f}R`\n"
            f"PnL: `${pnl_usd:+.3f}`\n"
            f"Trailing Activated: `{trailing_activated}`"
        )
        self._send(msg, parse_mode="Markdown")

    def send_text(self, message: str) -> None:
        self._send(message, parse_mode="MarkdownV2")

    def debug(self, message: str) -> None:
        """Send a plain-text diagnostic message — no MarkdownV2 escaping."""
        self._send(message, parse_mode=None)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def make_trade_id(symbol: str) -> str:
        return f"{symbol}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _make_id(symbol: str) -> str:
        return TelegramNotifier.make_trade_id(symbol)

    @staticmethod
    def _dir_label(direction: int) -> str:
        return {1: "LONG 📈", -1: "SHORT 📉"}.get(direction, "FLAT ⬜")

    @staticmethod
    def _fmt_ts(ts) -> str:
        import pandas as pd
        if ts is None:
            return "unknown"
        try:
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            else:
                t = t.tz_convert("UTC")
            wat = t + pd.Timedelta(hours=1)
            return wat.strftime("%Y-%m-%d %H:%M WAT")
        except Exception:
            return str(ts)

    def _escape_md(self, text: str) -> str:
        """
        Escape Telegram MarkdownV2 special characters.
        """
        escape_chars = r"_*[]()~`>#+-=|{}.!"
        for ch in escape_chars:
            text = text.replace(ch, f"\\{ch}")
        return text

    def _send(self, message: str, parse_mode: Optional[str] = "MarkdownV2") -> None:
        if parse_mode == "MarkdownV2":
            message = self._escape_md(message)

        payload: dict = {
            "chat_id": self.chat_id,
            "text": message,
        }

        # only include parse_mode if set — omitting it means Telegram renders plain text
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"[TELEGRAM SEND FAILED] {e} | message={message[:100]}")
            try:
                fallback = {
                    "chat_id": self.chat_id,
                    "text": f"[SEND FAILED] {str(e)[:200]}\nOriginal: {message[:100]}"
                }
                requests.post(self.api_url, json=fallback, timeout=10)
            except Exception:
                pass