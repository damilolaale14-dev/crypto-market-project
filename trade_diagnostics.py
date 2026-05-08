import pandas as pd
import numpy as np


def diagnose_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trade diagnostics aligned with next-bar execution engine.

    Evaluates trades in 3 independent dimensions:

        1) Market Opportunity
        2) Execution Quality
        3) Outcome

    Does NOT assume duration implies time held.
    """

    if trades_df.empty:
        print("\nNo trades to diagnose.")
        return trades_df

    df = trades_df.copy()

    # ==========================================================
    # BASIC NORMALIZATION
    # ==========================================================
    df["side"] = df["side"].astype(int)
    df["direction"] = df["side"].map({1: "LONG", -1: "SHORT"})

    WAT = pd.Timedelta(hours=1)
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True) + WAT

    # Fixed risk assumption
    FIXED_RISK = 10.0
    if "risk_per_trade" in df.columns:
        FIXED_RISK = float(df["risk_per_trade"].iloc[0])

    # ==========================================================
    # CORE METRICS
    # ==========================================================
    df["R"] = df["pnl"] / FIXED_RISK

    # Structural duration (not time)
    df["duration_bars"] = df["exit_idx"] - df["entry_idx"]

    # Detect instant exits
    df["instant_exit"] = df["duration_bars"] == 0

    # ==========================================================
    # OPPORTUNITY ANALYSIS
    # ==========================================================
    df["opportunity_R"] = df["MFE"] / FIXED_RISK

    df["had_opportunity"] = df["opportunity_R"] >= 1.0
    df["signal_correct"] = df["MFE"] > abs(df["MAE"])

    # ==========================================================
    # EXECUTION QUALITY
    # ==========================================================
    df["capture_efficiency"] = np.where(
        df["MFE"] > 0,
        df["pnl"] / df["MFE"],
        0.0
    )

    df["execution_quality"] = np.select(
        [
            df["capture_efficiency"] >= 0.60,
            df["capture_efficiency"] >= 0.30,
            df["capture_efficiency"] >= 0.0,
        ],
        ["EXCELLENT", "GOOD", "POOR"],
        default="FAILED"
    )

    # ==========================================================
    # SIGNAL QUALITY (MARKET ONLY)
    # ==========================================================
    df["signal_quality"] = np.select(
        [
            df["had_opportunity"],
            df["signal_correct"],
        ],
        ["GOOD", "DECENT"],
        default="BAD"
    )

    # ==========================================================
    # EXIT TYPE CLASSIFICATION
    # ==========================================================
    df["exit_type"] = np.select(
        [
            df["exit_reason"] == "take_profit",
            df["exit_reason"] == "stop_loss",
            df["exit_reason"] == "break_even",
            df["exit_reason"] == "signal_flip",
            df["exit_reason"] == "end_of_data",
        ],
        [
            "TARGET HIT",
            "STOP HIT",
            "BREAKEVEN",
            "SIGNAL EXIT",
            "FORCED EXIT"
        ],
        default="OTHER"
    )

    # ==========================================================
    # PER TRADE REPORT
    # ==========================================================
    print("\n=== TRADE DIAGNOSTICS ===")

    for _, t in df.iterrows():

        instant_tag = " | INSTANT EXIT" if t["instant_exit"] else ""

        print(
            f"\n{t['direction']} | "
            f"{t['entry_time']} → {t['exit_time']}{instant_tag}"
        )

        print(
            f"PnL: {t['pnl']:.2f} | "
            f"R: {t['R']:.2f} | "
            f"MFE: {t['MFE']:.2f} | "
            f"MAE: {t['MAE']:.2f}"
        )

        print(
            f"Opportunity: {t['opportunity_R']:.2f}R | "
            f"Captured: {t['capture_efficiency'] * 100:.0f}%"
        )

        print(
            f"Signal: {t['signal_quality']} | "
            f"Execution: {t['execution_quality']}"
        )

        print(
            f"Exit: {t['exit_type']} | "
            f"Duration: {t['duration_bars']} bars"
        )

    # ==========================================================
    # SUMMARY STATISTICS
    # ==========================================================
    print("\n=== SUMMARY ===")

    total = len(df)
    wins = (df["pnl"] > 0).sum()

    print(f"Total trades: {total}")
    print(f"Win rate: {(wins / total) * 100:.2f}%")

    print("\n--- Outcome ---")
    print(f"Avg R: {df['R'].mean():.2f}")
    print(f"Total PnL: {df['pnl'].sum():.2f}")

    print("\n--- Opportunity ---")
    print(f"Avg opportunity (R): {df['opportunity_R'].mean():.2f}")
    print(f"Good signals: {(df['signal_quality'] == 'GOOD').mean() * 100:.2f}%")

    print("\n--- Execution ---")
    print(
        f"Good execution: "
        f"{(df['execution_quality'].isin(['EXCELLENT','GOOD'])).mean() * 100:.2f}%"
    )

    print(f"Avg capture: {df['capture_efficiency'].mean():.2f}")

    print("\n--- Structure ---")
    print(f"Instant exits: {(df['instant_exit']).mean()*100:.2f}%")

    return df