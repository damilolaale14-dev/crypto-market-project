import pandas as pd
from datetime import timezone
from execution.hourly_runner import map_ltf_to_htf
from indicators.indicators import generate_signal, atr_ema
from strategy.lifecycle import PositionManager

df_1h = pd.read_parquet("data/cache/ETCUSDT_1h.parquet")
df_4h = pd.read_parquet("data/cache/ETCUSDT_4h.parquet")
df_5m = pd.read_parquet("data/cache/ETCUSDT_5m.parquet")

for d in (df_1h, df_4h, df_5m):
    d.index = pd.to_datetime(d.index, utc=True)

def simulate_with_early_entry(label, now_utc, cursor, pm):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    now_hour = now_utc.replace(minute=0, second=0, microsecond=0)

    df_sim  = df_1h[df_1h.index <= now_hour - pd.Timedelta(hours=1)].copy()
    htf_sim = df_4h[df_4h.index <= now_hour - pd.Timedelta(hours=1)].copy()

    minutes_floored      = (now_utc.minute // 5) * 5
    current_5m_boundary  = now_utc.replace(minute=minutes_floored, second=0, microsecond=0)
    current_5m_boundary  = pd.Timestamp(current_5m_boundary).tz_convert("UTC")

    # mirror update_symbol fetch — includes forming bar
    lltf_df = df_5m[df_5m.index <= now_utc].copy()

    # ── EARLY ENTRY LOGIC (the proposed change) ──────────────
    now_utc_ts       = pd.Timestamp(now_utc).tz_convert("UTC")
    seconds_elapsed  = (now_utc_ts - current_5m_boundary).total_seconds()
    boundary_in_data = current_5m_boundary in lltf_df.index

    print(f"current_5m_boundary: {current_5m_boundary}")
    print(f"seconds_elapsed:     {seconds_elapsed:.0f}")
    print(f"boundary_in_data:    {boundary_in_data}")

    boundary_is_hour_open = current_5m_boundary.minute == 0

    # generate signals first so we can check signal_on_prev_1h
    df_sig = generate_signal(df_sim.copy(), htf_sim.copy(), live=True)

    signal_on_prev_1h = len(df_sig) > 0 and int(df_sig["final_signal"].iloc[-1]) != 0

    if (seconds_elapsed >= 30
            and boundary_in_data
            and boundary_is_hour_open
            and signal_on_prev_1h):
        lltf_df = lltf_df[lltf_df.index <= current_5m_boundary].copy()
        ...
    else:
        lltf_df = lltf_df[lltf_df.index < current_5m_boundary].copy()
        ...

    print(f"lltf last bar:       {lltf_df.index[-1]}")

    lltf_df = lltf_df[lltf_df.index >= df_sig.index[0]].copy()
    lltf_df = map_ltf_to_htf(lltf_df, df_sig)
    lltf_df["final_signal"] = df_sig["final_signal"].reindex(lltf_df.index, method="ffill")

    signal_bar_ts  = df_sig.index[-1]
    signal_bar_end = signal_bar_ts + pd.Timedelta(hours=1)
    within = (lltf_df.index >= signal_bar_ts) & (lltf_df.index < signal_bar_end)
    lltf_df.loc[within, "final_signal"] = 0

    lltf_df["ATR"]    = df_sig["ATR"].reindex(lltf_df.index, method="ffill")
    lltf_df["ATR_5M"] = atr_ema(lltf_df, period=14)

    lltf_frozen = lltf_df.dropna(subset=["ltf_index"]).copy()
    lltf_frozen["ltf_index"] = lltf_frozen["ltf_index"].astype(int)

    new_bars = lltf_frozen if cursor is None else lltf_frozen[lltf_frozen.index > cursor]
    non_zero = (new_bars["final_signal"] != 0).sum() if not new_bars.empty else 0

    print(f"new_bars:            {len(new_bars)}")
    print(f"non_zero:            {non_zero}")

    window = lltf_frozen[
        (lltf_frozen.index >= pd.Timestamp("2026-05-10 16:50:00", tz="UTC")) &
        (lltf_frozen.index <= pd.Timestamp("2026-05-10 17:10:00", tz="UTC"))
    ]
    if not window.empty:
        print(f"\n5m bars 16:50–17:10:")
        print(window[["final_signal", "ltf_index"]].to_string())

    if non_zero > 0:
        print(f"\npm.update results:")
        for ts, row_5m in new_bars[new_bars["final_signal"] != 0].iterrows():
            bar_signal = int(row_5m["final_signal"])
            ltf_row    = df_sig.iloc[int(row_5m["ltf_index"])]
            result     = pm.update(
                df=df_sig,
                symbol="ETCUSDT",
                lltf_df=lltf_frozen,
                external_signal=bar_signal,
                external_row=ltf_row,
                current_5m_row=row_5m
            )
            state = result.get("state") if isinstance(result, dict) else result
            print(
                f"  ts={ts} | signal={bar_signal} | "
                f"reentry_lock={pm._reentry_lock.get('ETCUSDT')} | "
                f"state={state}"
            )
            if state == "OPEN":
                print(f"  *** ENTRY FIRED IN {label} ***")

    new_cursor = new_bars.index[-1] if not new_bars.empty else cursor
    print(f"\ncursor after: {new_cursor}")
    return new_cursor


pm = PositionManager(persist=False, notify=False)

cursor_1 = pd.Timestamp("2026-05-10 15:55:00", tz="UTC")

cursor_2 = simulate_with_early_entry(
    label   = "RUN 1 — 17:00:59 UTC",
    now_utc = pd.Timestamp("2026-05-10 17:00:59", tz="UTC"),
    cursor  = cursor_1,
    pm      = pm
)

cursor_3 = simulate_with_early_entry(
    label   = "RUN 2 — 17:06:17 UTC",
    now_utc = pd.Timestamp("2026-05-10 17:06:17", tz="UTC"),
    cursor  = cursor_2,
    pm      = pm
)

cursor_4 = simulate_with_early_entry(
    label   = "RUN 3 — 17:11:00 UTC (verify 17:05 bar excluded)",
    now_utc = pd.Timestamp("2026-05-10 17:11:00", tz="UTC"),
    cursor  = cursor_3,
    pm      = pm
)

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")

if "ETCUSDT" in pm.positions:
    entry_time = pm.positions["ETCUSDT"]["entry_time"]
    print(f"position open | entry_time={entry_time}")

    # check which run actually fired it
    # entry_time=17:00 could mean Run 1 OR Run 2 so we check cursor_2
    if cursor_2 == pd.Timestamp("2026-05-10 17:00:00", tz="UTC"):
        print("PASS — EARLY ENTRY WORKED | fired in Run 1 at 17:00:59")
    else:
        print("PASS — but fired in Run 2, not Run 1 | early entry did not trigger")
else:
    print("FAIL — no position opened in any run")