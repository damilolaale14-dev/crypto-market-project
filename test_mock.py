# test_freq_guard.py
import pandas as pd
import numpy as np
from execution.hourly_runner import map_ltf_to_htf


def make_1h_df(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 1000.0
    }, index=idx)


def make_4h_df_with_gap_in_first_10(n=100):
    """4H dataframe where bar 3 is missing — infer_freq returns None."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    idx = idx.delete(3)  # remove bar index 3 → gap in first 10
    return pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 1000.0
    }, index=idx)


def make_4h_df_clean(n=100):
    """4H dataframe with no gaps — infer_freq returns '4h'."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 1000.0
    }, index=idx)


def run_test(label, ltf_df, bad_df):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    # What does infer_freq think?
    inferred = pd.infer_freq(bad_df.index[:10])
    print(f"infer_freq on first 10 bars: {repr(inferred)}")
    print(f"Actual bar spacing (first 5): {bad_df.index[:5].to_series().diff().dropna().unique()}")

    # Does it pass the guard?
    allowed = (None, "h", "1h", "H", "1H", "60min", "60T", "T60")
    passes_guard = inferred in allowed
    print(f"Passes frequency guard: {passes_guard}")

    if passes_guard:
        print(f"⚠️  Guard did NOT catch this — running map_ltf_to_htf...")
        try:
            # Make a minimal 5m dataframe
            lltf_idx = pd.date_range("2024-01-01", periods=500, freq="5min", tz="UTC")
            lltf_df = pd.DataFrame({
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.5, "volume": 500.0,
                "final_signal": 0
            }, index=lltf_idx)

            # Pass the BAD df as ltf_df (simulating the bug)
            result = map_ltf_to_htf(lltf_df, bad_df)

            # Now pass the CORRECT 1H df for comparison
            correct = map_ltf_to_htf(lltf_df, ltf_df)

            # Compare ltf_index assignments
            diff = (result["ltf_index"] != correct["ltf_index"]).sum()
            total = len(result)
            print(f"ltf_index mismatches vs correct 1H mapping: {diff} / {total} bars ({diff/total*100:.1f}%)")

            if diff > 0:
                print(f"🔴 RESULT: {diff} bars mapped to WRONG parent candle")
                print(f"   First 5 correct ltf_index: {correct['ltf_index'].values[:5]}")
                print(f"   First 5 corrupt ltf_index: {result['ltf_index'].values[:5]}")
            else:
                print(f"🟢 RESULT: No corruption detected (bug may not apply here)")

        except ValueError as e:
            print(f"🟢 Guard raised ValueError as expected: {e}")
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
    else:
        print(f"🟢 Guard correctly caught wrong frequency")


if __name__ == "__main__":
    ltf_1h = make_1h_df(200)

    # Case 1: clean 4H — should be caught by guard
    run_test(
        label="Clean 4H passed as 1H — should be caught",
        ltf_df=ltf_1h,
        bad_df=make_4h_df_clean(100)
    )

    # Case 2: 4H with gap in first 10 bars — infer_freq returns None
    run_test(
        label="4H with gap in first 10 bars — infer_freq=None, guard may pass",
        ltf_df=ltf_1h,
        bad_df=make_4h_df_with_gap_in_first_10(100)
    )

    # Case 3: sanity check — clean 1H should pass and produce correct mapping
    print(f"\n{'='*60}")
    print(f"SANITY: Clean 1H — should pass and map correctly")
    print(f"{'='*60}")
    inferred = pd.infer_freq(ltf_1h.index[:10])
    print(f"infer_freq: {repr(inferred)}")
    lltf_idx = pd.date_range("2024-01-01", periods=500, freq="5min", tz="UTC")
    lltf_df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 500.0,
        "final_signal": 0
    }, index=lltf_idx)
    result = map_ltf_to_htf(lltf_df, ltf_1h)
    print(f"Mapping succeeded, ltf_index range: {result['ltf_index'].min()} → {result['ltf_index'].max()}")
    print(f"🟢 Sanity check passed")