"""
LyoXL historian ingestion and cleaning.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

MTORR_TO_PA = 0.13332
MBAR_TO_PA = 100.0
TP_COLUMNS = ["TP01", "TP02", "TP03", "TP04", "TP1", "TP2", "TP3", "TP4"]

def _first_column(df, candidates, required=False):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"Could not find any of these required columns: {candidates}. "
            f"Available columns: {sorted(df.columns)}"
        )
    return None

def _numeric_series(df, candidates, required=False):
    col = _first_column(df, candidates, required=required)
    if col is None:
        return pd.Series(np.nan, index=df.index)

    # Clean up strings: strip whitespace, remove units if appended, handle commas
    s = df[col].astype(str).str.strip()
    
    # Remove common unit suffixes if they accidentally ended up inside the cell
    s = s.str.replace(r'\s*(mTorr|mbar|Pa|C|°C|K|%)\s*$', '', regex=True)
    
    # Handle European decimals (1,50 -> 1.50) if there's exactly one comma and no dot
    s = s.apply(lambda x: x.replace(',', '.') if isinstance(x, str) and x.count(',') == 1 and '.' not in x else x)
    
    numeric = pd.to_numeric(s, errors="coerce")
    
    if numeric.notna().sum() == 0 and df[col].notna().sum() > 0:
        print(f"[warn] Column '{col}' matched but contains non-numeric data.", file=sys.stderr)
        print(f"[warn] First few raw values in '{col}': {df[col].dropna().head(5).tolist()}", file=sys.stderr)
        
    return numeric

def load_raw(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    # sep=None with engine='python' auto-detects tabs, commas, semicolons
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    df.columns = [str(c).strip() for c in df.columns]
    return df

def build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = _first_column(df, ["Date", "date", "DATE"], required=True)
    time_col = _first_column(df, ["Time", "time", "TIME"], required=True)

    date_str = df[date_col].astype(str).str.strip().str.strip('"')
    time_str = df[time_col].astype(str).str.strip().str.strip('"')
    
    # --- THE FIX ---
    # The historian exports milliseconds with a colon instead of a dot.
    # Example: '21:13:43:49' -> '21:13:43.49'
    time_str = time_str.str.replace(r'(\d{2}:\d{2}:\d{2}):(\d{1,3})$', r'\1.\2', regex=True)
    
    combined = date_str + " " + time_str

    # Parse with dayfirst=True for DD-MM-YYYY format
    ts = pd.to_datetime(combined, errors="coerce", dayfirst=True)
    
    # Fallback if it still fails
    if ts.isna().all():
        ts = pd.to_datetime(combined, errors="coerce")

    df["timestamp"] = ts
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        print(f"[warn] {n_bad} rows had unparseable Date/Time and will be dropped", file=sys.stderr)
        if n_bad == len(df):
            print("[fatal] ALL rows were dropped due to Date/Time parsing failure!", file=sys.stderr)
            
    return df.dropna(subset=["timestamp"])

def normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # --- Pressure Mapping (Single Source of Truth) ---
    # We rely solely on the 'VACUUM' column (high-sensitivity mTorr gauge).
    # The 'PRESSURE' column (mbar) is ignored due to low sensitivity (13 mbar floor).
    
    # Extract VACUUM (mTorr) and convert to Pa
    vacuum_series = _numeric_series(df, ["VACUUM", "Vacuum"])
    if vacuum_series.isna().all():
        raise ValueError("Required column 'VACUUM' not found. Cannot proceed without high-sensitivity pressure data.")
    
    # Convert mTorr to Pa: 1 mTorr = 0.133322 Pa
    df["capman_pa"] = vacuum_series * 0.133322
    
    # Aliases: pressure_pa and vacuum_pa both point to the same high-sensitivity data
    df["pressure_pa"] = df["capman_pa"]
    df["vacuum_pa"] = df["capman_pa"]
    
    # Pirani gauge (for convergence detection only)
    df["pirani_pa"] = _numeric_series(df, ["Pirani", "Pirani (mTorr)"]) * 0.133322
    
    # Vacuum setpoint (optional, for reference)
    df["vac_setpt_pa"] = _numeric_series(df, ["VacSetpt", "Vac Setpt", "Vacuum Setpt"]) * 0.133322

    # Temperature columns (all in C, convert to K)
    shelf_temp_c = _numeric_series(df, ["ShelfTemp", "Shelf Temp", "TS_x"])
    df["shelf_temp_k"] = shelf_temp_c + 273.15

    shelf_setpt_c = _numeric_series(df, ["ShelfSetpt", "Shelf Setpt"])
    df["shelf_setpt_k"] = shelf_setpt_c + 273.15

    cond_temp_c = _numeric_series(df, ["CondTemp", "Cond Temp"])
    df["cond_temp_k"] = cond_temp_c + 273.15

    prod_avg_c = _numeric_series(df, ["ProdAvg", "Prod Avg"])
    df["prod_avg_k"] = prod_avg_c + 273.15

    for tp in TP_COLUMNS:
        tp_c = _numeric_series(df, [tp])
        df[f"{tp.lower()}_k"] = tp_c + 273.15

    tp_k_cols = [f"{tp.lower()}_k" for tp in TP_COLUMNS if f"{tp.lower()}_k" in df.columns]
    if tp_k_cols:
        tp_mean_k = df[tp_k_cols].mean(axis=1)
        df["prod_avg_k"] = df["prod_avg_k"].fillna(tp_mean_k)
        
    # Also compute prod_min_k: the lowest reading among all TP probes at each timestep
    # This approximates the ice-front temperature when probes are fully immersed,
    # as the coldest probe is typically closest to the sublimation front.
    if tp_k_cols:
        df["prod_min_k"] = df[tp_k_cols].min(axis=1)

    # --- SANITY CHECK: Temperature gradient analysis ---
    # During PRIMARY DRYING: Shelf is heated, product is cold due to sublimation cooling
    #   Expected: T_shelf > T_product (typically by 5-40C depending on conditions)
    # During FREEZING/ANNEALING: Shelf is cooled, product cools down
    #   Expected: T_product > T_shelf (transiently, until equilibrium)
    # 
    # Since we don't know which phase each row represents, we only flag extreme cases
    # that suggest unit mismatches (C vs K) or sensor failures.
    if "shelf_temp_k" in df.columns and "prod_avg_k" in df.columns:
        delta_t = df["shelf_temp_k"] - df["prod_avg_k"]  # Positive means Shelf > Prod
        
        # Check for extreme differences (>80K suggests one temp is in wrong units)
        extreme_mask = np.abs(delta_t) > 80
        n_extreme = extreme_mask.sum()
        if n_extreme > 0:
            print(f"[warn] Extreme temperature differences detected: {n_extreme} rows.", file=sys.stderr)
            print(f"[warn]   Delta T (Shelf-Prod) range: {delta_t.min():.2f}K to {delta_t.max():.2f}K", file=sys.stderr)
            print(f"[warn]   This may indicate unit mismatches (C vs K) or sensor errors.", file=sys.stderr)
        
        # Report the general trend for user verification
        pct_shelf_warmer = 100.0 * (delta_t > 0).sum() / len(df)
        avg_delta = delta_t.mean()
        print(f"[info] Temperature trend: Shelf > Product in {pct_shelf_warmer:.1f}% of rows (avg Delta T = {avg_delta:.2f}K)", file=sys.stderr)
        
        # Note: In some systems, radiative heat gain can cause T_prod > T_shelf even during primary drying.
        # We no longer flag this as an error, but still report it for awareness.
        if pct_shelf_warmer < 50:
            print(f"[info]   Note: Product is warmer than Shelf in most rows.", file=sys.stderr)
            print(f"[info]   This can occur due to radiative heating from chamber walls/doors.", file=sys.stderr)
            print(f"[info]   The Pikal model will proceed, but verify data quality if fit fails.", file=sys.stderr)

    return df

def flag_primary_drying_end(df: pd.DataFrame, tol_pa: float = 2.0, sustain_minutes: float = 20.0) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    if "Cycle" not in df.columns:
        df["Cycle"] = 0
        
    df["converged"] = False
    df["pd_end_candidate"] = False
    
    if "pirani_pa" not in df.columns or "capman_pa" not in df.columns:
        return df

    valid = df["pirani_pa"].notna() & df["capman_pa"].notna()
    df.loc[valid, "converged"] = (df.loc[valid, "pirani_pa"] - df.loc[valid, "capman_pa"]).abs() < tol_pa

    for cycle_id, g in df.groupby("Cycle"):
        idx = g.index.to_numpy()
        conv = g["converged"].to_numpy(dtype=bool)
        ts = g["timestamp"].to_numpy()

        for i in range(len(conv)):
            if not conv[i]: continue
            window_end = ts[i] + np.timedelta64(int(sustain_minutes), "m")
            j = i
            while j < len(conv) and ts[j] <= window_end:
                if not conv[j]: break
                j += 1
            else:
                df.loc[idx[i], "pd_end_candidate"] = True
                break
    return df

def segment_cycles(df: pd.DataFrame) -> dict:
    df = df.copy()
    if "Cycle" not in df.columns:
        df["Cycle"] = 0
    cycles = {}
    for cycle_id, g in df.groupby("Cycle"):
        cycles[str(cycle_id)] = g.sort_values("timestamp").reset_index(drop=True)
    return cycles

def summarize(cycles: dict) -> pd.DataFrame:
    rows = []
    for cid, g in cycles.items():
        if g.empty: continue
        pd_end_rows = g[g["pd_end_candidate"]] if "pd_end_candidate" in g.columns else g.iloc[0:0]
        recipe = g["Recipe"].iloc[0] if "Recipe" in g.columns else None
        
        shelf_max_c = np.nan
        if "shelf_temp_k" in g.columns and g["shelf_temp_k"].notna().any():
            shelf_max_c = g["shelf_temp_k"].max() - 273.15
            
        prod_min_c = np.nan
        if "prod_avg_k" in g.columns and g["prod_avg_k"].notna().any():
            prod_min_c = g["prod_avg_k"].min() - 273.15

        rows.append({
            "cycle": cid, "recipe": recipe,
            "start": g["timestamp"].iloc[0], "end": g["timestamp"].iloc[-1],
            "n_rows": len(g),
            "pd_end_detected_at": pd_end_rows["timestamp"].iloc[0] if len(pd_end_rows) else None,
            "shelf_temp_max_c": shelf_max_c, "prod_avg_min_c": prod_min_c,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clean_cycles.parquet"))
    ap.add_argument("--summary-out", type=Path, default=Path("cycle_summary.csv"))
    args = ap.parse_args()

    df = load_raw(args.input_csv)
    df = build_timestamp(df)
    df = normalize_units(df)
    df = flag_primary_drying_end(df)

    cycles = segment_cycles(df)
    print(f"Parsed {len(cycles)} cycle(s): {list(cycles.keys())}")

    out_path = Path(args.out)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        try:
            df.to_parquet(out_path, index=False)
        except ImportError:
            print("[warn] pyarrow missing, falling back to CSV.", file=sys.stderr)
            df.to_csv(out_path.with_suffix(".csv"), index=False)

    summary = summarize(cycles)
    summary.to_csv(args.summary_out, index=False)
    print(f"Wrote cleaned data -> {out_path}")
    print(f"Wrote per-cycle summary -> {args.summary_out}")

if __name__ == "__main__":
    main()