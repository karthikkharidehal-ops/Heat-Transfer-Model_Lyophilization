"""
Debug script to trace ingestion step-by-step
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import functions from ingest
from ingest import load_raw, build_timestamp, normalize_units, _first_column, _numeric_series

def debug_ingestion(csv_path: str):
    print("=" * 60)
    print("STEP 1: Loading raw CSV")
    print("=" * 60)
    
    df = load_raw(Path(csv_path))
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns found: {list(df.columns)}")
    print()
    
    # Check for required columns
    print("=" * 60)
    print("STEP 2: Checking required columns")
    print("=" * 60)
    required = ["Date", "Time", "Cycle", "Phase", "Step"]
    for col in required:
        status = "✓ FOUND" if col in df.columns else "✗ MISSING"
        print(f"{status}: {col}")
    print()
    
    # Check Phase values
    print("=" * 60)
    print("STEP 3: Checking Phase column values")
    print("=" * 60)
    if "Phase" in df.columns:
        phase_values = df["Phase"].unique()
        print(f"Unique Phase values: {sorted(phase_values)}")
        phase_6_count = (df["Phase"] == 6).sum()
        print(f"Rows with Phase=6 (Primary Drying): {phase_6_count}")
    print()
    
    # Check Cycle and Step distribution
    print("=" * 60)
    print("STEP 4: Checking Cycle/Step distribution")
    print("=" * 60)
    if "Cycle" in df.columns and "Step" in df.columns:
        cycle_step_counts = df.groupby(["Cycle", "Phase"]).size()
        print("Rows per (Cycle, Phase):")
        print(cycle_step_counts)
    print()
    
    print("=" * 60)
    print("STEP 5: Building timestamp")
    print("=" * 60)
    try:
        df_ts = build_timestamp(df.copy())
        print(f"✓ Timestamp built successfully")
        print(f"  Valid timestamps: {df_ts['timestamp'].notna().sum()}")
        print(f"  Dropped rows: {len(df) - len(df_ts)}")
        df = df_ts
    except Exception as e:
        print(f"✗ Timestamp build failed: {e}")
        return
    print()
    
    print("=" * 60)
    print("STEP 6: Normalizing units - Column matching test")
    print("=" * 60)
    
    # Test each column mapping
    test_mappings = {
        "Pirani": ["Pirani", "Pirani (mTorr)"],
        "VACUUM": ["VACUUM", "Vacuum", "VACUUM (mTorr)"],
        "VacSetpt": ["VacSetpt", "Vac Setpt", "Vacuum Setpt"],
        "PRESSURE": ["PRESSURE", "Pressure", "Capman", "PRESSURE (mbar)"],
        "ShelfTemp": ["ShelfTemp", "Shelf Temp", "Shelf Temp (°C)", "Shelf Temp (C)"],
        "ProdAvg": ["ProdAvg", "Prod Avg", "Prod Avg (°C)", "Prod Avg (C)"],
        "TP01": ["TP01", "TP01 (°C)", "TP01 (C)"],
        "TP02": ["TP02", "TP02 (°C)", "TP02 (C)"],
        "TP03": ["TP03", "TP03 (°C)", "TP03 (C)"],
        "TP04": ["TP04", "TP04 (°C)", "TP04 (C)"],
    }
    
    for name, candidates in test_mappings.items():
        found_col = _first_column(df, candidates)
        if found_col:
            print(f"✓ {name}: matched to '{found_col}'")
            # Sample some values
            sample = df[found_col].dropna().head(3).tolist()
            print(f"    Sample values: {sample}")
        else:
            print(f"✗ {name}: NO MATCH for candidates {candidates}")
    print()
    
    print("=" * 60)
    print("STEP 7: Running normalize_units")
    print("=" * 60)
    try:
        df_norm = normalize_units(df.copy())
        print(f"✓ Units normalized")
        
        # Check which output columns were created
        output_cols = ["pirani_pa", "vacuum_pa", "pressure_pa", "shelf_temp_k", 
                       "prod_avg_k", "tp01_k", "tp02_k", "tp03_k", "tp04_k"]
        
        for col in output_cols:
            if col in df_norm.columns:
                non_null = df_norm[col].notna().sum()
                print(f"✓ {col}: {non_null} non-null values")
                if non_null > 0:
                    sample = df_norm[col].dropna().head(3).tolist()
                    print(f"    Sample: {sample}")
            else:
                print(f"✗ {col}: COLUMN NOT CREATED")
    except Exception as e:
        print(f"✗ normalize_units failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print("=" * 60)
    print("STEP 8: Filtering for Primary Drying (Phase=6)")
    print("=" * 60)
    
    df_pd = df_norm[df_norm["Phase"] == 6].copy()
    print(f"Rows with Phase=6: {len(df_pd)}")
    
    if len(df_pd) > 0:
        # Check for valid data in primary drying rows
        required_for_fit = ["pressure_pa", "prod_avg_k", "shelf_temp_k"]
        print("\nChecking required columns in Phase=6 rows:")
        for col in required_for_fit:
            if col in df_pd.columns:
                non_null = df_pd[col].notna().sum()
                print(f"  {col}: {non_null}/{len(df_pd)} non-null")
            else:
                print(f"  {col}: COLUMN MISSING")
        
        # Check how many rows have ALL required data
        valid_mask = df_pd[required_for_fit].notna().all(axis=1)
        print(f"\nRows with ALL required data: {valid_mask.sum()}/{len(df_pd)}")
        
        if valid_mask.sum() < 5:
            print("\n⚠ WARNING: Less than 5 valid rows for fitting!")
            print("This will cause the 'No usable segments' error.")
            
            # Show why rows are invalid
            print("\nFirst 5 rows with null analysis:")
            print(df_pd[required_for_fit].head(5))
    
    print()
    print("=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_ingest.py <your_file.csv>")
        sys.exit(1)
    
    debug_ingestion(sys.argv[1])
