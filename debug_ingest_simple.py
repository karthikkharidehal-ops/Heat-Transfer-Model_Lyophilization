#!/usr/bin/env python3
"""Debug script to trace ingestion step-by-step."""
import sys
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_ingest_simple.py <your_file.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("=" * 60)
    print("STEP 1: Load raw CSV")
    print("=" * 60)
    df = pd.read_csv(csv_path)
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    print()
    
    print("=" * 60)
    print("STEP 2: Check for Phase column and Phase=6 rows")
    print("=" * 60)
    if "Phase" in df.columns:
        phase_counts = df["Phase"].value_counts().sort_index()
        print(f"Phase value counts:\n{phase_counts}")
        phase_6_count = (df["Phase"] == 6).sum()
        print(f"Rows with Phase=6: {phase_6_count}")
    else:
        print("ERROR: 'Phase' column not found!")
    print()
    
    print("=" * 60)
    print("STEP 3: Check required columns exist")
    print("=" * 60)
    required = ["Cycle", "Step", "Phase", "Pirani", "VACUUM", "VacSetpt", 
                "ShelfTemp", "ShelfSetpt", "CondTemp", "ProdAvg", 
                "TP01", "TP02", "TP03", "TP04", "PRESSURE"]
    for col in required:
        exists = col in df.columns
        status = "✓" if exists else "✗ MISSING"
        print(f"  {status} {col}")
    print()
    
    print("=" * 60)
    print("STEP 4: Filter to Phase=6 and check data quality")
    print("=" * 60)
    if "Phase" not in df.columns:
        print("Cannot proceed without Phase column")
        return
    
    df_phase6 = df[df["Phase"] == 6].copy()
    print(f"Rows after filtering Phase=6: {len(df_phase6)}")
    
    if len(df_phase6) == 0:
        print("ERROR: No rows with Phase=6 found!")
        return
    
    # Check for NaN in critical columns
    critical_cols = ["Pirani", "VACUUM", "ShelfTemp", "TP01", "TP02", "TP03", "TP04"]
    print("\nNaN counts in Phase=6 rows:")
    for col in critical_cols:
        if col in df_phase6.columns:
            nan_count = df_phase6[col].isna().sum()
            non_null = df_phase6[col].notna().sum()
            print(f"  {col}: {nan_count} NaN, {non_null} non-null")
        else:
            print(f"  {col}: COLUMN MISSING")
    
    print()
    print("=" * 60)
    print("STEP 5: Sample data from Phase=6 rows")
    print("=" * 60)
    print(df_phase6[required].head(10).to_string())
    print()
    
    print("=" * 60)
    print("STEP 6: Check Cycle/Step combinations")
    print("=" * 60)
    if "Cycle" in df_phase6.columns and "Step" in df_phase6.columns:
        cycle_step_counts = df_phase6.groupby(["Cycle", "Step"]).size()
        print(f"Cycle/Step combinations in Phase=6:\n{cycle_step_counts}")
    else:
        print("Missing Cycle or Step columns")

if __name__ == "__main__":
    main()
