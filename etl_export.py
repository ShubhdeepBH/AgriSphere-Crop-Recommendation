# etl_export.py
"""
Export SQL tables to CSV for Power BI (Option A).
Writes CSVs into ./exports/ by default.

Usage:
    python etl_export.py          # creates/overwrites ./exports/*.csv
    python etl_export.py --out my_exports
"""
import argparse
import os
from db import init_db, export_tables_to_csv

def main(out_dir="exports"):
    # ensure DB tables exist (safe)
    init_db()
    export_tables_to_csv(out_dir)
    print(f"Exported CSVs to: {os.path.abspath(out_dir)}")
    print("Files: requests.csv, recommendations.csv, prices.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="exports", help="Output folder for CSVs")
    args = p.parse_args()
    main(args.out)
