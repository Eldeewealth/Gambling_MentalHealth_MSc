#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
import pyreadstat

def main():
    # 1) CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True, help="wave letter, e.g. k/l/n")
    ap.add_argument("--raw-dir", default="data/raw/UKHLS")
    ap.add_argument("--nrows", type=int, default=200, help="rows to preview (fast)")
    ap.add_argument("--col-limit", type=int, default=25, help="print only the first N column names; 0 = all")
    ap.add_argument("--sample-cols", nargs="*", default=["pidp","sex_dv","sex","age_dv","gor_dv","gamble_pgsi","ghq12score","indinui_xw"])
    args = ap.parse_args()

    # 2) Load a small slice so df.columns is available
    w = args.wave.lower()
    f = Path(args.raw_dir) / f"{w}_indresp.sav"
    if not f.exists():
        raise SystemExit(f"❌ Missing file: {f}")

    df, meta = pyreadstat.read_sav(str(f), row_limit=args.nrows, apply_value_formats=False)

    # 3) >>> THIS IS THE BLOCK YOU ASKED ABOUT <<<
    cols = list(df.columns)
    limit = args.col_limit if (args.col_limit is not None and args.col_limit > 0) else len(cols)
    print(f"\n🧾 Columns: {len(cols)} total")
    print(cols[:limit])

    # 4) Optional: show a compact sample of key columns (only those that exist)
    pick = [c for c in args.sample_cols if c in df.columns]
    if pick:
        print(f"\n🔎 Sample of key cols ({len(pick)}): {pick}")
        print(df[pick].head(10).to_string(index=False))
    else:
        print("\n(no requested sample columns found; showing head(5))")
        print(df.head(5).to_string(index=False))

    # 5) Optional: show a few value labels
    for var in ("sex_dv","gor_dv"):
        labels = meta.variable_value_labels.get(var)
        if labels:
            items = list(labels.items())[:8]
            print(f"\n🏷️ Value labels for {var} (first 8): {items}")

if __name__ == "__main__":
    main()
