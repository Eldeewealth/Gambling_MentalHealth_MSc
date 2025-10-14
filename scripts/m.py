#!/usr/bin/env python
from pathlib import Path
import argparse
import pandas as pd
import pyreadstat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", required=True, help="wave letter, e.g. k/l/n")
    ap.add_argument("--raw-dir", default="data/raw/UKHLS")
    ap.add_argument("--nrows", type=int, default=200, help="rows to preview (fast)")
    ap.add_argument("--sample-cols", nargs="*", default=["pidp","sex_dv","sex","age_dv","gor_dv","gamble_pgsi","ghq12score","indinui_xw"])
    args = ap.parse_args()

    w = args.wave.lower()
    f = Path(args.raw_dir) / f"{w}_indresp.sav"
    if not f.exists():
        raise SystemExit(f"❌ Missing file: {f}")

    print(f"\n📄 Previewing {f.name} (first {args.nrows} rows)…")
    # row_limit keeps it snappy; omit usecols to see *all* columns
    df, meta = pyreadstat.read_sav(str(f), row_limit=args.nrows, apply_value_formats=False)

    # Columns overview
    cols = list(df.columns)
    print(f"🧾 Columns: {len(cols)} total")
    print("• first 25:", cols[:25])
    if len(cols) > 25:
        print("• …")

    # Show a compact sample of key columns if present
    pick = [c for c in args.sample_cols if c in df.columns]
    if pick:
        print(f"\n🔎 Sample of key cols ({len(pick)}): {pick}")
        print(df[pick].head(10).to_string(index=False))
    else:
        print("\n(Your hint columns not found; showing head(5) of full frame)")
        print(df.head(5).to_string(index=False))

    # Quick numeric summaries for common vars if present
    for c in ("gamble_pgsi","ghq12score","age_dv"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").describe()
            print(f"\n📊 {c} describe():\n{s}")

    # Value labels (if the SAV has them) for a few variables
    for var in ("sex_dv","gor_dv"):
        try:
            labels = meta.variable_value_labels.get(var)
            if labels:
                # print a few label mappings
                items = list(labels.items())[:8]
                print(f"\n🏷️ Value labels for {var} (first 8): {items}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
