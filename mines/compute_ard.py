#!/usr/bin/env python3
"""
compute_ard.py - compute ARD from Flee output CSVs and compare with the Hopper reference runs.

ARD of one run = mean of the "Total error" column over all simulated days.
(The "ARD = ..." value printed by run_mines.py at the end of a run uses a different
formula and should NOT be used for comparison.)

Usage (from the repository root):
    python3 mines/compute_ard.py                     # reads mines/myoutputs/
    python3 mines/compute_ard.py path/to/folder      # any folder with *_conflict.csv files
"""
import csv
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "mines", "myoutputs")
ref_path = os.path.join(ROOT, "hopper", "results", "ard_summary.csv")


def run_ard(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        i = header.index("Total error")
        values = [float(row[i]) for row in reader if len(row) > i and row[i] != ""]
    return sum(values) / len(values)


results = {}
for path in glob.glob(os.path.join(folder, "**", "*_conflict.csv"), recursive=True):
    label = os.path.basename(path).split("_")[1]  # e.g. 0626140524_qmali_conflict.csv -> qmali
    results.setdefault(label, []).append(run_ard(path))

if not results:
    sys.exit(f"No *_conflict.csv files found under {folder}")

ref = {}
if os.path.exists(ref_path):
    with open(ref_path, newline="") as f:
        for row in csv.DictReader(f):
            ref[row["label"]] = row

print(f"{'label':10s} {'runs':>4s} {'your ARD':>9s}   {'Hopper range (20 runs)':>24s}  check")
for label in sorted(results):
    a = results[label]
    mean = statistics.mean(a)
    if label in ref:
        lo, hi = float(ref[label]["ard_min"]), float(ref[label]["ard_max"])
        each_ok = all(lo - 0.002 <= x <= hi + 0.002 for x in a)
        check = "OK" if each_ok else "outside Hopper range - check your setup"
        rng = f"{lo:.4f} - {hi:.4f}"
    else:
        check, rng = "", "(no reference)"
    print(f"{label:10s} {len(a):4d} {mean:9.4f}   {rng:>24s}  {check}")
