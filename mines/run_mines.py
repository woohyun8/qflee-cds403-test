#!/usr/bin/env python3
"""
run_mines.py — Flee runner with auto-timestamped CSV output.

Usage:
  python3 mines/run_mines.py <conflict_input_dir> <validation_dir> <days> <label>

Labels:
  mali, burundi, car, syria       → baseline run → mines/myoutputs/baseline/MMDDTTTT_<label>_conflict.csv
  qmali, qburundi, qcar, qsyria  → Q-Flee run   → mines/myoutputs/<label>/MMDDTTTT_<label>_conflict.csv

  TTTT = military time (24-hour HHMM, e.g. 1423 = 2:23 PM)

Example:
  python3 mines/run_mines.py conflict_input/mali conflict_validation/mali2012 300 qmali
"""

import os
import sys
from datetime import datetime, timezone, timedelta

# ── resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
FLEE_ROOT   = os.path.dirname(SCRIPT_DIR)
MINES_DIR   = SCRIPT_DIR
OUTPUTS_DIR = os.path.join(MINES_DIR, "myoutputs")

sys.path.insert(0, FLEE_ROOT)

# ── parse args ───────────────────────────────────────────────────────────────
if len(sys.argv) < 5:
    print(__doc__)
    sys.exit(1)

input_csv_directory      = sys.argv[1]
validation_data_directory = sys.argv[2]
duration                 = int(sys.argv[3])
label                    = sys.argv[4].lower()   # e.g. "qmali" or "mali"

# resolve relative paths from flee root
if not os.path.isabs(input_csv_directory):
    input_csv_directory = os.path.join(FLEE_ROOT, input_csv_directory)
if not os.path.isabs(validation_data_directory):
    validation_data_directory = os.path.join(FLEE_ROOT, validation_data_directory)

# ── determine output folder and filename ─────────────────────────────────────
is_qflee   = label.startswith("q")
subfolder  = label if is_qflee else "baseline"
EST        = timezone(timedelta(hours=-4))   # EDT (UTC-4) in June
timestamp  = datetime.now(EST).strftime("%m%d%H%M%S")
filename   = f"{timestamp}_{label}_conflict.csv"
output_dir = os.path.join(OUTPUTS_DIR, subfolder)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

print(f"[run_mines] Output → {output_path}", file=sys.stderr)

# ── run flee simulation ───────────────────────────────────────────────────────
from flee import flee, spawning
from flee.datamanager import handle_refugee_data, read_period
from flee import InputGeography
import numpy as np
import flee.postprocessing.analysis as a
from flee.SimulationSettings import SimulationSettings

# ── Q-Flee patch (applied before any agents are spawned) ─────────────────────
if is_qflee:
    _models_dir = os.path.join(MINES_DIR, "models")
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)
    scenario = label[1:]   # "qmali" → "mali"
    if scenario == "mali":
        import flee_mali as _qflee_model
    elif scenario == "car":
        import flee_car as _qflee_model
    elif scenario == "burundi":
        import flee_burundi as _qflee_model
    elif scenario == "2burundi":
        import flee_q2burundi as _qflee_model
    elif scenario == "syria":
        import flee_syria as _qflee_model
    else:
        print(f"[run_mines] WARNING: no Q-Flee model found for scenario '{scenario}'. Running baseline.", file=sys.stderr)
        _qflee_model = None
    if _qflee_model is not None:
        _qflee_model.setup(input_csv_directory)

start_date, end_time = read_period.read_sim_period(
    os.path.join(input_csv_directory, "sim_period.csv"))

if duration > 0:
    end_time = duration

# Prefer flee/simsetting.yml (sets hasten=1 → full population, insert_day0=True,
# move rules, etc.). Fall back to root-level, then empty.yml.
_simsetting_candidates = [
    os.path.join(FLEE_ROOT, "flee", "simsetting.yml"),
    os.path.join(FLEE_ROOT, "simsetting.yml"),
    os.path.join(FLEE_ROOT, "empty.yml"),
]
_simsetting_path = next(p for p in _simsetting_candidates if os.path.exists(p))
print(f"[run_mines] simsetting → {_simsetting_path}", file=sys.stderr)
flee.SimulationSettings.ReadFromYML(_simsetting_path)

flee.SimulationSettings.ConflictInputFile = os.path.join(input_csv_directory, "conflicts.csv")
flee.SimulationSettings.FloodLevelInputFile = os.path.join(input_csv_directory, "flood_level.csv")

e  = flee.Ecosystem(start_date, demographics_test_prefix=input_csv_directory)
ig = InputGeography.InputGeography()

ig.ReadLocationsFromCSV(os.path.join(input_csv_directory, "locations.csv"))
ig.ReadLinksFromCSV(os.path.join(input_csv_directory, "routes.csv"))
ig.ReadClosuresFromCSV(os.path.join(input_csv_directory, "closures.csv"))

e, lm = ig.StoreInputGeographyInEcosystem(e)

if SimulationSettings.spawn_rules["read_from_agents_csv_file"]:
    ig.ReadAgentsFromCSV(e, os.path.join(input_csv_directory, "agents.csv"))

d = handle_refugee_data.RefugeeTable(
    csvformat="generic",
    data_directory=validation_data_directory,
    start_date=start_date,
    data_layout="data_layout.csv",
    population_scaledown_factor=SimulationSettings.optimisations["PopulationScaleDownFactor"],
    start_empty=SimulationSettings.spawn_rules["EmptyCampsOnDay0"])

d.ReadL1Corrections(os.path.join(input_csv_directory, "registration_corrections.csv"))

camp_locations = e.get_camp_names()

# ── Q-Flee runtime refs (injected after lm and d are ready) ──────────────────
if is_qflee and '_qflee_model' in dir() and _qflee_model is not None:
    if hasattr(_qflee_model, 'set_runtime_refs'):
        _qflee_model.set_runtime_refs(lm, d, camp_locations)

output_header = "Day,Date,"
for l in camp_locations:
    spawning.add_initial_refugees(e, d, lm[l])
    output_header += "%s sim,%s data,%s error," % (lm[l].name, lm[l].name, lm[l].name)
output_header += "Total error,refugees in camps (UNHCR),total refugees (simulation),raw UNHCR refugee count,refugees in camps (simulation),refugee_debt"

if SimulationSettings.log_levels["idp_totals"] > 0:
    output_header += ",total IDPs"

lines = [output_header]
refugee_debt = 0
refugees_raw = 0

for t in range(0, end_time):
    ig.AddNewConflictZones(e, t)
    new_refs, refugees_raw, refugee_debt = spawning.spawn_daily_displaced(e, t, d)
    spawning.refresh_spawn_weights(e)
    e.enact_border_closures(t)
    e.evolve()

    camps    = [lm[i] for i in camp_locations]
    loc_data = [d.get_field(i, t) for i in camp_locations]

    refugees_in_camps_sim = sum(c.numAgents for c in camps)
    errors     = [a.rel_error(lm[i].numAgents, loc_data[j]) for j, i in enumerate(camp_locations)]
    abs_errors = [a.abs_error(lm[i].numAgents, loc_data[j]) for j, i in enumerate(camp_locations)]

    row = "%s,%s" % (t, e.date_string)
    for i in range(len(errors)):
        row += ",%s,%s,%s" % (lm[camp_locations[i]].numAgents, loc_data[i], errors[i])

    if refugees_raw > 0:
        row += ",%s,%s,%s,%s,%s,%s" % (
            float(np.sum(abs_errors)) / float(refugees_raw),
            int(sum(loc_data)), e.numAgents(), refugees_raw,
            refugees_in_camps_sim, refugee_debt)
    else:
        row += ",0.0,0,{},0,{},0".format(e.numAgents(), refugees_in_camps_sim)

    if SimulationSettings.log_levels["idp_totals"] > 0:
        row += ",{}".format(e.numIDPs())

    lines.append(row)

# ── write output ─────────────────────────────────────────────────────────────
with open(output_path, "w") as f:
    f.write("\n".join(lines) + "\n")

# ── print ARD summary ─────────────────────────────────────────────────────────
total_errors = []
for row in lines[1:]:
    cols = row.split(",")
    try:
        total_errors.append(float(cols[len(camp_locations) * 3 + 1]))
    except (IndexError, ValueError):
        pass

if total_errors:
    ard = sum(total_errors) / len(total_errors)
    print(f"[run_mines] ARD = {ard:.4f}  |  n={len(total_errors)}  |  {filename}", file=sys.stderr)
