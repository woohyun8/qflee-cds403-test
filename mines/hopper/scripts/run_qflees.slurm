#!/bin/bash
#SBATCH --job-name=flee_qflee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=1-10
#SBATCH --output=/home/jcolcha1/flee/mines/hopper/houtputs/qflee_%A_%a.out
#SBATCH --error=/home/jcolcha1/flee/mines/hopper/houtputs/qflee_%A_%a.err
#SBATCH --account=css610sp26
#SBATCH --partition=normal

# ── Q-Flee Ensemble ────────────────────────────────────────────────────────────
# Job array: each array index is one independent trial.
# %A = master job ID, %a = array index (1–10).
#
# Q-Flee model files (mines/models/):
#   flee_car.py     → shared Q-table + informed prior (LOCALITY_DECAY=300km)
#   flee_mali.py    → shared Q-table + informed prior (LOCALITY_DECAY=500km)
#   flee_burundi.py → shared Q-table + informed prior (LOCALITY_DECAY=150km)
#   flee_syria.py   → shared Q-table + informed prior (LOCALITY_DECAY=400km)
#
# Scenarios and sim periods:
#   qcar     → conflict_input/car     | conflict_validation/car2013    | 820 days
#   qmali    → conflict_input/mali    | conflict_validation/mali2012   | 300 days
#   qburundi → conflict_input/burundi | conflict_validation/burundi2015| 396 days
#   qsyria   → conflict_input/syria   | conflict_validation/syria2013  | 1095 days
#
# Communication layer: disabled by default (LONG_RANGE_PROB=0.0 in each model).
# ──────────────────────────────────────────────────────────────────────────────

# ── Population scale ───────────────────────────────────────────────────────────
# flee/simsetting.yml sets hasten=1 (PopulationScaleDownFactor=1).
# Every simulated agent represents exactly one displaced person — full 1:1 scale.

# ── Environment ────────────────────────────────────────────────────────────────
FLEE_ROOT="/home/jcolcha1/flee"

cd "$FLEE_ROOT" || { echo "ERROR: FLEE_ROOT not found: $FLEE_ROOT"; exit 1; }

# Load Python and activate virtual environment
# Run `module avail python` on Hopper to confirm exact module name
module load python/3.8.6-ff
source "$FLEE_ROOT/flee-env/bin/activate"

# ── Sim lengths ────────────────────────────────────────────────────────────────
CAR_DAYS=820
MALI_DAYS=300
BURUNDI_DAYS=396
SYRIA_DAYS=1095

echo "=============================="
echo "Trial ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================="

# ── Q-Flee CAR 2013 ───────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Q-Flee CAR (${CAR_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/car \
    conflict_validation/car2013 \
    $CAR_DAYS \
    qcar

# ── Q-Flee Mali 2012 ──────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Q-Flee Mali (${MALI_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/mali \
    conflict_validation/mali2012 \
    $MALI_DAYS \
    qmali

# ── Q-Flee Burundi 2015 ───────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Q-Flee Burundi (${BURUNDI_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/burundi \
    conflict_validation/burundi2015 \
    $BURUNDI_DAYS \
    qburundi

# ── Q-Flee Burundi 2015 v2 (strengthened prior, PRIOR_SCALE=0.20) ──────────────
echo "[$(date +%H:%M:%S)] Running Q-Flee Burundi v2 (${BURUNDI_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/burundi \
    conflict_validation/burundi2015 \
    $BURUNDI_DAYS \
    q2burundi

# ── Q-Flee Syria 2013 ─────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Q-Flee Syria (${SYRIA_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/syria \
    conflict_validation/syria2013 \
    $SYRIA_DAYS \
    qsyria

echo "=============================="
echo "Trial ${SLURM_ARRAY_TASK_ID} complete: $(date)"
echo "=============================="
