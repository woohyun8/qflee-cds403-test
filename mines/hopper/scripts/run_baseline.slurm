#!/bin/bash
#SBATCH --job-name=flee_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=1-10
#SBATCH --output=/home/jcolcha1/flee/mines/hopper/houtputs/baseline_%A_%a.out
#SBATCH --error=/home/jcolcha1/flee/mines/hopper/houtputs/baseline_%A_%a.err
#SBATCH --account=css610sp26
#SBATCH --partition=normal

# ── Q-Flee Baseline Ensemble ───────────────────────────────────────────────────
# Job array: each array index is one independent trial.
# %A = master job ID, %a = array index (1–10).
#
# Scenarios and sim periods:
#   car     → conflict_input/car     | conflict_validation/car2013    | 820 days
#   mali    → conflict_input/mali    | conflict_validation/mali2012   | 300 days
#   burundi → conflict_input/burundi | conflict_validation/burundi2015| 396 days
#   syria   → conflict_input/syria   | conflict_validation/syria2013  | 1095 days
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

# ── CAR 2013 ──────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running CAR baseline (${CAR_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/car \
    conflict_validation/car2013 \
    $CAR_DAYS \
    car

# ── Mali 2012 ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Mali baseline (${MALI_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/mali \
    conflict_validation/mali2012 \
    $MALI_DAYS \
    mali

# ── Burundi 2015 ──────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Burundi baseline (${BURUNDI_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/burundi \
    conflict_validation/burundi2015 \
    $BURUNDI_DAYS \
    burundi

# ── Syria 2013 ────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] Running Syria baseline (${SYRIA_DAYS} days)..."
python3 mines/run_mines.py \
    conflict_input/syria \
    conflict_validation/syria2013 \
    $SYRIA_DAYS \
    syria

echo "=============================="
echo "Trial ${SLURM_ARRAY_TASK_ID} complete: $(date)"
echo "=============================="
