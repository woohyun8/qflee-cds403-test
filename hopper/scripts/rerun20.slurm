#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=css610sp26
#SBATCH --partition=normal
#SBATCH --output=/home/wsong8/css610/q-flee/mines/hopper/houtputs/rerun_%x_%A_%a.out
#SBATCH --error=/home/wsong8/css610/q-flee/mines/hopper/houtputs/rerun_%x_%A_%a.err

INPUT=$1; VALID=$2; DAYS=$3; LABEL=$4
FLEE_ROOT="/home/wsong8/css610/q-flee"
cd "$FLEE_ROOT" || exit 1
export PYTHONPATH="$FLEE_ROOT:$PYTHONPATH"
module load python/3.8.6-ff
source "$FLEE_ROOT/../flee-env/bin/activate"

sleep $(( SLURM_ARRAY_TASK_ID * 7 ))
START=$(date +%s)
python3 mines/run_mines.py conflict_input/$INPUT conflict_validation/$VALID $DAYS $LABEL \
  2> >(grep -v -e "^Location:" -e "Empty route" >&2)
STATUS=$?
END=$(date +%s)
echo "RUNTIME,$LABEL,$SLURM_ARRAY_TASK_ID,$SLURM_JOB_ID,$(hostname),$SLURM_CPUS_PER_TASK,$((END-START)),$STATUS"
