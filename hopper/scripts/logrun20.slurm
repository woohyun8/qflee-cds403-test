#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=css610sp26
#SBATCH --partition=normal
#SBATCH --output=/home/wsong8/css610/q-flee-log/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/wsong8/css610/q-flee-log/slurm_logs/%x_%A_%a.err

INPUT=$1; VALID=$2; DAYS=$3; LABEL=$4
export AGENT_SAMPLE=$5; export AGENT_COUNTS=1
ROOT=/home/wsong8/css610/q-flee-log
ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
RUNDIR=$ROOT/logs20/$LABEL/run$ID
export PYTHONPATH="$ROOT:$PYTHONPATH"
module load python/3.8.6-ff
source /home/wsong8/css610/flee-env/bin/activate
rm -rf $RUNDIR; mkdir -p $RUNDIR && cd $RUNDIR

sleep $(( SLURM_ARRAY_TASK_ID * 7 ))
START=$(date +%s)
python3 $ROOT/mines/run_mines.py conflict_input/$INPUT conflict_validation/$VALID $DAYS $LABEL \
  2> >(grep -v -e "^Location:" -e "Empty route" | tee run_stderr.log >&2)
STATUS=$?
END=$(date +%s)
sleep 5
CSV=$(grep -o "Output → .*" run_stderr.log | head -1 | sed 's/Output → //')
cp "$CSV" $RUNDIR/
gzip -f agents.out.0 location_counts.csv links.out.0
echo "RUNTIME,$LABEL,$SLURM_ARRAY_TASK_ID,$SLURM_JOB_ID,$(hostname),$SLURM_CPUS_PER_TASK,$((END-START)),$STATUS,sample_1_in_$AGENT_SAMPLE"
ls -la $RUNDIR
