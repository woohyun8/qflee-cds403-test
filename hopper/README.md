# Hopper runs (reference results)

This folder holds the reference runs of this repository's code on George Mason University's
Hopper cluster. Students do **not** need Hopper: running the same code on a laptop should give
ARD values inside the ranges below (see the main README).

## What was run

- Code: this repository (`mines/run_mines.py`, Q-Flee models in `mines/models/`, settings in `flee/simsetting.yml`).
- 8 conditions: baseline Flee and Q-Flee for Mali (300 days), Burundi (396 days), CAR (820 days) and Syria (1,095 days).
- 20 runs per condition, 160 runs in total, November 18-19, 2026.
- Each run used a single core (`--cpus-per-task=1`, 8 GB memory). Runs were submitted as SLURM job arrays,
  20 runs per condition, so many runs executed at the same time on different cores and nodes.
- Agent-level logging was turned on for these runs so that a dashboard can replay agent movement.
  Only the logging code was changed (see `scripts/Diagnostics_logging_patch.diff`); the simulation
  logic is unchanged. A separate set of 160 runs without logging (November 17) gave the same ARD values
  to within 0.0006 per condition.

## Folder contents

| Path | Contents |
|---|---|
| `results/<label>/runNN/*_conflict.csv` | Output CSV of each run (label: `mali`, `qmali`, `burundi`, `qburundi`, `car`, `qcar`, `syria`, `qsyria`) |
| `results/ard_summary.csv` | ARD per condition: mean, SD, min, max over the 20 runs |
| `runtime/runtime_per_run.csv` | Runtime, node, cores and exit status of each of the 160 runs |
| `runtime/runtime_summary.csv` | Runtime per condition (minutes) |
| `runtime/runtime_summary_nov17_no_agent_logging.csv` | Runtime of the November 17 set run without agent logging |
| `scripts/rerun20.slurm` | SLURM script used for the runs without agent logging |
| `scripts/logrun20.slurm` | SLURM script used for the runs in `results/` (agent logging on) |
| `scripts/Diagnostics_logging_patch.diff` | The change to `flee/Diagnostics.py` that samples the agent log |

The agent logs themselves (about 1.1 GB) are not stored in this repository.

## ARD (mean of the "Total error" column over all days)

| Scenario | Baseline mean (range) | Q-Flee mean (range) |
|---|---|---|
| Mali | 0.2761 (0.2744-0.2784) | 0.2746 (0.2721-0.2771) |
| Burundi | 0.5198 (0.5187-0.5214) | 0.5232 (0.5212-0.5251) |
| CAR | 0.3406 (0.3392-0.3420) | 0.3422 (0.3410-0.3432) |
| Syria | 0.2630 (0.2624-0.2635) | 0.2653 (0.2648-0.2660) |

## Runtime on Hopper (minutes per run, single core)

| Scenario | Baseline | Q-Flee |
|---|---|---|
| Mali | 3.6 | 3.2 |
| Burundi | 10.1 | 9.5 |
| CAR | 29.1 | 33.0 |
| Syria | 185.0 | 193.1 |

All 160 runs finished with exit status 0. Runtimes vary with the node type the scheduler assigns.

## Running on Hopper yourself

The scripts contain paths for user `wsong8` (`/home/wsong8/css610/...`), the SLURM account `css610sp26`
and a virtual environment at `/home/wsong8/css610/flee-env`. Change these to your own before using them.
Example submission for 20 Mali Q-Flee runs:

    sbatch --array=1-20 --job-name=qmali --time=00:30:00 hopper/scripts/rerun20.slurm mali mali2012 300 qmali
