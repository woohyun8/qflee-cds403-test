# Q-Flee for CDS 403

This repository contains Flee 3 with the Q-Flee extension (Q-Learning route choice) used in our
forced-migration experiments. You can run every scenario on your own laptop. Your results should fall
inside the ranges we obtained from 20 runs per scenario on GMU's Hopper cluster (see `hopper/`).

## 1. Set up (about 5 minutes)

You need Python 3.8 or newer.

    git clone https://github.com/woohyun8/qflee-cds403-test.git
    cd qflee-cds403-test
    python3 -m venv venv
    source venv/bin/activate          # Windows: venv\Scripts\activate
    pip install numpy pandas pyyaml beartype

(`requirements.txt` also lists packages such as `mpi4py` and `muscle3` that are only needed for
parallel and coupled runs. You do not need them here.)

## 2. Run a scenario

Run from the repository root. The last argument chooses the model: `mali` is baseline Flee,
`qmali` is Q-Flee.

    python3 mines/run_mines.py conflict_input/mali conflict_validation/mali2012 300 mali
    python3 mines/run_mines.py conflict_input/mali conflict_validation/mali2012 300 qmali

| Scenario | Command arguments | Days |
|---|---|---|
| Mali | `conflict_input/mali conflict_validation/mali2012 300 <mali or qmali>` | 300 |
| Burundi | `conflict_input/burundi conflict_validation/burundi2015 396 <burundi or qburundi>` | 396 |
| CAR | `conflict_input/car conflict_validation/car2013 820 <car or qcar>` | 820 |
| Syria | `conflict_input/syria conflict_validation/syria2013 1095 <syria or qsyria>` | 1095 |

Approximate time per run on an Apple M3 Pro laptop (single core): Mali about 1 minute, Burundi 2-3 minutes,
CAR 7-8 minutes, Syria 40-45 minutes. The terminal prints many "Empty route" warnings; these are normal.

Each run writes one CSV to `mines/myoutputs/` (baseline runs in `baseline/`, Q-Flee runs in
`q<scenario>/`). The file name starts with the date and time of the run. Each row is one simulated day;
for each camp it lists the simulated population (`sim`), the UNHCR population (`data`) and the error.
If you start two runs within the same second, the second file overwrites the first.

## 3. Check your result

    python3 mines/compute_ard.py

This computes the ARD of every run in `mines/myoutputs/` and compares it with the Hopper range.
ARD is the mean of the `Total error` column over all days. Ignore the `ARD = ...` number that
`run_mines.py` prints at the end of a run: it uses a different formula.

Flee is stochastic, so every run gives a slightly different value. A result inside (or within 0.002 of)
the Hopper range means your run reproduces ours.

| Scenario | Baseline ARD range | Q-Flee ARD range |
|---|---|---|
| Mali | 0.2744-0.2784 | 0.2721-0.2771 |
| Burundi | 0.5187-0.5214 | 0.5212-0.5251 |
| CAR | 0.3392-0.3420 | 0.3410-0.3432 |
| Syria | 0.2624-0.2635 | 0.2648-0.2660 |

The output CSVs of all 160 Hopper runs, their runtimes and the SLURM scripts are in `hopper/`.

---

# Flee
[![Build Status](https://travis-ci.com/djgroen/flee.svg?branch=master)](https://travis-ci.com/djgroen/flee)
[![GitHub Issues](https://img.shields.io/github/issues/djgroen/flee.svg)](https://github.com/djgroen/flee/issues)
[![GitHub last-commit](https://img.shields.io/github/last-commit/djgroen/flee.svg)](https://github.com/djgroen/flee/commits/master)


Flee is an agent-based modelling toolkit which is purpose-built for simulating
the movement of individuals across geographical locations. Flee is currently
used primarily for modelling the movements of forcibly displaced
persons (FDPs).

Full documentation of Flee is available at https://flee.readthedocs.io.

Official releases will be done on https://www.github.com/djgroen/flee. (flee-release is deprecated and will be removed in due course.)

## Parallel performance testing

Parallel tests can be performed using test_par.py. The interface is as follows:

mpirun -np <cores> python3 tests/test_par.py [options]
  
Options can be as follows:

* "-p", "--parallelmode" - Parallelization mode ([advanced], classic, cl-hilat OR adv-lowlat).
* "-N", "--initialagents" - Number of agents at the start of the simulation [100000].
* "-d", "--newagentsperstep", Number of agents added per time step [1000].
* "-t", "--simulationperiod", Duration of the simulation in days [10].

Here are a few settings good for benchmarking:

* `mpirun -np <cores> python3 test_par.py -N 500000 -p advanced -d 10000 -t 10`
* `mpirun -np <cores> python3 test_par.py -N 500000 -p classic -d 10000 -t 10`
* `mpirun -np <cores> python3 test_par.py -N 500000 -p cl-hilat -d 10000 -t 10`
* `mpirun -np <cores> python3 test_par.py -N 500000 -p adv-lowlat -d 10000 -t 10`


# Acknowledgements
The development on Flee has been made possible through funding from the Horizon 2020 funded HiDALGO project (grant no. 824115, https://hidalgo-project.eu) and VECMA (grant no. 800925, https://www.vecma.eu).
