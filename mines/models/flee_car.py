"""
flee_car.py — Q-learning integration for the CAR scenario.

STRATEGY: shared Q-table + informed geographic prior

  Informed Prior (pre-simulation):
    Agents start with Q-values initialized from Dijkstra distances,
    NOT from zero. This encodes Level 2 (Route-familiar) awareness:
    CAR communities had route knowledge from the 2004-2007 Bush War
    displacement cycles. By 2013, people near historically-used camps
    knew roughly which direction led to safety.

      Q_TABLE[state][action] = max(0, km_progress) × PRIOR_SCALE
                                × exp(-dist_to_camp / LOCALITY_DECAY_KM)

  Shared Q-table:
    All agents update a single global Q-table. This is a simplifying
    assumption (collective intelligence) that enables convergence:
    ~100k agents × ~50 steps = ~4M Bellman updates → Q-values
    converge within 5-10 simulation days.

    Individual tables were tested but with only ~50 updates per agent
    they are in the noise regime — Bellman updates degrade the prior
    rather than refining it. Shared table with informed prior is the
    working configuration.

    Limitation acknowledged: individual tables + communication model
    (30% phone penetration, ego-network sharing) is the historically
    accurate architecture and remains future work.

  Score Augmentation:
    Q-values bias flee's getEndPointScore via a capped multiplier:
      score × (1 + q_val / Q_MAX)

  Bellman Reward (pure geographic, no deficit):
    R = (km_before - km_after) × PROG_SCALE   ← km progress toward camp
      + ARRIVAL_BONUS  if endpoint is camp
      - CONFLICT_PENALTY × conflict_intensity

    No deficit signal: when camps are overpopulated, deflecting agents
    to geometrically wrong locations cascades ARD error. Removed.

  Communication Layer (disabled; set LONG_RANGE_PROB > 0 to enable):
    Phone owners share Q-value updates with a contact elsewhere,
    propagating personal experience through ego networks (GSMA 2013:
    ~30% CAR mobile penetration, 2-4 trusted contacts per person).
"""

import csv
import heapq
import math
import os
import random
import sys

# ── Q-learning hyperparameters ─────────────────────────────────────────────────
LEARNING_RATE    = 0.7
DISCOUNT_FACTOR  = 0.95
PROG_SCALE       = 0.02    # reward per km of progress toward nearest camp
ARRIVAL_BONUS    = 10.0    # reward on stepping into a camp
CONFLICT_PENALTY = 3.0     # penalty per unit of conflict intensity
Q_MAX            = 50.0    # caps score multiplier: max 2×

# ── Prior knowledge hyperparameters ───────────────────────────────────────────
PRIOR_SCALE       = 0.05   # Q_init per km of progress (100km → Q≈5 near camps)
LOCALITY_DECAY_KM = 300.0  # e-folding distance for prior confidence

# ── Communication hyperparameters (disabled) ───────────────────────────────────
PHONE_PENETRATION = 0.30   # CAR 2013 ~30% (GSMA/GeoPoll)
LONG_RANGE_PROB   = 0.00   # disabled; set >0 to enable ego-network Q-sharing
LONG_RANGE_WEIGHT = 0.30

# ── Module-level state ─────────────────────────────────────────────────────────
Q_TABLE:       dict = {}   # {state → {action → q_value}} — shared across agents
CAMP_DISTANCE: dict = {}   # {loc → km to nearest camp}
NEAREST_CAMP:  dict = {}   # {loc → nearest camp name}
ADJ:           dict = {}   # {loc → [(neighbour, km), ...]}
AGENT_COMM:    dict = {}   # {id(agent) → comm_strength ∈ [0,1]}

_LM         = None
_D          = None
_CAMP_NAMES = []
_patched    = False


# ── CSV helper ─────────────────────────────────────────────────────────────────

def _col(reader, *candidates):
    for fn in reader.fieldnames:
        if fn.strip().replace('"', '') in candidates:
            return fn
    raise KeyError(f"None of {candidates} found in {reader.fieldnames}")


# ── Distance computation ───────────────────────────────────────────────────────

def _build_distances(input_csv_directory: str):
    """Multi-source Dijkstra from all camps. Returns (dist, nearest, adj)."""
    locs_file = os.path.join(input_csv_directory, "locations.csv")
    camps: set = set()
    with open(locs_file, newline="") as f:
        reader = csv.DictReader(f)
        nc = _col(reader, "#name", "name")
        tc = _col(reader, "location_type")
        for row in reader:
            name = row[nc].strip().strip('"')
            lt   = row[tc].strip().strip('"').lower()
            if "camp" in lt:
                camps.add(name)

    routes_file = os.path.join(input_csv_directory, "routes.csv")
    adj: dict = {}
    with open(routes_file, newline="") as f:
        reader = csv.DictReader(f)
        n1c = _col(reader, "#name1", "name1")
        n2c = _col(reader, "name2")
        dc  = _col(reader, "distance")
        for row in reader:
            a  = row[n1c].strip().strip('"')
            b  = row[n2c].strip().strip('"')
            km = float(row[dc])
            adj.setdefault(a, []).append((b, km))
            adj.setdefault(b, []).append((a, km))

    INF  = float("inf")
    dist = {c: 0.0 for c in camps}
    src  = {c: c   for c in camps}
    heap = [(0.0, c) for c in camps]
    heapq.heapify(heap)
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, INF):
            continue
        for nbr, km in adj.get(node, []):
            new_d = d + km
            if new_d < dist.get(nbr, INF):
                dist[nbr] = new_d
                src[nbr]  = src[node]
                heapq.heappush(heap, (new_d, nbr))

    return dist, src, adj


def _init_q_table() -> None:
    """
    Pre-populate Q_TABLE with geographic prior.

    Encodes Level 2 (Route-familiar) awareness: community knowledge
    from CAR's 2004-2007 Bush War displacement, which persisted into
    2013. Confidence decays with distance — near-camp communities have
    stronger, more precise route knowledge than distant ones.

    Only positive progress gets a prior > 0; moving away from camps
    starts at 0 (no negative bias).
    """
    INF = float("inf")
    n_entries = 0
    for state, neighbours in ADJ.items():
        d_state = CAMP_DISTANCE.get(state, INF)
        if d_state >= INF:
            continue
        locality = math.exp(-d_state / LOCALITY_DECAY_KM)
        Q_TABLE[state] = {}
        for nbr, _ in neighbours:
            d_nbr = CAMP_DISTANCE.get(nbr, INF)
            if d_nbr >= INF:
                continue
            km_progress = d_state - d_nbr
            Q_TABLE[state][nbr] = max(0.0, km_progress) * PRIOR_SCALE * locality
            n_entries += 1
    print(f"[Q-Flee CAR] Q_TABLE prior: {n_entries} entries "
          f"(PRIOR_SCALE={PRIOR_SCALE}, LOCALITY_DECAY={LOCALITY_DECAY_KM}km)",
          file=sys.stderr)


# ── Deficit helper ─────────────────────────────────────────────────────────────

def _get_deficit(camp_name: str, time: int) -> float:
    if _D is None or _LM is None or not camp_name:
        return 0.0
    try:
        data_val = _D.get_field(camp_name, time)
        sim_val  = _LM[camp_name].numAgents if camp_name in _LM else 0
        if data_val > 0:
            return (data_val - sim_val) / float(data_val)
    except Exception:
        pass
    return 0.0


# ── Communication helper ───────────────────────────────────────────────────────

def _get_comm_strength(agent) -> float:
    """70% silent (no phone); 30% phone owners with strength U(0.3, 0.8)."""
    aid = id(agent)
    if aid not in AGENT_COMM:
        AGENT_COMM[aid] = (random.uniform(0.3, 0.8)
                           if random.random() < PHONE_PENETRATION
                           else 0.0)
    return AGENT_COMM[aid]


# ── Runtime injection ──────────────────────────────────────────────────────────

def set_runtime_refs(lm, d, camp_names):
    global _LM, _D, _CAMP_NAMES
    _LM         = lm
    _D          = d
    _CAMP_NAMES = list(camp_names)
    print(f"[Q-Flee CAR] runtime refs set ({len(_CAMP_NAMES)} camps).",
          file=sys.stderr)


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup(input_csv_directory: str) -> None:
    global CAMP_DISTANCE, NEAREST_CAMP, ADJ, Q_TABLE, _patched

    if _patched:
        return

    CAMP_DISTANCE, NEAREST_CAMP, ADJ = _build_distances(input_csv_directory)
    print("[Q-Flee CAR] Dijkstra done.", file=sys.stderr)

    Q_TABLE = {}
    _init_q_table()

    from flee import moving

    _orig_score  = moving.getEndPointScore
    _orig_select = moving.selectRoute

    # ── Score augmentation ────────────────────────────────────────────────────
    def _q_score(agent, endpoint, time):
        base = _orig_score(agent, endpoint, time)
        if agent.location.camp:
            return base
        q_val = Q_TABLE.get(agent.location.name, {}).get(endpoint.name, 0.0)
        if q_val > 0.0:
            base *= (1.0 + q_val / Q_MAX)
        return base

    # ── Route selection + Bellman update ──────────────────────────────────────
    def _q_select(a, time, debug=False, return_all_routes=False,
                  system2_active=False):
        route = _orig_select(a, time, debug, return_all_routes, system2_active)

        if return_all_routes or not route or a.location.camp:
            return route

        first_hop = route[0]
        state     = a.location.name
        INF       = float("inf")

        # Reward: pure geographic progress (no deficit)
        d_before = CAMP_DISTANCE.get(state,     INF)
        d_after  = CAMP_DISTANCE.get(first_hop, INF)
        reward   = 0.0
        if d_before < INF and d_after < INF:
            reward += (d_before - d_after) * PROG_SCALE

        for lnk in a.location.links:
            if lnk.endpoint.name == first_hop:
                tgt = lnk.endpoint
                if tgt.camp:
                    reward += ARRIVAL_BONUS
                if tgt.conflict > 0:
                    reward -= CONFLICT_PENALTY * max(0.0, tgt.conflict)
                break

        # Bellman update on shared table
        Q_TABLE.setdefault(state, {})
        old_q   = Q_TABLE[state].get(first_hop, 0.0)
        nxt     = Q_TABLE.get(first_hop, {})
        max_nxt = max(nxt.values()) if nxt else 0.0
        Q_TABLE[state][first_hop] = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_nxt - old_q
        )

        # ── Optional: ego-network Q-sharing (phone owners) ────────────────────
        # Disabled (LONG_RANGE_PROB = 0.0). Re-enable by setting > 0.
        comm = _get_comm_strength(a)
        if comm > 0.0 and _CAMP_NAMES and random.random() < LONG_RANGE_PROB * comm:
            target_camp  = random.choice(_CAMP_NAMES)
            camp_deficit = _get_deficit(target_camp, time)
            if abs(camp_deficit) > 0.05:
                for lnk in a.location.links:
                    nbr    = lnk.endpoint.name
                    toward = (nbr == target_camp or
                              NEAREST_CAMP.get(nbr) == target_camp)
                    if toward:
                        Q_TABLE.setdefault(state, {})
                        old_lr = Q_TABLE[state].get(nbr, 0.0)
                        signal = camp_deficit * 20.0 * comm
                        Q_TABLE[state][nbr] = (
                            old_lr + LEARNING_RATE * LONG_RANGE_WEIGHT * (signal - old_lr)
                        )

        return route

    moving.getEndPointScore = _q_score
    moving.selectRoute      = _q_select
    _patched = True
    print("[Q-Flee CAR] shared Q-table + informed prior patched.",
          file=sys.stderr)
