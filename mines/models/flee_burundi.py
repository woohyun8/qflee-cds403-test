"""
flee_burundi.py — Q-learning integration for the Burundi scenario.

STRATEGY: shared Q-table + informed geographic prior

  Informed Prior (pre-simulation):
    Agents start with Q-values initialized from Dijkstra distances,
    NOT from zero. This encodes Level 2 (Route-familiar) awareness:

    Burundi has one of the densest displacement histories in Africa.
    The 1993-2005 civil war (triggered by the assassination of President
    Ndadaye) killed ~300,000 people and displaced hundreds of thousands
    to the same camps in Tanzania, Rwanda, DRC, and Uganda that received
    refugees again in 2015. By the time Pierre Nkurunziza's bid for a
    third term in April 2015 triggered the new crisis, entire communities
    had lived the routes to Mahama, Nyarugusu, and Nduta within living
    memory. In many cases, the 2015 refugees were children or grandchildren
    of people who had fled on the same corridors in the 1990s.

    This is among the strongest empirical cases for the informed-prior
    assumption: prior displacement experience was recent, salient, and
    applicable to the same geographic destinations.

      Q_TABLE[state][action] = max(0, km_progress) × PRIOR_SCALE
                                × exp(-dist_to_camp / LOCALITY_DECAY_KM)

    LOCALITY_DECAY_KM = 150: Burundi is one of Africa's smallest countries
    (~50,000 km²); routes average ~50km. Even distant communities (e.g.,
    Cankuzo, 300km+ to Mahama) had route knowledge through prior war
    displacement. Decay is set narrow to reflect precision of local
    knowledge — communities knew specific border crossing points.

  Shared Q-table:
    All agents update a single global Q-table. ~100k agents × ~200
    simulation days = ~20M Bellman updates → rapid convergence.

    Simplifying assumption: shared table implies collective intelligence.
    Individual tables + communication model (ego-network sharing
    calibrated to Burundi 2015 mobile penetration ~20%, GSMA data)
    is the architecturally correct extension and remains future work.

  Score Augmentation:
    Q-values bias flee's getEndPointScore via a capped multiplier:
      score × (1 + q_val / Q_MAX)

  Bellman Reward (pure geographic, no deficit):
    R = (km_before - km_after) × PROG_SCALE
      + ARRIVAL_BONUS  if endpoint is camp
      - CONFLICT_PENALTY × conflict_intensity

  Communication Layer (disabled; set LONG_RANGE_PROB > 0 to enable):
    Phone owners share Q-value updates with a contact elsewhere.
    Burundi 2015: ~20% mobile penetration (GSMA estimate).
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
PROG_SCALE       = 0.02
ARRIVAL_BONUS    = 10.0
CONFLICT_PENALTY = 3.0
Q_MAX            = 50.0

# ── Prior knowledge hyperparameters ───────────────────────────────────────────
# Burundi: small country, dense displacement history; extremely strong
# community route knowledge from 1993-2005 civil war to the same camps.
# Short routes → narrow locality decay (150km).
PRIOR_SCALE       = 0.05
LOCALITY_DECAY_KM = 150.0

# ── Communication hyperparameters (disabled) ───────────────────────────────────
PHONE_PENETRATION = 0.20   # Burundi 2015 ~20% (GSMA estimate)
LONG_RANGE_PROB   = 0.00
LONG_RANGE_WEIGHT = 0.30

# ── Module-level state ─────────────────────────────────────────────────────────
Q_TABLE:       dict = {}
CAMP_DISTANCE: dict = {}
NEAREST_CAMP:  dict = {}
ADJ:           dict = {}
AGENT_COMM:    dict = {}

_LM         = None
_D          = None
_CAMP_NAMES = []
_patched    = False


def _col(reader, *candidates):
    for fn in reader.fieldnames:
        if fn.strip().replace('"', '') in candidates:
            return fn
    raise KeyError(f"None of {candidates} found in {reader.fieldnames}")


def _build_distances(input_csv_directory: str):
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
    Pre-populate Q_TABLE from Dijkstra distances.

    Encodes community route knowledge from Burundi's 1993-2005 civil
    war: same camps (Nyarugusu, Nduta in Tanzania; Mahama in Rwanda;
    Nakivale in Uganda; Lusenda in DRC) received refugees then and again
    in 2015. Near-camp communities have precise, strong priors;
    LOCALITY_DECAY_KM = 150km reflects the country's small scale.
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
    print(f"[Q-Flee Burundi] Q_TABLE prior: {n_entries} entries "
          f"(PRIOR_SCALE={PRIOR_SCALE}, LOCALITY_DECAY={LOCALITY_DECAY_KM}km)",
          file=sys.stderr)


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


def _get_comm_strength(agent) -> float:
    aid = id(agent)
    if aid not in AGENT_COMM:
        AGENT_COMM[aid] = (random.uniform(0.3, 0.8)
                           if random.random() < PHONE_PENETRATION
                           else 0.0)
    return AGENT_COMM[aid]


def set_runtime_refs(lm, d, camp_names):
    global _LM, _D, _CAMP_NAMES
    _LM         = lm
    _D          = d
    _CAMP_NAMES = list(camp_names)
    print(f"[Q-Flee Burundi] runtime refs set ({len(_CAMP_NAMES)} camps).",
          file=sys.stderr)


def setup(input_csv_directory: str) -> None:
    global CAMP_DISTANCE, NEAREST_CAMP, ADJ, Q_TABLE, _patched

    if _patched:
        return

    CAMP_DISTANCE, NEAREST_CAMP, ADJ = _build_distances(input_csv_directory)
    print("[Q-Flee Burundi] Dijkstra done.", file=sys.stderr)

    Q_TABLE = {}
    _init_q_table()

    from flee import moving

    _orig_score  = moving.getEndPointScore
    _orig_select = moving.selectRoute

    def _q_score(agent, endpoint, time):
        base = _orig_score(agent, endpoint, time)
        if agent.location.camp:
            return base
        q_val = Q_TABLE.get(agent.location.name, {}).get(endpoint.name, 0.0)
        if q_val > 0.0:
            base *= (1.0 + q_val / Q_MAX)
        return base

    def _q_select(a, time, debug=False, return_all_routes=False,
                  system2_active=False):
        route = _orig_select(a, time, debug, return_all_routes, system2_active)

        if return_all_routes or not route or a.location.camp:
            return route

        first_hop = route[0]
        state     = a.location.name
        INF       = float("inf")

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

        Q_TABLE.setdefault(state, {})
        old_q   = Q_TABLE[state].get(first_hop, 0.0)
        nxt     = Q_TABLE.get(first_hop, {})
        max_nxt = max(nxt.values()) if nxt else 0.0
        Q_TABLE[state][first_hop] = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * max_nxt - old_q
        )

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
    print("[Q-Flee Burundi] shared Q-table + informed prior patched.",
          file=sys.stderr)
