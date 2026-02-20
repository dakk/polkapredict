#!/usr/bin/env python3
"""Polkadot NPoS Election Prediction.

Fetches staking data from the Polkadot Asset Hub (via RPC) and Subscan API
(for validator names), then estimates validator backing.

Default mode: Ranks waiting validators by estimated backing.
  Fetches all nominators and ledgers from the Asset Hub in bulk,
  then calculates backing = self_stake + sum(nominator_active / target_count).

Usage:
    source ~/venv/bin/activate
    python polkadot_election_prediction.py
    python polkadot_election_prediction.py --address <VALIDATOR_STASH>
    python polkadot_election_prediction.py --use-subscan --max-validators 10
"""

import urllib.request
import urllib.error
import json
import time
import argparse
import sys
import os
from collections import defaultdict
from datetime import datetime, timezone

# ── Configuration ───────────────────────────────────────────────────────

CHAIN_CONFIG = {
    "polkadot": {
        "api_base": "https://polkadot.api.subscan.io",
        "rpc": "wss://polkadot-asset-hub-rpc.polkadot.io",
        "decimals": 10_000_000_000,  # 1 DOT = 10^10 planck
        "token": "DOT",
        "seats": 300,
    },
    "kusama": {
        "api_base": "https://kusama.api.subscan.io",
        "rpc": "wss://kusama-asset-hub-rpc.polkadot.io",
        "decimals": 1_000_000_000_000,  # 1 KSM = 10^12 planck
        "token": "KSM",
        "seats": 1000,
    },
}

API_BASE = CHAIN_CONFIG["polkadot"]["api_base"]
ASSET_HUB_RPC = CHAIN_CONFIG["polkadot"]["rpc"]
DOT_DECIMALS = CHAIN_CONFIG["polkadot"]["decimals"]
TOKEN = CHAIN_CONFIG["polkadot"]["token"]
DEFAULT_SEATS = CHAIN_CONFIG["polkadot"]["seats"]
PAGE_SIZE = 100
MAX_RETRIES = 5
RETRY_DELAY = 5

config = {"delay": 0.5, "chain": "polkadot"}  # seconds between API calls


# ── NPoS Algorithm (Sequential Phragmen) ───────────────────────────────
# Adapted from https://github.com/paritytech/consensus/blob/master/NPoS/npos.py

class Edge:
    def __init__(self, nominator_id, validator_id):
        self.nominator_id = nominator_id
        self.validator_id = validator_id
        self.load = 0
        self.weight = 0
        self.candidate = None


class Nominator:
    def __init__(self, nominator_id, budget, targets):
        self.nominator_id = nominator_id
        self.budget = budget
        self.edges = [Edge(self.nominator_id, vid) for vid in targets]
        self.load = 0


class Candidate:
    def __init__(self, validator_id, index):
        self.validator_id = validator_id
        self.valindex = index
        self.approval_stake = 0
        self.backed_stake = 0
        self.elected = False
        self.score = 0
        self.scoredenom = 0


def setuplists(votelist):
    nomlist = [Nominator(v[0], v[1], v[2]) for v in votelist]
    candidate_dict = {}
    candidate_array = []
    num_candidates = 0
    for nom in nomlist:
        for edge in nom.edges:
            vid = edge.validator_id
            if vid in candidate_dict:
                edge.candidate = candidate_array[candidate_dict[vid]]
            else:
                candidate_dict[vid] = num_candidates
                c = Candidate(vid, num_candidates)
                candidate_array.append(c)
                edge.candidate = c
                num_candidates += 1
    return nomlist, candidate_array


def calculate_approval(nomlist):
    for nom in nomlist:
        for edge in nom.edges:
            edge.candidate.approval_stake += nom.budget


def seq_phragmen(votelist, num_to_elect):
    nomlist, candidates = setuplists(votelist)
    calculate_approval(nomlist)

    elected_candidates = []
    for round_num in range(num_to_elect):
        for c in candidates:
            if not c.elected:
                if c.approval_stake > 0:
                    c.score = 1 / c.approval_stake
                else:
                    c.score = float("inf")
        for nom in nomlist:
            for edge in nom.edges:
                if not edge.candidate.elected:
                    if edge.candidate.approval_stake > 0:
                        edge.candidate.score += (
                            nom.budget * nom.load / edge.candidate.approval_stake
                        )

        best_candidate = None
        best_score = float("inf")
        for c in candidates:
            if not c.elected and c.score < best_score:
                best_score = c.score
                best_candidate = c

        if best_candidate is None:
            break

        best_candidate.elected = True
        best_candidate.electedpos = round_num
        elected_candidates.append(best_candidate)

        for nom in nomlist:
            for edge in nom.edges:
                if edge.candidate == best_candidate:
                    edge.load = best_candidate.score - nom.load
                    nom.load = best_candidate.score

    for c in elected_candidates:
        c.backed_stake = 0

    for nom in nomlist:
        for edge in nom.edges:
            if nom.load > 0.0:
                edge.weight = nom.budget * edge.load / nom.load
                edge.candidate.backed_stake += edge.weight
            else:
                edge.weight = 0

    return nomlist, elected_candidates


def equalise(nom, tolerance):
    elected_edges = [e for e in nom.edges if e.candidate.elected]
    if len(elected_edges) < 2:
        return 0.0

    stake_used = sum(e.weight for e in elected_edges)
    backed_stakes = [e.candidate.backed_stake for e in elected_edges]
    backing_stakes = [
        e.candidate.backed_stake for e in elected_edges if e.weight > 0.0
    ]

    if len(backing_stakes) > 0:
        difference = max(backing_stakes) - min(backed_stakes)
        difference += nom.budget - stake_used
        if difference < tolerance:
            return difference
    else:
        difference = nom.budget

    for edge in nom.edges:
        edge.candidate.backed_stake -= edge.weight
        edge.weight = 0

    elected_edges.sort(key=lambda x: x.candidate.backed_stake)
    cumulative = 0
    last_index = len(elected_edges) - 1

    for i in range(len(elected_edges)):
        backed = elected_edges[i].candidate.backed_stake
        if backed * i - cumulative > nom.budget:
            last_index = i - 1
            break
        cumulative += backed

    last_stake = elected_edges[last_index].candidate.backed_stake
    ways_to_split = last_index + 1
    excess = nom.budget + cumulative - last_stake * ways_to_split

    for edge in elected_edges[:ways_to_split]:
        edge.weight = excess / ways_to_split + last_stake - edge.candidate.backed_stake
        edge.candidate.backed_stake += edge.weight

    return difference


def equalise_all(nomlist, max_iterations, tolerance):
    for _ in range(max_iterations):
        max_diff = 0
        for nom in nomlist:
            diff = equalise(nom, tolerance)
            max_diff = max(diff, max_diff)
        if max_diff < tolerance:
            return


def seq_phragmen_with_equalise(votelist, num_to_elect):
    nomlist, elected = seq_phragmen(votelist, num_to_elect)
    equalise_all(nomlist, 10, 0)
    return nomlist, elected


# ── Subscan API Client ─────────────────────────────────────────────────

def api_request(endpoint, payload):
    url = f"{API_BASE}{endpoint}"
    data = json.dumps(payload).encode()
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read())
            if result.get("code") == 10008:
                wait = RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue
            if result.get("code") != 0:
                print(f"\n  API error: {result.get('message', 'unknown')} (code {result.get('code')})")
                return None
            return result.get("data", {})
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  Request failed after {MAX_RETRIES} attempts: {e}")
                return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  Request failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def fetch_all_pages(endpoint, base_payload, desc="items"):
    all_items = []
    page = 0
    while True:
        if page > 0:
            time.sleep(config["delay"])
        payload = {**base_payload, "row": PAGE_SIZE, "page": page}
        data = api_request(endpoint, payload)
        if not data:
            break
        items = data.get("list") or []
        total_count = data.get("count", 0)
        all_items.extend(items)
        if len(items) < PAGE_SIZE or (total_count and len(all_items) >= total_count):
            break
        page += 1
    return all_items


# ── Validator Name Extraction ──────────────────────────────────────────

def get_validator_name(validator):
    display = validator.get("stash_account_display", {})
    people = display.get("people", {})
    if "display" in people:
        return people["display"]
    if "parent" in people:
        parent = people["parent"]
        return f"{parent.get('display', '')}:{parent.get('sub_symbol', '')}"
    return display.get("address", "unknown")[:20] + "..."


# ── RPC Data Fetching (Asset Hub) ─────────────────────────────────────

def _rpc_connect(rpc_url):
    """Create a SubstrateInterface connection and initialize runtime."""
    from substrateinterface import SubstrateInterface

    substrate = SubstrateInterface(url=rpc_url, auto_discover=False)
    head = substrate.rpc_request("chain_getHead", [])["result"]
    substrate.init_runtime(block_hash=head)
    return substrate, head


def _rpc_query_map_with_retry(rpc_url, head, pallet, storage, params=None,
                               max_retries=3, retry_delay=5):
    """Run query_map with automatic reconnect on connection errors.

    Yields (key, value) pairs. If the connection drops mid-iteration,
    reconnects and restarts from the beginning (skipping already-seen keys).
    """
    seen_keys = {}  # key -> value, preserves results across retries
    for attempt in range(max_retries):
        try:
            from substrateinterface import SubstrateInterface

            substrate = SubstrateInterface(url=rpc_url, auto_discover=False)
            substrate.init_runtime(block_hash=head)
            call_params = params or []
            result = substrate.query_map(
                pallet, storage, call_params, block_hash=head
            )
            for key, value in result:
                k = key.value if hasattr(key, "value") else key
                if k not in seen_keys:
                    seen_keys[k] = value
                    yield k, value
            substrate.close()
            return  # completed successfully
        except (ConnectionError, BrokenPipeError, OSError, Exception) as e:
            err_name = type(e).__name__
            # Don't retry on non-connection errors
            if err_name in ("ValueError", "TypeError", "KeyError", "AttributeError"):
                raise
            if attempt < max_retries - 1:
                print(f"\n    Connection error ({err_name}), "
                      f"reconnecting in {retry_delay}s... "
                      f"({len(seen_keys)} entries preserved)")
                time.sleep(retry_delay)
            else:
                print(f"\n    Failed after {max_retries} attempts: {e}")
                raise


def fetch_staking_data_rpc(rpc_url):
    """Fetch all nominator and staking data from Polkadot Asset Hub via RPC.

    Returns:
        nominator_targets: dict of nom_addr -> list of target validator addrs
        stake_active: dict of stash_addr -> active stake in planck
        active_era_validators: set of active validator addresses
    """
    substrate, head = _rpc_connect(rpc_url)

    block_num = int(
        substrate.rpc_request("chain_getHeader", [head])["result"]["number"], 16
    )
    active_era = substrate.query("Staking", "ActiveEra", block_hash=head)
    era_idx = active_era.value["index"] if active_era.value else None
    print(f"  Asset Hub block: {block_num}, active era: {era_idx}")
    substrate.close()

    # Get active validator set from ErasStakersOverview
    active_era_validators = set()
    if era_idx is not None:
        print("  Fetching active validator set...")
        for account, overview in _rpc_query_map_with_retry(
            rpc_url, head, "Staking", "ErasStakersOverview", [era_idx]
        ):
            active_era_validators.add(account)
        print(f"    {len(active_era_validators)} active validators")

    # Iterate all nominators
    print("  Fetching all nominators...")
    nominator_targets = {}
    start = time.time()
    count = 0
    for account, nominations in _rpc_query_map_with_retry(
        rpc_url, head, "Staking", "Nominators"
    ):
        nominator_targets[account] = nominations.value["targets"]
        count += 1
        if count % 5000 == 0:
            elapsed = time.time() - start
            print(f"    {count} nominators ({elapsed:.0f}s)...")
    elapsed = time.time() - start
    print(f"    {count} nominators fetched in {elapsed:.0f}s")

    # Iterate all staking ledgers
    print("  Fetching all staking ledgers...")
    stake_active = {}
    start = time.time()
    count = 0
    for controller, ledger in _rpc_query_map_with_retry(
        rpc_url, head, "Staking", "Ledger"
    ):
        stash = ledger.value["stash"]
        active = ledger.value["active"]
        stake_active[stash] = active
        count += 1
        if count % 5000 == 0:
            elapsed = time.time() - start
            print(f"    {count} ledgers ({elapsed:.0f}s)...")
    elapsed = time.time() - start
    print(f"    {count} ledgers fetched in {elapsed:.0f}s")

    return nominator_targets, stake_active, active_era_validators


def build_ranking_rpc(validator_names, rpc_url):
    """Build waiting validator ranking using Asset Hub RPC data.

    Args:
        validator_names: dict of addr -> name (from Subscan)
        rpc_url: Asset Hub WebSocket URL
    """
    print(f"\n[2/3] Fetching staking data from Asset Hub...")
    nominator_targets, stake_active, active_validators = fetch_staking_data_rpc(
        rpc_url
    )

    # Build reverse index: validator_addr -> set of nominator_addrs
    print(f"\n[3/3] Calculating estimated backing...")
    validator_nominators = defaultdict(set)
    for nom_addr, targets in nominator_targets.items():
        for target in targets:
            validator_nominators[target].add(nom_addr)

    # Get all registered validators (anyone who has nominators or is in active set)
    all_validators = set(validator_nominators.keys()) | active_validators

    # Calculate estimated backing for each validator
    validator_info = {}
    for addr in all_validators:
        is_active = addr in active_validators
        self_stake = stake_active.get(addr, 0)
        nom_backing = 0.0
        for nom_addr in validator_nominators.get(addr, set()):
            nom_active = stake_active.get(nom_addr, 0)
            target_count = len(nominator_targets.get(nom_addr, []))
            if nom_active > 0 and target_count > 0:
                nom_backing += (nom_active / DOT_DECIMALS) / target_count

        name = validator_names.get(addr, addr[:20] + "...")
        validator_info[addr] = {
            "name": name,
            "bonded_owner": self_stake,
            "is_active": is_active,
            "estimated_backing": self_stake / DOT_DECIMALS + nom_backing,
            "nominator_backing": nom_backing,
        }

    return validator_info


def lookup_single_rpc(address, validator_names, rpc_url):
    """Lookup a single validator using Asset Hub RPC data."""
    print(f"\n[2/3] Fetching staking data from Asset Hub...")
    nominator_targets, stake_active, active_validators = fetch_staking_data_rpc(
        rpc_url
    )

    name = validator_names.get(address, address[:20] + "...")
    is_active = address in active_validators
    self_stake = stake_active.get(address, 0)
    status = "ACTIVE" if is_active else "WAITING"

    # Find all nominators that target this address
    nom_backing = 0.0
    nom_count = 0
    for nom_addr, targets in nominator_targets.items():
        if address in targets:
            nom_count += 1
            nom_active = stake_active.get(nom_addr, 0)
            if nom_active > 0 and len(targets) > 0:
                nom_backing += (nom_active / DOT_DECIMALS) / len(targets)

    target_backing = self_stake / DOT_DECIMALS + nom_backing

    # Build quick ranking of all waiting validators using same data
    print(f"\n[3/3] Calculating rankings...")
    validator_nominators = defaultdict(set)
    for nom_addr, targets in nominator_targets.items():
        for target in targets:
            validator_nominators[target].add(nom_addr)

    all_backings = []
    all_validators = set(validator_nominators.keys()) | active_validators
    for addr in all_validators:
        if addr in active_validators:
            continue  # skip active for ranking
        v_self = stake_active.get(addr, 0) / DOT_DECIMALS
        v_nom = 0.0
        for nom_addr in validator_nominators.get(addr, set()):
            n_active = stake_active.get(nom_addr, 0)
            n_targets = len(nominator_targets.get(nom_addr, []))
            if n_active > 0 and n_targets > 0:
                v_nom += (n_active / DOT_DECIMALS) / n_targets
        v_name = validator_names.get(addr, addr[:20] + "...")
        all_backings.append((v_name, addr, v_self + v_nom))

    all_backings.sort(key=lambda x: x[2], reverse=True)

    # Find target's position
    target_rank = None
    for i, (_, addr, _) in enumerate(all_backings):
        if addr == address:
            target_rank = i + 1
            break

    # Get minimum active validator backing
    min_active_backing = None
    for addr in active_validators:
        v_self = stake_active.get(addr, 0) / DOT_DECIMALS
        v_nom = 0.0
        for nom_addr in validator_nominators.get(addr, set()):
            n_active = stake_active.get(nom_addr, 0)
            n_targets = len(nominator_targets.get(nom_addr, []))
            if n_active > 0 and n_targets > 0:
                v_nom += (n_active / DOT_DECIMALS) / n_targets
        total = v_self + v_nom
        if min_active_backing is None or total < min_active_backing:
            min_active_backing = total

    # Display results
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"  {address}")
    print(f"{'=' * 70}")
    print(f"  Status:              {status}")
    print(f"  Self-stake:          {self_stake / DOT_DECIMALS:>14,.0f} {TOKEN}")
    print(f"  Nominator backing:   {nom_backing:>14,.0f} {TOKEN}")
    print(f"  Estimated total:     {target_backing:>14,.0f} {TOKEN}")
    print(f"  Nominators:          {nom_count:>14}")

    if target_rank:
        overall_pos = len(active_validators) + target_rank
        print(f"\n  Waiting rank:        {target_rank} / {len(all_backings)}")
        print(f"  Overall position:    ~{overall_pos}")

    if min_active_backing:
        gap = min_active_backing - target_backing
        print(f"\n  Last active backing: {min_active_backing:>14,.0f} {TOKEN}")
        if gap > 0:
            print(f"  Gap to active set:   {gap:>14,.0f} {TOKEN}")
        else:
            print(f"  Above active min by: {-gap:>14,.0f} {TOKEN}")

    # Show nearby validators
    if target_rank and len(all_backings) > 1:
        start_idx = max(0, target_rank - 4)
        end_idx = min(len(all_backings), target_rank + 3)
        print(f"\n  Nearby validators:")
        print(f"  {'Rank':<6}{'Name':<40}{'Est. Backing':>14}")
        print(f"  {'-' * 6}{'-' * 40}{'-' * 14}")
        for idx in range(start_idx, end_idx):
            v_name, v_addr, v_est = all_backings[idx]
            marker = " >>>" if v_addr == address else "    "
            print(f"  {idx + 1:<6}{v_name:<40}{v_est:>12,.0f}{marker}")


# ── Subscan Data Fetching ─────────────────────────────────────────────

def fetch_active_validators():
    print("[1/3] Fetching active validators...")
    validators = fetch_all_pages(
        "/api/scan/staking/validators",
        {"order": "asc", "order_field": "rank_validator"},
        desc="active validators",
    )
    print(f"  Total active validators: {len(validators)}")
    return validators


def fetch_waiting_validators():
    print("  Fetching waiting validators...")
    validators = fetch_all_pages(
        "/api/scan/staking/waiting",
        {"order": "desc", "order_field": "bonded_owner"},
        desc="waiting validators",
    )
    print(f"  Total waiting validators: {len(validators)}")
    return validators


def fetch_nominators_for_validator(validator_address):
    return fetch_all_pages(
        "/api/scan/staking/nominators",
        {"address": validator_address, "order": "desc", "order_field": "bonded"},
        desc="nominators",
    )


def fetch_nominator_vote_info(nom_address):
    """Fetch a nominator's bonded amount and active validator target count.

    The /voted endpoint returns active validators the nominator stakes with.
    The 'bonded' field inside each entry is the nominator's total bonded amount.
    Returns (bonded_planck, active_target_count).
    """
    data = api_request(
        "/api/scan/staking/voted",
        {"address": nom_address, "row": 100, "page": 0},
    )
    if not data:
        return 0, 0
    voted_list = data.get("list") or []
    if voted_list:
        # bonded is the nominator's total bonded, same in every entry
        bonded = int(voted_list[0].get("bonded", "0") or "0")
        return bonded, len(voted_list)
    # Nominator might only back waiting validators (no active targets)
    # Fall back to /nominator endpoint for bonded amount
    data = api_request("/api/scan/staking/nominator", {"address": nom_address})
    if data:
        bonded = int(data.get("bonded", "0") or "0")
        return bonded, 0
    return 0, 0


def get_nominator_address(nom):
    """Extract nominator address from API response entry."""
    nom_display = nom.get("stash_account_display", {})
    if not nom_display:
        nom_display = nom.get("account_display", {})
    return (nom_display or {}).get("address", "")


# ── Ranking Mode (waiting validators) ─────────────────────────────────

def build_waiting_ranking(waiting_validators):
    """Fetch actual nominator data and estimate backing for each waiting validator.

    Phase 1: Fetch nominator addresses for each waiting validator.
    Phase 2: For each unique nominator, fetch their bonded amount and target count.
    Phase 3: Estimate backing = self_stake + sum(nominator_bonded / target_count).
    """
    validator_info = {}
    validator_noms = {}  # val_addr -> set of nom_addrs

    total = len(waiting_validators)

    # Phase 1: Fetch nominator addresses for each waiting validator
    print(f"\n[2/3] Fetching nominator lists for {total} waiting validators...")
    for i, v in enumerate(waiting_validators):
        addr = v.get("stash_account_display", {}).get("address", "")
        if not addr:
            continue
        name = get_validator_name(v)
        bonded_owner = int(v.get("bonded_owner", "0") or "0")
        count_noms = int(v.get("count_nominators", 0) or 0)

        validator_info[addr] = {
            "name": name,
            "bonded_owner": bonded_owner,
            "is_active": False,
        }

        print(
            f"  [{i + 1}/{total}] {name[:30]:<30s} ({count_noms} noms)",
            end="",
            flush=True,
        )

        if count_noms == 0:
            validator_noms[addr] = set()
            print(" skipped")
            continue

        time.sleep(config["delay"])
        nominators = fetch_nominators_for_validator(addr)
        noms = set()
        for nom in nominators:
            nom_address = get_nominator_address(nom)
            if nom_address:
                noms.add(nom_address)

        validator_noms[addr] = noms
        print(f" -> {len(noms)} nominators")

    # Collect unique nominator addresses and count waiting targets per nominator
    all_nom_addrs = set()
    nominator_waiting_count = defaultdict(int)  # how many scanned waiting vals each nom backs
    for val_addr, noms in validator_noms.items():
        for nom_addr in noms:
            all_nom_addrs.add(nom_addr)
            nominator_waiting_count[nom_addr] += 1

    # Phase 2: For each unique nominator, fetch bonded + active target count
    # The /voted endpoint returns only active validator targets, so
    # total_targets = active_targets + waiting_targets_seen_in_scan
    nominator_cache = {}  # nom_addr -> (bonded_planck, active_target_count)
    num_noms = len(all_nom_addrs)
    print(f"\n[3/3] Resolving {num_noms} unique nominators' stake data...")
    start_phase2 = time.time()

    for i, nom_addr in enumerate(all_nom_addrs):
        time.sleep(config["delay"])
        bonded, active_targets = fetch_nominator_vote_info(nom_addr)
        nominator_cache[nom_addr] = (bonded, active_targets)

        done = i + 1
        if done % 100 == 0 or done == num_noms:
            elapsed = time.time() - start_phase2
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (num_noms - done) / rate if rate > 0 else 0
            print(
                f"    {done}/{num_noms} resolved "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    # Phase 3: Calculate estimated backing for each validator
    # Per-nominator contribution = bonded / (active_targets + waiting_targets_seen)
    for val_addr, info in validator_info.items():
        self_stake = info["bonded_owner"] / DOT_DECIMALS
        nom_backing = 0.0
        noms = validator_noms.get(val_addr, set())
        for nom_addr in noms:
            bonded, active_targets = nominator_cache.get(nom_addr, (0, 0))
            waiting_targets = nominator_waiting_count.get(nom_addr, 1)
            total_targets = active_targets + waiting_targets
            if total_targets > 0 and bonded > 0:
                nom_backing += (bonded / DOT_DECIMALS) / total_targets
        info["estimated_backing"] = self_stake + nom_backing
        info["nominator_backing"] = nom_backing

    return validator_info


def display_ranking(validator_info):
    """Display waiting validators ranked by estimated backing."""
    waiting = [
        (addr, info)
        for addr, info in validator_info.items()
        if not info.get("is_active", False)
    ]
    waiting.sort(key=lambda x: x[1]["estimated_backing"], reverse=True)

    print(f"\n{'=' * 115}")
    print(f"  WAITING VALIDATORS RANKED BY ESTIMATED BACKING")
    print(f"{'=' * 115}")

    header = (
        f" {'Rank':<6}{'Validator Name':<42}"
        f"{'Self-Stake':>14}{'Nom Backing':>16}{'Total':>16}  {'Address'}"
    )
    print(f"\n{header}")
    print(f" {'-' * 6}{'-' * 42}{'-' * 14}{'-' * 16}{'-' * 16}  {'-' * 48}")

    for rank, (addr, info) in enumerate(waiting, 1):
        name = info["name"]
        self_dot = info["bonded_owner"] / DOT_DECIMALS
        nom_dot = info["nominator_backing"]
        total_dot = info["estimated_backing"]
        print(
            f" {rank:<6}{name:<42}{self_dot:>12,.0f}{nom_dot:>14,.0f}"
            f"{total_dot:>16,.0f}  {addr}"
        )

    if waiting:
        backings = [info["estimated_backing"] for _, info in waiting]
        print(f"\n--- Summary ---")
        print(f"  Total waiting validators: {len(waiting)}")
        print(f"  Min estimated backing:  {min(backings):>14,.0f} {TOKEN}")
        print(f"  Max estimated backing:  {max(backings):>14,.0f} {TOKEN}")
        print(f"  Avg estimated backing:  {sum(backings) / len(backings):>14,.0f} {TOKEN}")


# ── Full Election Mode (Phragmen) ─────────────────────────────────────

def build_nomination_graph(all_validators):
    """Build full nomination graph for Phragmen election (active + waiting)."""
    validator_info = {}
    nominator_budgets = {}
    nominator_targets = defaultdict(set)

    active_validators = []
    waiting_validators = []
    for v in all_validators:
        if v.get("rank_validator"):
            active_validators.append(v)
        else:
            waiting_validators.append(v)

    total = len(all_validators)
    print(
        f"[2/4] Building nomination graph ({len(active_validators)} active + "
        f"{len(waiting_validators)} waiting)..."
    )

    # For ACTIVE validators: fetch individual nominators (bonded is non-zero)
    for i, v in enumerate(active_validators):
        addr = v.get("stash_account_display", {}).get("address", "")
        if not addr:
            continue

        name = get_validator_name(v)
        bonded_owner = int(v.get("bonded_owner", "0") or "0")
        count_noms = int(v.get("count_nominators", 0) or 0)

        validator_info[addr] = {
            "name": name,
            "bonded_owner": bonded_owner,
            "is_active": True,
        }

        print(
            f"  [{i + 1}/{total}] {name[:30]:<30s} ({count_noms} noms)",
            end="",
            flush=True,
        )

        if count_noms == 0:
            print(" skipped")
            continue

        time.sleep(config["delay"])
        nominators = fetch_nominators_for_validator(addr)
        fetched = 0

        for nom in nominators:
            nom_address = get_nominator_address(nom)
            if not nom_address:
                continue

            nom_bonded = int(nom.get("bonded", "0") or "0")
            if nom_address in nominator_budgets:
                nominator_budgets[nom_address] = max(
                    nominator_budgets[nom_address], nom_bonded
                )
            else:
                nominator_budgets[nom_address] = nom_bonded

            nominator_targets[nom_address].add(addr)
            fetched += 1

        print(f" -> {fetched} nominators")

    # For WAITING validators: fetch nominator addresses, then resolve via /voted
    idx_offset = len(active_validators)
    waiting_nom_addrs = set()
    waiting_val_noms = {}  # val_addr -> set of nom_addrs

    for i, v in enumerate(waiting_validators):
        addr = v.get("stash_account_display", {}).get("address", "")
        if not addr:
            continue

        name = get_validator_name(v)
        bonded_owner = int(v.get("bonded_owner", "0") or "0")
        count_noms = int(v.get("count_nominators", 0) or 0)

        validator_info[addr] = {
            "name": name,
            "bonded_owner": bonded_owner,
            "is_active": False,
        }

        print(
            f"  [{idx_offset + i + 1}/{total}] {name[:30]:<30s} ({count_noms} noms)",
            end="",
            flush=True,
        )

        if count_noms == 0:
            print(" skipped")
            continue

        time.sleep(config["delay"])
        nominators = fetch_nominators_for_validator(addr)
        noms = set()
        for nom in nominators:
            nom_address = get_nominator_address(nom)
            if nom_address:
                noms.add(nom_address)
                if nom_address not in nominator_budgets:
                    waiting_nom_addrs.add(nom_address)

        waiting_val_noms[addr] = noms
        print(f" -> {len(noms)} nominators")

    # Resolve waiting-only nominators
    unknown = waiting_nom_addrs - set(nominator_budgets.keys())
    if unknown:
        print(f"\n  Resolving {len(unknown)} nominators' stake data...")
        for i, nom_addr in enumerate(unknown):
            time.sleep(config["delay"])
            bonded, targets = fetch_nominator_vote_info(nom_addr)
            nominator_budgets[nom_addr] = bonded
            for t in targets:
                nominator_targets[nom_addr].add(t)
            if (i + 1) % 100 == 0 or i + 1 == len(unknown):
                print(f"    {i + 1}/{len(unknown)} resolved")

    # Add waiting validators' nominator->validator edges
    for val_addr, noms in waiting_val_noms.items():
        for nom_addr in noms:
            nominator_targets[nom_addr].add(val_addr)

    print(
        f"  Scanned {total} validators, "
        f"found {len(nominator_targets)} unique nominators"
    )
    return validator_info, nominator_budgets, nominator_targets


def build_votelist(validator_info, nominator_budgets, nominator_targets):
    votelist = []

    for nom_addr, targets in nominator_targets.items():
        budget = nominator_budgets.get(nom_addr, 0)
        if budget > 0 and targets:
            valid_targets = [t for t in targets if t in validator_info]
            if valid_targets:
                votelist.append((nom_addr, budget / DOT_DECIMALS, valid_targets))

    self_count = 0
    for val_addr, info in validator_info.items():
        if info["bonded_owner"] > 0:
            votelist.append(
                (f"self:{val_addr}", info["bonded_owner"] / DOT_DECIMALS, [val_addr])
            )
            self_count += 1

    return votelist, self_count


# ── Results Display ────────────────────────────────────────────────────

def display_results(elected_candidates, validator_info, active_addresses, num_seats):
    print(f"\n{'=' * 105}")
    print(f"  PREDICTED NPoS ELECTION RESULTS ({num_seats} seats)")
    print(f"{'=' * 105}")

    elected_sorted = sorted(
        elected_candidates, key=lambda c: c.backed_stake, reverse=True
    )

    print(
        f"\n {'Rank':<6}{'Status':<8}{'Validator Name':<42}"
        f"{'Backed Stake (DOT)':>20}  {'Address'}"
    )
    print(f" {'-' * 6}{'-' * 8}{'-' * 42}{'-' * 20}  {'-' * 48}")

    newly_elected = []
    elected_addrs = set()

    for rank, cand in enumerate(elected_sorted, 1):
        addr = cand.validator_id
        elected_addrs.add(addr)
        info = validator_info.get(addr, {})
        name = info.get("name", addr[:20] + "...")
        dot_stake = cand.backed_stake
        was_active = addr in active_addresses

        status = "ACTIVE" if was_active else "NEW"
        if not was_active:
            newly_elected.append((rank, name, dot_stake, addr))

        print(f" {rank:<6}{status:<8}{name:<42}{dot_stake:>18,.2f}  {addr}")

    if newly_elected:
        print(f"\n--- Waiting validators predicted to ENTER the active set ---")
        for rank, name, stake, addr in newly_elected:
            print(f"  Rank {rank}: {name:<40s} {stake:>18,.2f} DOT  {addr}")

    dropped = []
    for addr in active_addresses:
        if addr not in elected_addrs and addr in validator_info:
            name = validator_info[addr].get("name", addr[:20] + "...")
            dropped.append((name, addr))

    if dropped:
        print(f"\n--- Active validators predicted to DROP from the set ---")
        for name, addr in dropped:
            print(f"  {name:<40s} {addr}")

    stakes = [c.backed_stake for c in elected_candidates]
    print(f"\n--- Summary ---")
    print(f"  Total seats:       {num_seats}")
    print(f"  Min backed stake:  {min(stakes):>18,.2f} {TOKEN}")
    print(f"  Max backed stake:  {max(stakes):>18,.2f} {TOKEN}")
    print(f"  Avg backed stake:  {sum(stakes) / len(stakes):>18,.2f} {TOKEN}")
    if newly_elected:
        print(f"  Newly elected:     {len(newly_elected)}")
    if dropped:
        print(f"  Dropped:           {len(dropped)}")


# ── JSON Output ───────────────────────────────────────────────────

def write_election_json(validator_info, mode):
    """Write validator ranking data to data/<chain>/election.json."""
    sorted_vals = sorted(
        validator_info.items(),
        key=lambda x: x[1]["estimated_backing"],
        reverse=True,
    )
    active_count = sum(1 for _, v in sorted_vals if v.get("is_active"))
    waiting_count = sum(1 for _, v in sorted_vals if not v.get("is_active"))

    validators = []
    for rank, (addr, info) in enumerate(sorted_vals, 1):
        validators.append({
            "rank": rank,
            "name": info["name"],
            "address": addr,
            "is_active": info.get("is_active", False),
            "self_stake_dot": round(info["bonded_owner"] / DOT_DECIMALS, 2),
            "nominator_backing_dot": round(info.get("nominator_backing", 0), 2),
            "estimated_total_dot": round(info["estimated_backing"], 2),
        })

    chain = config["chain"]
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", chain, "election.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chain": chain,
        "token": TOKEN,
        "mode": mode,
        "active_validators": active_count,
        "waiting_validators": waiting_count,
        "validators": validators,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON written to {output_path}")


# ── Main ───────────────────────────────────────────────────────────────

def fetch_validator_names():
    """Fetch validator names from Subscan (active + waiting lists)."""
    print("[1/3] Fetching validator names from Subscan...")
    active = fetch_all_pages(
        "/api/scan/staking/validators",
        {"order": "asc", "order_field": "rank_validator"},
        desc="active validators",
    )
    print(f"  Active validators: {len(active)}")

    time.sleep(config["delay"])
    waiting = fetch_all_pages(
        "/api/scan/staking/waiting",
        {"order": "desc", "order_field": "bonded_owner"},
        desc="waiting validators",
    )
    print(f"  Waiting validators: {len(waiting)}")

    names = {}
    for v in active + waiting:
        addr = v.get("stash_account_display", {}).get("address", "")
        if addr:
            names[addr] = get_validator_name(v)
    return names, active, waiting


def main():
    global API_BASE, ASSET_HUB_RPC, DOT_DECIMALS, TOKEN, DEFAULT_SEATS

    parser = argparse.ArgumentParser(
        description="NPoS validator election prediction"
    )
    parser.add_argument(
        "--chain",
        type=str,
        choices=list(CHAIN_CONFIG.keys()),
        default="polkadot",
        help="Chain to analyze (default: polkadot)",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="",
        help="Lookup a single validator address",
    )
    parser.add_argument(
        "--rpc",
        type=str,
        default="",
        help="Asset Hub RPC endpoint (overrides chain default)",
    )
    parser.add_argument(
        "--use-subscan",
        action="store_true",
        default=False,
        help="Use Subscan API instead of RPC (slower, rate-limited)",
    )
    parser.add_argument(
        "--seats",
        type=int,
        default=0,
        help="Seats for Phragmen election mode (0 = chain default)",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        default=False,
        help="Include active validators (Subscan/Phragmen mode only)",
    )
    parser.add_argument(
        "--max-validators",
        type=int,
        default=0,
        help="Limit waiting validators scanned (Subscan mode only, 0 = all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between Subscan API calls (default: 0.5s)",
    )
    args = parser.parse_args()

    # Apply chain config
    chain_cfg = CHAIN_CONFIG[args.chain]
    API_BASE = chain_cfg["api_base"]
    ASSET_HUB_RPC = args.rpc or chain_cfg["rpc"]
    DOT_DECIMALS = chain_cfg["decimals"]
    TOKEN = chain_cfg["token"]
    DEFAULT_SEATS = args.seats or chain_cfg["seats"]
    config["delay"] = args.delay
    config["chain"] = args.chain

    start_time = time.time()
    print(f"=== {args.chain.capitalize()} NPoS Election Prediction ===")
    print()

    if args.use_subscan:
        # ── Subscan mode (legacy, slower) ──
        active_validators = fetch_active_validators()
        active_addresses = set()
        for v in active_validators:
            addr = v.get("stash_account_display", {}).get("address", "")
            if addr:
                active_addresses.add(addr)
        waiting = fetch_waiting_validators()

        if args.address:
            print(f"\n    Mode: Single validator lookup (Subscan)")
            lookup_single_validator(args.address, active_validators, waiting)
        elif not args.include_active:
            print(f"\n    Mode: Estimated backing ranking (Subscan)")
            candidates = list(waiting)
            if args.max_validators > 0:
                candidates = candidates[: args.max_validators]
                print(f"    Limited to {len(candidates)} waiting validators")
            validator_info = build_waiting_ranking(candidates)
            display_ranking(validator_info)
            write_election_json(validator_info, "subscan")
        else:
            print(f"\n    Mode: Full Phragmen election (Subscan)")
            print(f"    Seats: {DEFAULT_SEATS}")
            all_candidates = list(active_validators) + list(waiting)
            if args.max_validators > 0:
                all_candidates = (
                    list(active_validators) + waiting[: args.max_validators]
                )
            validator_info, nominator_budgets, nominator_targets_map = (
                build_nomination_graph(all_candidates)
            )
            print(f"\n[3/4] Building votelist...")
            votelist, self_count = build_votelist(
                validator_info, nominator_budgets, nominator_targets_map
            )
            print(
                f"  {len(nominator_targets_map)} nominators + {self_count} "
                f"self-stakes = {len(votelist)} vote entries"
            )
            num_to_elect = min(DEFAULT_SEATS, len(validator_info))
            print(f"\n[4/4] Running Sequential Phragmen for {num_to_elect} seats...")
            nomlist, elected = seq_phragmen_with_equalise(votelist, num_to_elect)
            display_results(elected, validator_info, active_addresses, num_to_elect)
    else:
        # ── RPC mode (default, faster) ──
        validator_names, _, _ = fetch_validator_names()

        if args.address:
            print(f"\n    Mode: Single validator lookup (RPC)")
            lookup_single_rpc(args.address, validator_names, ASSET_HUB_RPC)
        else:
            print(f"\n    Mode: Full ranking via Asset Hub RPC")
            validator_info = build_ranking_rpc(validator_names, ASSET_HUB_RPC)
            display_ranking(validator_info)
            write_election_json(validator_info, "rpc")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
