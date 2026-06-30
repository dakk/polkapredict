#!/usr/bin/env python3
"""Count active validators with zero self-stake using on-chain data."""

import argparse
import json
import os
from datetime import datetime, timezone

from polkadot_common import (
    CHAIN_CONFIG,
    IDENTITY_CACHE_TTL_DAYS,
    connect,
    fetch_identity,
    load_identity_cache,
    save_identity_cache,
)

RELAY_RPCS = CHAIN_CONFIG["polkadot"]["relay_rpcs"]
ASSET_HUB_RPCS = CHAIN_CONFIG["polkadot"]["asset_hub_rpcs"]
PEOPLE_RPCS = CHAIN_CONFIG["polkadot"]["people_rpcs"]
DOT_DECIMALS = CHAIN_CONFIG["polkadot"]["decimals"]
TOKEN = CHAIN_CONFIG["polkadot"]["token"]
THRESHOLD_AMOUNT = CHAIN_CONFIG["polkadot"]["threshold"]


def get_active_validators():
    """Get active validator set from relay chain Session::Validators."""
    substrate = connect(RELAY_RPCS)
    result = substrate.query("Session", "Validators")
    validators = [str(v) for v in result.value]
    substrate.close()
    return validators


def get_staking_overview(active_set):
    """Get self-stake from Asset Hub ErasStakersOverview for the active era."""
    substrate = connect(ASSET_HUB_RPCS)
    active_era = substrate.query("Staking", "ActiveEra")
    era_index = active_era.value["index"]
    print(f"Active era: {era_index}")

    result = substrate.query_map(
        "Staking", "ErasStakersOverview", params=[era_index], max_results=1000
    )
    stakers = {}
    for key, value in result:
        addr = str(key.value)
        stakers[addr] = value.value

    substrate.close()

    overview = {}
    for addr in active_set:
        if addr in stakers:
            overview[addr] = stakers[addr]["own"]
        else:
            overview[addr] = 0
    return overview


def get_identities(addresses, chain):
    """Get display names from People chain, using a shared cache with TTL."""
    cache = load_identity_cache(chain)
    now = datetime.now(timezone.utc)
    ttl_seconds = IDENTITY_CACHE_TTL_DAYS * 86400

    stale = [
        addr for addr in addresses
        if addr not in cache
        or (now - datetime.fromisoformat(cache[addr]["cached_at"])).total_seconds() > ttl_seconds
    ]

    if stale:
        print(f"  Fetching {len(stale)} identities from chain ({len(addresses) - len(stale)} cached)...")
        substrate = connect(PEOPLE_RPCS)
        for addr in stale:
            name = fetch_identity(substrate, addr)
            cache[addr] = {"name": name, "cached_at": now.isoformat()}
        substrate.close()
        save_identity_cache(chain, cache)
    else:
        print(f"  All {len(addresses)} identities served from cache.")

    return {addr: cache[addr]["name"] for addr in addresses}


def compute_history_buckets(under_list, total, under_count, threshold):
    """Compute bucket distribution from the under-threshold list."""
    cuts = [0, 0.1, 0.25, 0.5, 0.75, 1.0]
    boundaries = [c * threshold for c in cuts]
    bucket_keys = ["0_to_1k", "1k_to_2500", "2500_to_5k", "5k_to_7500", "7500_to_10k"]

    buckets = {
        "zero": 0,
        "above_10k": total - under_count,
    }
    for key in bucket_keys:
        buckets[key] = 0

    for v in under_list:
        s = v["self_stake_dot"]
        if s == 0:
            buckets["zero"] += 1
        else:
            for i in range(len(boundaries) - 1):
                if s <= boundaries[i + 1]:
                    buckets[bucket_keys[i]] += 1
                    break

    return buckets


def update_history(result, chain):
    """Append today's data point to the history file."""
    history_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", chain, "selfstake_history.json",
    )

    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {
            "chain": chain,
            "token": result.get("token", "DOT"),
            "threshold": result.get("threshold", 10000),
            "data": [],
        }

    date = result["generated_at"][:10]
    under_list = result.get("under_threshold", result.get("under_10k", []))
    under_count = result.get(
        "under_threshold_count",
        result.get("under_10k_count", len(under_list)),
    )
    threshold = result.get("threshold", 10000)

    entry = {
        "date": date,
        "generated_at": result["generated_at"],
        "total_active_validators": result["total_active_validators"],
        "zero_selfstake_count": result["zero_selfstake_count"],
        "under_threshold_count": under_count,
        "buckets": compute_history_buckets(
            under_list, result["total_active_validators"], under_count, threshold,
        ),
    }

    replaced = False
    for i, existing in enumerate(history["data"]):
        if existing["date"] == date:
            history["data"][i] = entry
            replaced = True
            break
    if not replaced:
        history["data"].append(entry)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History updated: {history_path} ({len(history['data'])} data points)")


def main():
    global RELAY_RPCS, ASSET_HUB_RPCS, PEOPLE_RPCS, DOT_DECIMALS, TOKEN, THRESHOLD_AMOUNT

    parser = argparse.ArgumentParser(
        description="Count active validators with zero self-stake"
    )
    parser.add_argument(
        "--chain",
        type=str,
        choices=list(CHAIN_CONFIG.keys()),
        default="polkadot",
        help="Chain to analyze (default: polkadot)",
    )
    args = parser.parse_args()

    chain_cfg = CHAIN_CONFIG[args.chain]
    RELAY_RPCS = chain_cfg["relay_rpcs"]
    ASSET_HUB_RPCS = chain_cfg["asset_hub_rpcs"]
    PEOPLE_RPCS = chain_cfg["people_rpcs"]
    DOT_DECIMALS = chain_cfg["decimals"]
    TOKEN = chain_cfg["token"]
    THRESHOLD_AMOUNT = chain_cfg["threshold"]

    print(f"=== {args.chain.capitalize()} Self-Stake Analysis ===\n")
    print("Fetching active validator set from relay chain...")
    active_set = get_active_validators()
    print(f"Active validators: {len(active_set)}")

    print("Fetching staking overview from Asset Hub...")
    overview = get_staking_overview(active_set)

    print("Fetching identities from People chain...")
    names = get_identities(active_set, args.chain)

    THRESHOLD = THRESHOLD_AMOUNT * DOT_DECIMALS

    zero_stake = []
    under_threshold = []
    above_threshold = []

    for addr in active_set:
        own = overview[addr]
        name = names.get(addr, addr[:20] + "...")
        dot_amount = own / DOT_DECIMALS

        if own == 0:
            zero_stake.append((name, addr, dot_amount))
        if own < THRESHOLD:
            under_threshold.append((name, addr, dot_amount))
        else:
            above_threshold.append((name, addr, dot_amount))

    total = len(active_set)
    print(f"\n{'=' * 50}")
    print(f"Total active validators: {total}")
    print(
        f"Validators with self-stake == 0: {len(zero_stake)} ({len(zero_stake) / total * 100:.1f}%)"
    )
    print(
        f"Validators with self-stake < {THRESHOLD_AMOUNT:,} {TOKEN}: {len(under_threshold)} ({len(under_threshold) / total * 100:.1f}%)"
    )

    print(f"\n--- Validators with 0 self-stake ---")
    for i, (name, address, _) in enumerate(zero_stake, 1):
        print(f"  {i:3d}. {name:<40s} {address}")

    print(f"\n--- Validators with self-stake > 0 but < {THRESHOLD_AMOUNT:,} {TOKEN} ---")
    non_zero_under = [(n, a, d) for n, a, d in under_threshold if d > 0]
    for i, (name, address, dot) in enumerate(non_zero_under, 1):
        print(f"  {i:3d}. {name:<40s} {dot:>12,.2f} {TOKEN}  {address}")

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", args.chain, "selfstake.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chain": args.chain,
        "token": TOKEN,
        "threshold": THRESHOLD_AMOUNT,
        "active_era": overview.get("_era"),
        "total_active_validators": total,
        "zero_selfstake_count": len(zero_stake),
        "under_threshold_count": len(under_threshold),
        "zero_selfstake": [
            {"name": n, "address": a, "self_stake_dot": d} for n, a, d in zero_stake
        ],
        "under_threshold": [
            {"name": n, "address": a, "self_stake_dot": round(d, 2)}
            for n, a, d in under_threshold
        ],
        "above_threshold_count": len(above_threshold),
        "above_threshold": [
            {"name": n, "address": a, "self_stake_dot": round(d, 2)}
            for n, a, d in above_threshold
        ],
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON written to {output_path}")

    update_history(result, args.chain)


if __name__ == "__main__":
    main()
