#!/usr/bin/env python3
"""Count active validators with zero self-stake using on-chain data."""

import argparse
import json
import os
from datetime import datetime, timezone

from substrateinterface import SubstrateInterface

CHAIN_CONFIG = {
    "polkadot": {
        "relay_rpc": "wss://rpc.ibp.network/polkadot",
        "asset_hub_rpc": "wss://sys.ibp.network/asset-hub-polkadot",
        "people_rpc": "wss://sys.ibp.network/people-polkadot",
        "decimals": 10_000_000_000,  # 1 DOT = 10^10 planck
        "token": "DOT",
        "threshold": 10_000,
    },
    "kusama": {
        "relay_rpc": "wss://rpc.ibp.network/kusama",
        "asset_hub_rpc": "wss://sys.ibp.network/asset-hub-kusama",
        "people_rpc": "wss://sys.ibp.network/people-kusama",
        "decimals": 1_000_000_000_000,  # 1 KSM = 10^12 planck
        "token": "KSM",
        "threshold": 10,
    },
}

RELAY_RPC = CHAIN_CONFIG["polkadot"]["relay_rpc"]
ASSET_HUB_RPC = CHAIN_CONFIG["polkadot"]["asset_hub_rpc"]
PEOPLE_RPC = CHAIN_CONFIG["polkadot"]["people_rpc"]
DOT_DECIMALS = CHAIN_CONFIG["polkadot"]["decimals"]
TOKEN = CHAIN_CONFIG["polkadot"]["token"]
THRESHOLD_AMOUNT = CHAIN_CONFIG["polkadot"]["threshold"]


def get_active_validators():
    """Get active validator set from relay chain Session::Validators."""
    substrate = SubstrateInterface(url=RELAY_RPC)
    result = substrate.query("Session", "Validators")
    validators = [str(v) for v in result.value]
    substrate.close()
    return validators


def get_staking_overview(active_set):
    """Get self-stake from Asset Hub ErasStakersOverview for the active era."""
    substrate = SubstrateInterface(url=ASSET_HUB_RPC)
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

    # Return only active validators
    overview = {}
    for addr in active_set:
        if addr in stakers:
            overview[addr] = stakers[addr]["own"]
        else:
            overview[addr] = 0
    return overview


def get_identities(addresses):
    """Get display names from People chain Identity::IdentityOf."""
    substrate = SubstrateInterface(url=PEOPLE_RPC)
    names = {}
    for addr in addresses:
        try:
            result = substrate.query("Identity", "IdentityOf", params=[addr])
            if result.value:
                info = result.value["info"]
                display = info.get("display", {})
                raw = display.get("Raw")
                if raw:
                    names[addr] = raw
                    continue
        except Exception:
            pass
        # Check if it's a sub-identity
        try:
            result = substrate.query("Identity", "SuperOf", params=[addr])
            if result.value:
                parent_addr, sub_data = result.value
                parent_addr = str(parent_addr)
                sub_name = sub_data.get("Raw", "")
                parent_result = substrate.query(
                    "Identity", "IdentityOf", params=[parent_addr]
                )
                if parent_result.value:
                    parent_display = (
                        parent_result.value["info"].get("display", {}).get("Raw", "")
                    )
                    names[addr] = f"{parent_display}:{sub_name}"
                    continue
        except Exception:
            pass
        names[addr] = addr[:20] + "..."
    substrate.close()
    return names


def main():
    global RELAY_RPC, ASSET_HUB_RPC, PEOPLE_RPC, DOT_DECIMALS, TOKEN, THRESHOLD_AMOUNT

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
    RELAY_RPC = chain_cfg["relay_rpc"]
    ASSET_HUB_RPC = chain_cfg["asset_hub_rpc"]
    PEOPLE_RPC = chain_cfg["people_rpc"]
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
    names = get_identities(active_set)

    THRESHOLD = THRESHOLD_AMOUNT * DOT_DECIMALS

    zero_stake = []
    under_threshold = []

    for addr in active_set:
        own = overview[addr]
        name = names.get(addr, addr[:20] + "...")
        dot_amount = own / DOT_DECIMALS

        if own == 0:
            zero_stake.append((name, addr, dot_amount))
        if own < THRESHOLD:
            under_threshold.append((name, addr, dot_amount))

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

    # Write JSON output
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
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON written to {output_path}")


if __name__ == "__main__":
    main()
