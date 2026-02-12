#!/usr/bin/env python3
"""Count active Polkadot validators with zero self-stake using on-chain data."""

import json
import os
from datetime import datetime, timezone

from substrateinterface import SubstrateInterface

RELAY_RPC = "wss://rpc.ibp.network/polkadot"
ASSET_HUB_RPC = "wss://sys.ibp.network/asset-hub-polkadot"
PEOPLE_RPC = "wss://sys.ibp.network/people-polkadot"
DOT_DECIMALS = 10_000_000_000


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
    print("Fetching active validator set from relay chain...")
    active_set = get_active_validators()
    print(f"Active validators: {len(active_set)}")

    print("Fetching staking overview from Asset Hub...")
    overview = get_staking_overview(active_set)

    print("Fetching identities from People chain...")
    names = get_identities(active_set)

    THRESHOLD = 10_000 * DOT_DECIMALS

    zero_stake = []
    under_10k = []

    for addr in active_set:
        own = overview[addr]
        name = names.get(addr, addr[:20] + "...")
        dot_amount = own / DOT_DECIMALS

        if own == 0:
            zero_stake.append((name, addr, dot_amount))
        if own < THRESHOLD:
            under_10k.append((name, addr, dot_amount))

    total = len(active_set)
    print(f"\n{'=' * 50}")
    print(f"Total active validators: {total}")
    print(
        f"Validators with self-stake == 0: {len(zero_stake)} ({len(zero_stake) / total * 100:.1f}%)"
    )
    print(
        f"Validators with self-stake < 10k DOT: {len(under_10k)} ({len(under_10k) / total * 100:.1f}%)"
    )

    print(f"\n--- Validators with 0 self-stake ---")
    for i, (name, address, _) in enumerate(zero_stake, 1):
        print(f"  {i:3d}. {name:<40s} {address}")

    print(f"\n--- Validators with self-stake > 0 but < 10k DOT ---")
    non_zero_under_10k = [(n, a, d) for n, a, d in under_10k if d > 0]
    for i, (name, address, dot) in enumerate(non_zero_under_10k, 1):
        print(f"  {i:3d}. {name:<40s} {dot:>12,.2f} DOT  {address}")

    # Write JSON output
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "selfstake.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_era": overview.get("_era"),
        "total_active_validators": total,
        "zero_selfstake_count": len(zero_stake),
        "under_10k_count": len(under_10k),
        "zero_selfstake": [
            {"name": n, "address": a, "self_stake_dot": d} for n, a, d in zero_stake
        ],
        "under_10k": [
            {"name": n, "address": a, "self_stake_dot": round(d, 2)}
            for n, a, d in under_10k
        ],
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON written to {output_path}")


if __name__ == "__main__":
    main()
