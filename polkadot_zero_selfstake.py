#!/usr/bin/env python3
"""Count active Polkadot validators with zero self-stake using Subscan API."""

import urllib.request
import json
import time

API_URL = "https://polkadot.api.subscan.io/api/scan/staking/validators"
PAGE_SIZE = 100
TOTAL_VALIDATORS = 600


def get_validator_name(validator):
    display = validator.get("stash_account_display", {})
    people = display.get("people", {})
    if "display" in people:
        return people["display"]
    if "parent" in people:
        parent = people["parent"]
        return f"{parent.get('display', '')}:{parent.get('sub_symbol', '')}"
    return display.get("address", "unknown")[:20] + "..."


def fetch_validators():
    all_validators = []
    pages = (TOTAL_VALIDATORS + PAGE_SIZE - 1) // PAGE_SIZE

    for page in range(pages):
        if page > 0:
            time.sleep(2)
        payload = json.dumps({
            "order": "asc",
            "order_field": "rank_validator",
            "row": PAGE_SIZE,
            "page": page,
        }).encode()
        req = urllib.request.Request(
            API_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
        validators = result["data"]["list"]
        all_validators.extend(validators)
        print(f"Page {page}: fetched {len(validators)} validators")

    return all_validators


def main():
    validators = fetch_validators()
    # DOT has 10 decimal places
    DOT_DECIMALS = 10_000_000_000
    THRESHOLD = 10_000 * DOT_DECIMALS

    zero_stake = []
    under_10k = []

    for v in validators:
        bonded = int(v["bonded_owner"])
        name = get_validator_name(v)
        address = v["stash_account_display"]["address"]
        dot_amount = bonded / DOT_DECIMALS

        if bonded == 0:
            zero_stake.append((name, address, dot_amount))
        if bonded < THRESHOLD:
            under_10k.append((name, address, dot_amount))

    total = len(validators)
    print(f"\n{'=' * 50}")
    print(f"Total active validators: {total}")
    print(f"Validators with self-stake == 0: {len(zero_stake)} ({len(zero_stake) / total * 100:.1f}%)")
    print(f"Validators with self-stake < 10k DOT: {len(under_10k)} ({len(under_10k) / total * 100:.1f}%)")

    print(f"\n--- Validators with 0 self-stake ---")
    for i, (name, address, _) in enumerate(zero_stake, 1):
        print(f"  {i:3d}. {name:<40s} {address}")

    print(f"\n--- Validators with self-stake > 0 but < 10k DOT ---")
    non_zero_under_10k = [(n, a, d) for n, a, d in under_10k if d > 0]
    for i, (name, address, dot) in enumerate(non_zero_under_10k, 1):
        print(f"  {i:3d}. {name:<40s} {dot:>12,.2f} DOT  {address}")


if __name__ == "__main__":
    main()
