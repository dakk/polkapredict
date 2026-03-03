#!/usr/bin/env python3
"""One-time extraction of historical selfstake data from git commits."""

import json
import os
import subprocess
from collections import OrderedDict


THRESHOLD = 10000  # DOT


def get_commits_with_selfstake():
    """Get all commits that modified selfstake.json, oldest first."""
    result = subprocess.run(
        [
            "git", "log", "--all", "--format=%H", "--reverse", "--",
            "data/selfstake.json", "data/polkadot/selfstake.json",
        ],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    commits = result.stdout.strip().split("\n")
    return [c for c in commits if c]


def load_selfstake_from_commit(commit_hash):
    """Try to load selfstake.json from a given commit."""
    cwd = os.path.dirname(os.path.abspath(__file__))
    for path in ["data/polkadot/selfstake.json", "data/selfstake.json"]:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{path}"],
            capture_output=True, text=True, cwd=cwd,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    return None


def compute_buckets(data):
    """Compute bucket distribution from per-validator data."""
    under_list = data.get("under_threshold", data.get("under_10k", []))
    total = data["total_active_validators"]
    under_count = data.get("under_threshold_count", data.get("under_10k_count", len(under_list)))

    buckets = {
        "zero": 0,
        "0_to_1k": 0,
        "1k_to_2500": 0,
        "2500_to_5k": 0,
        "5k_to_7500": 0,
        "7500_to_10k": 0,
        "above_10k": total - under_count,
    }
    for v in under_list:
        s = v["self_stake_dot"]
        if s == 0:
            buckets["zero"] += 1
        elif s <= 1000:
            buckets["0_to_1k"] += 1
        elif s <= 2500:
            buckets["1k_to_2500"] += 1
        elif s <= 5000:
            buckets["2500_to_5k"] += 1
        elif s <= 7500:
            buckets["5k_to_7500"] += 1
        else:
            buckets["7500_to_10k"] += 1

    return buckets


def main():
    commits = get_commits_with_selfstake()
    print(f"Found {len(commits)} commits with selfstake data")

    history = OrderedDict()

    for commit in commits:
        data = load_selfstake_from_commit(commit)
        if not data:
            print(f"  {commit[:7]}: could not load selfstake data, skipping")
            continue

        date = data["generated_at"][:10]
        under_count = data.get(
            "under_threshold_count",
            data.get("under_10k_count", len(data.get("under_threshold", data.get("under_10k", [])))),
        )

        entry = {
            "date": date,
            "generated_at": data["generated_at"],
            "total_active_validators": data["total_active_validators"],
            "zero_selfstake_count": data["zero_selfstake_count"],
            "under_threshold_count": under_count,
            "buckets": compute_buckets(data),
        }
        action = "updated" if date in history else "added"
        history[date] = entry
        print(f"  {commit[:7]} ({date}): {action} — zero={entry['zero_selfstake_count']}, under={under_count}")

    output = {
        "chain": "polkadot",
        "token": "DOT",
        "threshold": THRESHOLD,
        "data": list(history.values()),
    }

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "polkadot", "selfstake_history.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(history)} data points to {output_path}")


if __name__ == "__main__":
    main()
