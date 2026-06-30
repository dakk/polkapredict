#!/usr/bin/env python3
"""Shared configuration and helpers for polkadot_zero_selfstake.py and polkadot_election_prediction.py."""

import json
import os
from datetime import datetime, timezone

from substrateinterface import SubstrateInterface

CHAIN_CONFIG = {
    "polkadot": {
        "relay_rpcs": [
            "wss://polkadot.api.onfinality.io/public-ws",
            "wss://rpc.ibp.network/polkadot",
        ],
        "asset_hub_rpcs": [
            "wss://polkadot-asset-hub-rpc.polkadot.io",
            "wss://statemint.api.onfinality.io/public-ws",
            "wss://sys.ibp.network/asset-hub-polkadot",
        ],
        "people_rpcs": [
            "wss://polkadot-people-rpc.polkadot.io",
            "wss://sys.ibp.network/people-polkadot",
        ],
        "decimals": 10_000_000_000,  # 1 DOT = 10^10 planck
        "token": "DOT",
        "threshold": 10_000,
        "seats": 300,
        "api_base": "https://polkadot.api.subscan.io",
    },
    "kusama": {
        "relay_rpcs": [
            "wss://kusama.api.onfinality.io/public-ws",
            "wss://rpc.ibp.network/kusama",
        ],
        "asset_hub_rpcs": [
            "wss://kusama-asset-hub-rpc.polkadot.io",
            "wss://sys.ibp.network/asset-hub-kusama",
        ],
        "people_rpcs": [
            "wss://kusama-people-rpc.polkadot.io",
            "wss://sys.ibp.network/people-kusama",
        ],
        "decimals": 1_000_000_000_000,  # 1 KSM = 10^12 planck
        "token": "KSM",
        "threshold": 10,
        "seats": 1000,
        "api_base": "https://kusama.api.subscan.io",
    },
}

IDENTITY_CACHE_TTL_DAYS = 28

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def identity_cache_path(chain):
    return os.path.join(_BASE_DIR, "data", chain, "identity_cache.json")


def load_identity_cache(chain):
    path = identity_cache_path(chain)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_identity_cache(chain, cache):
    path = identity_cache_path(chain)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def connect(rpcs):
    """Try each RPC endpoint in order; return the first SubstrateInterface that connects."""
    last_exc = None
    for url in rpcs:
        try:
            substrate = SubstrateInterface(url=url)
            print(f"  Connected to {url}")
            return substrate
        except Exception as exc:
            print(f"  {url} unavailable: {exc}")
            last_exc = exc
    raise ConnectionError(f"All RPC endpoints failed. Last error: {last_exc}")


def fetch_identity(substrate, addr):
    """Resolve display name for a single address from an open People chain connection."""
    try:
        result = substrate.query("Identity", "IdentityOf", params=[addr])
        if result.value:
            raw = result.value["info"].get("display", {}).get("Raw")
            if raw:
                return raw
    except Exception:
        pass
    try:
        result = substrate.query("Identity", "SuperOf", params=[addr])
        if result.value:
            parent_addr, sub_data = result.value
            parent_addr = str(parent_addr)
            sub_name = sub_data.get("Raw", "")
            parent_result = substrate.query("Identity", "IdentityOf", params=[parent_addr])
            if parent_result.value:
                parent_display = (
                    parent_result.value["info"].get("display", {}).get("Raw", "")
                )
                return f"{parent_display}:{sub_name}"
    except Exception:
        pass
    return addr[:20] + "..."
