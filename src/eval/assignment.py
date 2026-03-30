"""Deterministic user assignment helpers for offline experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from hashlib import sha256


def validate_split_config(split_config: Mapping[str, int]) -> dict[str, int]:
    """Validate and normalize a percentage-based experiment split config."""

    if not split_config:
        raise ValueError("split_config must define at least one variant.")

    normalized: dict[str, int] = {}
    total = 0
    for variant_name, percentage in split_config.items():
        if not isinstance(percentage, int):
            raise TypeError("Split percentages must be integers.")
        if percentage <= 0:
            raise ValueError("Split percentages must be positive integers.")
        normalized[str(variant_name)] = percentage
        total += percentage

    if total != 100:
        raise ValueError(f"Split percentages must sum to 100. Received {total}.")

    return normalized


def stable_hash_to_bucket(value: str | int, modulo: int = 100) -> int:
    """Map an ID to a deterministic hash bucket in ``[0, modulo)``."""

    if modulo <= 0:
        raise ValueError("modulo must be greater than 0.")

    digest = sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo


def assign_variant(user_id: str | int, split_config: Mapping[str, int]) -> str:
    """Assign one user to a named variant using deterministic hashing."""

    normalized_split = validate_split_config(split_config)
    bucket = stable_hash_to_bucket(user_id, modulo=100)

    cumulative = 0
    for variant_name, percentage in normalized_split.items():
        cumulative += percentage
        if bucket < cumulative:
            return variant_name

    raise RuntimeError("Failed to assign a variant from the provided split_config.")


def assign_users(
    user_ids: Iterable[str | int],
    split_config: Mapping[str, int],
) -> dict[str | int, str]:
    """Assign multiple users to experiment variants deterministically."""

    normalized_split = validate_split_config(split_config)
    return {
        user_id: assign_variant(user_id=user_id, split_config=normalized_split)
        for user_id in user_ids
    }
