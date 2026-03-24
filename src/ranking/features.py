"""Feature generation for ranking models."""

from __future__ import annotations

import pandas as pd


RANKING_METADATA_COLUMNS = {"user_id", "item_id", "label"}


def build_user_feature_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-user aggregate features from training interactions."""

    user_group = train_df.groupby("user_id")
    user_features = user_group.agg(
        user_train_interaction_count=("item_id", "size"),
        user_train_unique_item_count=("item_id", "nunique"),
        user_train_avg_event_weight=("event_weight", "mean"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    ).reset_index()

    user_features["user_train_days_span"] = (
        user_features["last_timestamp"] - user_features["first_timestamp"]
    ).dt.total_seconds().div(86400.0).fillna(0.0)
    reference_timestamp = train_df["timestamp"].max()
    user_features["user_train_recency_days"] = (
        reference_timestamp - user_features["last_timestamp"]
    ).dt.total_seconds().div(86400.0).fillna(0.0)

    return user_features[
        [
            "user_id",
            "user_train_interaction_count",
            "user_train_unique_item_count",
            "user_train_avg_event_weight",
            "user_train_days_span",
            "user_train_recency_days",
        ]
    ]


def build_item_feature_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-item aggregate features from training interactions."""

    item_features = (
        train_df.groupby("item_id")
        .agg(
            item_popularity_count=("user_id", "size"),
            item_popularity_weighted=("event_weight", "sum"),
        )
        .reset_index()
    )

    event_counts = (
        train_df.groupby(["item_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={
                "view": "item_event_view_count",
                "addtocart": "item_event_addtocart_count",
                "transaction": "item_event_transaction_count",
            }
        )
        .reset_index()
    )

    item_features = item_features.merge(event_counts, on="item_id", how="left")
    for column in [
        "item_event_view_count",
        "item_event_addtocart_count",
        "item_event_transaction_count",
    ]:
        if column not in item_features.columns:
            item_features[column] = 0.0

    return item_features


def build_user_item_feature_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build optional user-item interaction features from the training history."""

    ordered = train_df.sort_values("timestamp")
    last_event_weight = (
        ordered.groupby(["user_id", "item_id"])["event_weight"]
        .last()
        .rename("user_item_last_event_weight")
        .reset_index()
    )
    seen_count = (
        train_df.groupby(["user_id", "item_id"])
        .size()
        .rename("item_seen_count_for_user")
        .reset_index()
    )
    return seen_count.merge(last_event_weight, on=["user_id", "item_id"], how="outer")


def encode_item_metadata(
    item_features_df: pd.DataFrame | None,
    max_metadata_columns: int = 5,
) -> pd.DataFrame:
    """Encode a small number of item metadata columns into numeric features."""

    if item_features_df is None or item_features_df.empty:
        return pd.DataFrame(columns=["item_id"])

    encoded = item_features_df.copy()
    metadata_columns = [column for column in encoded.columns if column != "item_id"]
    selected_columns = metadata_columns[:max_metadata_columns]

    output = encoded[["item_id"]].copy()
    for column in selected_columns:
        numeric = pd.to_numeric(encoded[column], errors="coerce")
        if numeric.notna().all():
            output[f"meta_{column}"] = numeric.astype(float)
            continue

        filled = encoded[column].fillna("__missing__").astype(str)
        codes, _ = pd.factorize(filled, sort=True)
        output[f"meta_{column}_encoded"] = pd.Series(codes, index=encoded.index).astype(float)

    return output


def add_ranking_features(
    candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    item_features_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach numeric ranking features to candidate rows."""

    features = candidates.copy()

    user_features = build_user_feature_table(train_df)
    item_features = build_item_feature_table(train_df)
    user_item_features = build_user_item_feature_table(train_df)
    metadata_features = encode_item_metadata(item_features_df)

    features = features.merge(user_features, on="user_id", how="left")
    features = features.merge(item_features, on="item_id", how="left")
    features = features.merge(user_item_features, on=["user_id", "item_id"], how="left")
    features = features.merge(metadata_features, on="item_id", how="left")

    optional_zero_fill = [
        "item_seen_count_for_user",
        "user_item_last_event_weight",
        "item_event_view_count",
        "item_event_addtocart_count",
        "item_event_transaction_count",
    ]
    for column in optional_zero_fill:
        if column not in features.columns:
            features[column] = 0.0

    features = features.fillna(0.0)
    feature_columns = get_feature_columns(features)
    features[feature_columns] = features[feature_columns].astype(float)
    return features, feature_columns


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model-ready numeric feature columns."""

    excluded = set(RANKING_METADATA_COLUMNS)
    feature_columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    return feature_columns
