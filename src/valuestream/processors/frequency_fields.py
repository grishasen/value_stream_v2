"""Stable internal field names shared by frequency-response execution paths."""

from __future__ import annotations

RANK_COLUMN = "__valuestream_frequency_rank"
ROW_ORDER_COLUMN = "__valuestream_frequency_row_order"
POSITIVE_COLUMN = "__valuestream_frequency_positive"
EXPOSED_COLUMN = "__valuestream_frequency_exposed"
SEEN_CURRENT_COLUMN = "__valuestream_frequency_seen_current"
SEEN_HISTORY_COLUMN = "__valuestream_frequency_seen_history"
CONTACT_ORDER_COLUMN = "__valuestream_frequency_contact_order"
EXPOSURE_SEQUENCE_COLUMN = "__valuestream_frequency_exposure_sequence"
WINDOW_START_COLUMN = "__valuestream_frequency_window_start"
BOUNDARY_TIME_COLUMN = "__valuestream_frequency_boundary_time"
BOUNDARY_SEQUENCE_COLUMN = "__valuestream_frequency_boundary_sequence"
RUNNER_PROPENSITY_COLUMN = "__valuestream_runner_propensity"
RUNNER_PRIORITY_COLUMN = "__valuestream_runner_priority"
RUNNER_AVAILABLE_COLUMN = "__valuestream_runner_available"
CHECKPOINT_PARTITION_ORDER_COLUMN = "__valuestream_frequency_checkpoint_partition_order"
CHECKPOINT_LOCAL_ORDER_COLUMN = "__valuestream_frequency_checkpoint_local_order"
CHECKPOINT_SHARD_COLUMN = "__valuestream_checkpoint_shard"
DECISION_DAY_COLUMN = "__valuestream_frequency_decision_day"
PRIOR_EXPOSURES_COLUMN = "__valuestream_frequency_prior_exposures"


__all__ = [
    "BOUNDARY_SEQUENCE_COLUMN",
    "BOUNDARY_TIME_COLUMN",
    "CHECKPOINT_LOCAL_ORDER_COLUMN",
    "CHECKPOINT_PARTITION_ORDER_COLUMN",
    "CHECKPOINT_SHARD_COLUMN",
    "CONTACT_ORDER_COLUMN",
    "DECISION_DAY_COLUMN",
    "EXPOSED_COLUMN",
    "EXPOSURE_SEQUENCE_COLUMN",
    "POSITIVE_COLUMN",
    "PRIOR_EXPOSURES_COLUMN",
    "RANK_COLUMN",
    "ROW_ORDER_COLUMN",
    "RUNNER_AVAILABLE_COLUMN",
    "RUNNER_PRIORITY_COLUMN",
    "RUNNER_PROPENSITY_COLUMN",
    "SEEN_CURRENT_COLUMN",
    "SEEN_HISTORY_COLUMN",
    "WINDOW_START_COLUMN",
]
