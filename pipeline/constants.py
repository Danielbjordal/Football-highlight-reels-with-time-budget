"""Shared constants used across pipeline orchestration and assembly."""

ALLOWED_TAGS = {
    "goal",
    "shot",
    "penalty",
    "red_card",
    "redcard",
    "start_phase",
    "end_of_game",
    "free_kick",
}

# Ordering fallbacks/offsets used when clips are stitched into one timeline.
ORDER_FALLBACK = 10_000_000
REACTION_ORDER_OFFSET = 0.0005
REPLAY_ORDER_OFFSET = 0.001
