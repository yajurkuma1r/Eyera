from collections import defaultdict


class ApproachDetector:
    def __init__(self):
        # Store depth history for each tracked object
        self.depth_history = defaultdict(list)

        # Tuning parameters
        self.history_size = 5
        self.approach_threshold = 30
        self.away_threshold = -30

    def update(self, track_id: int, depth: float) -> str:
        history = self.depth_history[track_id]

        # Add new depth reading
        history.append(depth)

        # Keep only last N readings
        if len(history) > self.history_size:
            history.pop(0)

        # Need enough data before making decisions
        if len(history) < 3:
            return "UNKNOWN"

        oldest = history[0]
        newest = history[-1]

        change = newest - oldest

        if change > self.approach_threshold:
            return "APPROACHING"

        elif change < self.away_threshold:
            return "MOVING_AWAY"

        else:
            return "STATIONARY"