from collections import defaultdict


class ApproachDetector:
    def __init__(self):
        # Store depth history for each tracked object
        self.depth_history = defaultdict(list)
        self.confirmed_count = defaultdict(int)

        # Tuning parameters
        self.history_size = 10
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

            self.confirmed_count[track_id] += 1

            if self.confirmed_count[track_id] >= 6:
                return "CONFIRMED_APPROACHING"

            return "APPROACHING"


        elif change < self.away_threshold:

            self.confirmed_count[track_id] = 0
            return "MOVING_AWAY"


        else:

            self.confirmed_count[track_id] = 0
            return "STATIONARY"