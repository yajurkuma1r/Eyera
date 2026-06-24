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

        # Wait until history buffer is full
        if len(history) < self.history_size:
            return "UNKNOWN"

        # Split history into two halves
        first_half = history[:5]
        second_half = history[-5:]

        # Compute average depth for each half
        old_avg = sum(first_half) / len(first_half)
        new_avg = sum(second_half) / len(second_half)

        # Positive = approaching
        # Negative = moving away
        change = new_avg - old_avg

        if change > self.approach_threshold:

            self.confirmed_count[track_id] += 1

            if self.confirmed_count[track_id] >= 4:
                return "CONFIRMED_APPROACHING"

            return "APPROACHING"

        elif change < self.away_threshold:

            self.confirmed_count[track_id] = 0
            return "MOVING_AWAY"

        else:

            self.confirmed_count[track_id] = 0
            return "STATIONARY"