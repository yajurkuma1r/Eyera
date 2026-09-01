from collections import defaultdict


class ApproachDetector:
    """
    Tracks object depth history across frames to detect approaching obstacles.
    """

    def __init__(self):
        self.depth_history = defaultdict(list)
        self.confirmed_count = defaultdict(int)

        self.history_size = 10
        self.approach_threshold = 30
        self.away_threshold = -30

    def update(self, track_id: int, depth: float) -> str:
        history = self.depth_history[track_id]
        history.append(depth)

        if len(history) > self.history_size:
            history.pop(0)

        if len(history) < self.history_size:
            return "UNKNOWN"

        first_half = history[:5]
        second_half = history[-5:]

        old_avg = sum(first_half) / len(first_half)
        new_avg = sum(second_half) / len(second_half)

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
