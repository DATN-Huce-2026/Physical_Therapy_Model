"""Tach tung lan lap (rep) tu chuoi goc khop da lam muot."""

from statistics import median


class RepSegmenter:
    """Phat hien mot rep hoan chinh: bien A -> bien B -> tro lai bien A."""

    def __init__(self, window=5, min_distance=12, min_amplitude=8.0):
        self.window = window
        self.min_distance = min_distance
        self.min_amplitude = min_amplitude

    def find_reps(self, samples, joint_names):
        """Tra ve [(start_frame, turning_frame, end_frame), ...]."""
        signal = self._representative_signal(samples, joint_names)
        extrema = self._filtered_extrema(signal)
        reps = []
        index = 0
        while index + 2 < len(extrema):
            start, middle, end = extrema[index:index + 3]
            if start[1] == end[1] and start[1] != middle[1]:
                reps.append((start[0], middle[0], end[0]))
                index += 2
            else:
                index += 1
        return reps

    @staticmethod
    def _representative_signal(samples, joint_names):
        signal = []
        for sample in samples:
            values = [sample.get(name) for name in joint_names]
            values = [value for value in values if value is not None]
            signal.append(median(values) if values else None)
        return signal

    def _filtered_extrema(self, signal):
        candidates = []
        for i, value in enumerate(signal):
            if value is None:
                continue
            left = [v for v in signal[max(0, i - self.window):i] if v is not None]
            right = [v for v in signal[i + 1:i + self.window + 1] if v is not None]
            if not left or not right:
                continue
            if value >= max(left) and value >= max(right):
                candidates.append((i, "max", value))
            elif value <= min(left) and value <= min(right):
                candidates.append((i, "min", value))

        compact = []
        for item in candidates:
            if compact and item[1] == compact[-1][1] and item[0] - compact[-1][0] < self.min_distance:
                if (item[1] == "max" and item[2] > compact[-1][2]) or (item[1] == "min" and item[2] < compact[-1][2]):
                    compact[-1] = item
                continue
            compact.append(item)

        filtered = []
        for item in compact:
            if not filtered:
                filtered.append(item)
                continue
            previous = filtered[-1]
            if item[1] != previous[1] and abs(item[2] - previous[2]) >= self.min_amplitude:
                filtered.append(item)
        return filtered
