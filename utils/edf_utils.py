from datetime import datetime, timedelta


def time_to_index(file_start: datetime, timestamp: datetime, sampling_freq_hz: float) -> float:
    """Based on the specified datetime in a recording, calculate the index of that timestamp.
    Note: the index will be returned as a float so that the reverse conversion can be made accurately.
    If actually used to index, round it."""
    time_dif = timestamp - file_start
    return time_dif.total_seconds() * sampling_freq_hz


def index_to_time(start_time: datetime, index: int, sampling_freq_hz: float):
    """Based on the specified index in a recording, calculate the datetime of that index."""
    time_dif = timedelta(seconds=index / sampling_freq_hz)
    return start_time + time_dif
