from pandas import Timedelta

from config.constants import SAMPLING_FREQUENCY_HZ as sfreq
from utils.utils import safe_float_to_int


class _Interval:
    def __init__(self, approx_dur: Timedelta = None, exact_dur: Timedelta = None, n_samples: int = None,
                 label: str = None):
        """Represents an interval with an approximate and exact duration, as well as the number of samples based on the
        sampling frequency."""
        self.approx_dur = approx_dur
        self.n_samples = n_samples
        self.exact_dur = exact_dur
        self.label = label

    @staticmethod
    def init_seg(approx_dur: Timedelta, label: str = None) -> "_Interval":
        """Construct the Interval for a segment"""
        # Because the sample frequency is a decimal fraction, the clip and segment lengths will be based on it, so that
        # slices correspond to full indexes
        # The number of points (=samples) in a segment
        n_samples = round(approx_dur.seconds * sfreq)
        exact_dur = Timedelta(seconds=n_samples / sfreq)
        return _Interval(approx_dur, exact_dur, n_samples, label)

    @staticmethod
    def init_non_seg(approx_dur: Timedelta, seg: "_Interval", label: str = None) -> "_Interval":
        """After the segment has been initialized, use this to initialize other Interval's.
        Calculates the number of samples and the exact duration of an approximate duration based on the length of a
        segment and the sampling frequency. This is done so that the number of samples is a whole number."""
        segs_per_window = approx_dur / seg.approx_dur
        assert segs_per_window % 1 == 0, "The approximate length isn't a multiple of the segment length"
        n_samples = safe_float_to_int(segs_per_window * seg.n_samples)
        exact_dur = Timedelta(seconds=n_samples / sfreq)
        return _Interval(approx_dur, exact_dur, n_samples, label)

    def __add__(self, other: "_Interval") -> "_Interval":
        if isinstance(other, _Interval):
            return _Interval(self.approx_dur + other.approx_dur, self.exact_dur + other.exact_dur,
                             self.n_samples + other.n_samples)
        return NotImplemented


SEGMENT = _Interval.init_seg(Timedelta(seconds=15))
CLIP = _Interval.init_non_seg(Timedelta(minutes=10), SEGMENT)

INTER_PRE = _Interval.init_non_seg(Timedelta(minutes=175), SEGMENT, 'inter_pre')
PREICTAL = _Interval.init_non_seg(Timedelta(minutes=60), SEGMENT, 'preictal')
# How much time is between the end of the preictal interval and the seizure
HORIZON = _Interval.init_non_seg(Timedelta(minutes=5), SEGMENT, 'horizon')
POSTICTAL = _Interval.init_non_seg(Timedelta(minutes=60), SEGMENT, 'postictal')
INTER_POST = _Interval.init_non_seg(Timedelta(minutes=180), SEGMENT, 'inter_post')
INTERICTAL = _Interval(label='interictal')

# How much of an offset from a previous szr is necessary for a szr to not be a lead szr
LEAD = INTER_PRE + PREICTAL + HORIZON

CLIPS_PER_PREICTAL_INTERVAL = safe_float_to_int(PREICTAL.n_samples / CLIP.n_samples)
SEGMENTS_PER_PREICTAL_INTERVAL = safe_float_to_int(PREICTAL.n_samples / SEGMENT.n_samples)
SEGMENTS_PER_CLIP = safe_float_to_int(CLIP.n_samples / SEGMENT.n_samples)

if __name__ == "__main__":
    for name in ["SEGMENT", "CLIP", "INTER_PRE", "PREICTAL", "HORIZON", "POSTICTAL", "INTER_POST", "LEAD"]:
        iv = locals()[name]
        print(f'{name} parameters')
        print(f'approx_dur: {iv.exact_dur}')
        print(f'exact_dur : {iv.exact_dur}')
        print(f'n_samples : {iv.n_samples}')
        print()

    print(f'CLIPS_PER_PREICTAL_INTERVAL: {CLIPS_PER_PREICTAL_INTERVAL}')
    print(f'SEGMENTS_PER_PREICTAL_INTERVAL: {SEGMENTS_PER_PREICTAL_INTERVAL}')
    print(f'SEGMENTS_PER_CLIP: {SEGMENTS_PER_CLIP}')
